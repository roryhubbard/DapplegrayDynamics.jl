# Reduced Hessian SQP Method (Boggs-Tolle 1996, Section 3.4)
# Sequential Quadratic Programming using null space decomposition

using LinearAlgebra
using Printf

"""
    ReducedHessianSettings{T}

Settings for the reduced Hessian SQP algorithm.
"""
@kwdef struct ReducedHessianSettings{T<:AbstractFloat}
    max_iter::Int = 100
    verbose::Bool = true

    # Convergence tolerances
    optimality_tol::T = 1e-6
    feasibility_tol::T = 1e-6

    # Line search parameters
    ls_max_iter::Int = 20
    ls_α_initial::T = 1.0
    ls_β::T = 0.5  # Backtracking factor
    ls_c1::T = 1e-4  # Armijo parameter
end

"""
    ReducedHessianSQPSolver{T}

Reduced Hessian SQP solver using null space decomposition with exact Hessian.

# Algorithm (Boggs-Tolle 1996, Section 3.4):
1. Compute exact Lagrangian Hessian: ∇²L = ∇²f + ∇²h + ∇²g
2. Decompose: dx = Z_k p_z + Y_k p_y
3. Range space: p_y = -(∇h^T Y)^{-1} h  (linear solve)
4. Null space: p_z = -R^{-1} Z^T[∇f + ∇²L Y p_y]  where R = Z^T ∇²L Z
5. Line search: α ∈ (0,1] to reduce merit function
6. Update: x_{k+1} = x_k + α dx
"""
struct ReducedHessianSQPSolver{T}
    mechanism::Mechanism{T}
    f::AbstractVector{<:AdjacentKnotPointsFunction}  # Objectives
    h::AbstractVector{<:AdjacentKnotPointsFunction}  # Equality constraints
    g::AbstractVector{<:AdjacentKnotPointsFunction}  # Inequality constraints

    x::DiscreteTrajectory{T,T}      # Primal variables
    λ::AbstractVector{T}             # Equality multipliers
    ν::AbstractVector{T}             # Inequality multipliers

    settings::ReducedHessianSettings{T}
    guts::Dict{Symbol,Any}           # Iteration history
end

"""
    ReducedHessianSQPSolver(mechanism, objectives, equalities, inequalities,
                           x0, λ0, ν0, settings)

Construct a Reduced Hessian SQP solver.
"""
function ReducedHessianSQPSolver(
    mechanism::Mechanism{T},
    objectives::AbstractVector{<:AdjacentKnotPointsFunction},
    equalities::AbstractVector{<:AdjacentKnotPointsFunction},
    inequalities::AbstractVector{<:AdjacentKnotPointsFunction},
    x0::DiscreteTrajectory{T,T},
    λ0::AbstractVector{T},
    ν0::AbstractVector{T},
    settings::ReducedHessianSettings{T}
) where T

    guts = Dict{Symbol,Any}()

    return ReducedHessianSQPSolver(
        mechanism, objectives, equalities, inequalities,
        x0, λ0, ν0, settings, guts
    )
end

"""
    compute_null_range_bases(∇h::Matrix{T}) -> (Z, Y)

Compute orthonormal bases for null space and range space of ∇h^T.

Uses QR factorization: if ∇h = [Y R; 0 0] then
- Y: range space basis (first m columns of Q)
- Z: null space basis (last n-m columns of Q)

# Returns
- Z: n × (n-m) matrix, columns span null(∇h^T)
- Y: n × m matrix, columns span range(∇h)
"""
function compute_null_range_bases(∇h::Matrix{T}) where T
    n, m = size(∇h)  # n variables, m constraints

    # QR factorization of ∇h
    Q, R = qr(∇h)
    Q = Matrix(Q)  # Convert to dense for easier indexing

    # Y = first m columns (range space)
    # Z = last (n-m) columns (null space)
    Y = Q[:, 1:m]
    Z = Q[:, m+1:end]

    return Z, Y
end

"""
    compute_range_space_step(∇h, Y, h) -> p_y

Compute range space component to satisfy linearized constraints.

Solves: ∇h^T Y p_y = -h
"""
function compute_range_space_step(∇h::Matrix{T}, Y::Matrix{T}, h::Vector{T}) where T
    # p_y = -(∇h^T Y)^{-1} h
    p_y = -(∇h' * Y) \ h
    return p_y
end

"""
    compute_null_space_step(Z, B, ∇f, Y, p_y) -> p_z

Compute null space component via reduced Newton step.

Minimizes in null space: ∇f_reduced^T p_z + (1/2) p_z^T R p_z
where:
- R = Z^T B Z (reduced Hessian)
- ∇f_reduced = Z^T [∇f + B Y p_y]

Solution: p_z = -R^{-1} ∇f_reduced
"""
function compute_null_space_step(
    Z::Matrix{T},
    B::Matrix{T},
    ∇f::Vector{T},
    Y::Matrix{T},
    p_y::Vector{T}
) where T
    # Reduced Hessian: R = Z^T B Z
    R = Z' * B * Z

    # Reduced gradient: g = Z^T [∇f + B Y p_y]
    g_reduced = Z' * (∇f + B * (Y * p_y))

    # Solve: R p_z = -g_reduced
    p_z = -R \ g_reduced

    return p_z
end

"""
    merit_function_l1(x, h_vals, g_vals, f_val, ρ)

ℓ₁ exact penalty function: φ(x) = f(x) + ρ(||h(x)||₁ + ||g⁻(x)||₁)

where g⁻(x) = max(0, -g(x))
"""
function merit_function_l1(f_val::T, h_vals::Vector{T}, g_vals::Vector{T}, ρ::T) where T
    constraint_violation = sum(abs, h_vals) + sum(x -> max(zero(T), -x), g_vals)
    return f_val + ρ * constraint_violation
end

"""
    backtracking_line_search(x, dx, f, h, g, ρ, settings)

Armijo backtracking line search on ℓ₁ merit function.

Finds α ∈ (0,1] such that:
    φ(x + α dx) ≤ φ(x) + c₁ α Dφ(x; dx)

where Dφ is the directional derivative (approximated).
"""
function backtracking_line_search(
    x::DiscreteTrajectory{T,T},
    dx::Vector{T},
    f::AbstractVector{<:AdjacentKnotPointsFunction},
    h::AbstractVector{<:AdjacentKnotPointsFunction},
    g::AbstractVector{<:AdjacentKnotPointsFunction},
    ρ::T,
    settings::ReducedHessianSettings{T}
) where T

    α = settings.ls_α_initial

    # Current merit function value
    f_curr = sum(f_func(x) for f_func in f)
    h_curr = vcat([h_func(x) for h_func in h]...)
    g_curr = vcat([g_func(x) for g_func in g]...)
    φ_curr = merit_function_l1(f_curr, h_curr, g_curr, ρ)

    # Directional derivative (approximate)
    # For ℓ₁ penalty: Dφ ≈ ∇f^T dx - ρ ||h||₁
    Dφ = -ρ * sum(abs, h_curr)  # Simplified

    for iter in 1:settings.ls_max_iter
        # Trial point
        x_trial = copy(x)
        x_trial.knotpoints .+= α * dx

        # Evaluate merit function
        f_trial = sum(f_func(x_trial) for f_func in f)
        h_trial = vcat([h_func(x_trial) for h_func in h]...)
        g_trial = vcat([g_func(x_trial) for g_func in g]...)
        φ_trial = merit_function_l1(f_trial, h_trial, g_trial, ρ)

        # Armijo condition
        if φ_trial <= φ_curr + settings.ls_c1 * α * Dφ
            return α, φ_trial
        end

        # Backtrack
        α *= settings.ls_β
    end

    # If line search fails, return small step
    return settings.ls_β^settings.ls_max_iter, φ_curr
end

"""
    solve!(solver::ReducedHessianSQPSolver)

Main SQP iteration loop using reduced Hessian method with exact Hessian.
"""
function solve!(solver::ReducedHessianSQPSolver{T}) where T
    settings = solver.settings

    # Initialize history
    solver.guts[:objective] = T[]
    solver.guts[:constraint_violation] = T[]
    solver.guts[:optimality] = T[]
    solver.guts[:step_size] = T[]

    # Penalty parameter (adaptive, needs to be larger than max dual)
    ρ = 10.0 * one(T)

    for k in 1:settings.max_iter
        x_k = solver.x
        λ_k = solver.λ
        ν_k = solver.ν

        # Evaluate objective: value, gradient, Hessian
        f_val, ∇f, ∇²f = super_hessian_objective(solver.f, x_k)

        # Evaluate equality constraints: values, Jacobian, Hessian
        h_val, Jh, ∇²h = super_hessian_constraints(solver.h, x_k, λ_k)

        # Evaluate inequality constraints: values, Jacobian, Hessian
        g_val, Jg, ∇²g = super_hessian_constraints(solver.g, x_k, ν_k)

        # Exact Lagrangian Hessian
        ∇²L = ∇²f + ∇²h + ∇²g

        # Compute constraint violation
        constraint_violation = norm(h_val, 1) + sum(x -> max(zero(T), -x), g_val)

        # Compute KKT optimality (gradient of Lagrangian)
        ∇L = ∇f + Jh' * λ_k + Jg' * ν_k
        optimality = norm(∇L, Inf)

        # Store metrics
        push!(solver.guts[:objective], f_val)
        push!(solver.guts[:constraint_violation], constraint_violation)
        push!(solver.guts[:optimality], optimality)

        if settings.verbose
            @printf("Iter %3d: f = %.6e, ||h|| = %.6e, opt = %.6e\n",
                    k, f_val, constraint_violation, optimality)
        end

        # Check convergence
        if constraint_violation < settings.feasibility_tol && optimality < settings.optimality_tol
            settings.verbose && println("Converged!")
            return solver
        end

        # Check if we have equality constraints
        if isempty(h_val)
            # No equality constraints - just use Newton step
            dx = -∇²L \ ∇f
        else
            # Compute null and range space bases
            # Note: Jh is the Jacobian (m × n), so ∇h = Jh' (n × m)
            ∇h = Matrix(Jh')
            Z, Y = compute_null_range_bases(∇h)

            # Range space step (satisfies linearized constraints)
            p_y = compute_range_space_step(∇h, Y, h_val)

            # Null space step (minimizes in reduced space with exact Hessian)
            # Uses R = Z^T ∇²L Z as the reduced Hessian
            p_z = compute_null_space_step(Z, ∇²L, ∇f, Y, p_y)

            # Total step: dx = Z p_z + Y p_y
            dx = Z * p_z + Y * p_y
        end

        # Line search with merit function
        α, φ_new = backtracking_line_search(
            x_k, dx, solver.f, solver.h, solver.g, ρ, settings
        )

        push!(solver.guts[:step_size], α)

        if settings.verbose
            @printf("  Step size: α = %.6e\n", α)
        end

        # Update primal variables
        kp = knotpoints(solver.x)
        kp .+= α * dx

        # Update dual variables using least-squares multiplier estimate
        # λ_{k+1} = (Jh Jh')^{-1} Jh ∇f_{k+1}
        # For now, keep multipliers fixed (can be improved with proper dual update)

        # Update penalty parameter if needed (ensure ρ > ||λ||_∞ and ||ν||_∞)
        if !isempty(λ_k)
            ρ = max(ρ, 2 * norm(λ_k, Inf))
        end
        if !isempty(ν_k)
            ρ = max(ρ, 2 * norm(ν_k, Inf))
        end
    end

    settings.verbose && println("Max iterations reached without convergence")
    return solver
end

# Accessors
primal(solver::ReducedHessianSQPSolver) = solver.x
equality_multipliers(solver::ReducedHessianSQPSolver) = solver.λ
inequality_multipliers(solver::ReducedHessianSQPSolver) = solver.ν
