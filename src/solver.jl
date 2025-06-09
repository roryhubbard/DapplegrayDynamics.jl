struct SQPSolver{T}
    mechanism::Mechanism{T}
    f::AbstractVector{<:AdjacentKnotPointsFunction}
    g::AbstractVector{<:AdjacentKnotPointsFunction}
    h::AbstractVector{<:AdjacentKnotPointsFunction}
    x::DiscreteTrajectory{T}
    λ::AbstractVector{T}
    v::AbstractVector{T}

    function SQPSolver(
        mechanism::Mechanism{T},
        f::AbstractVector{<:AdjacentKnotPointsFunction},
        g::AbstractVector{<:AdjacentKnotPointsFunction},
        h::AbstractVector{<:AdjacentKnotPointsFunction},
        x::DiscreteTrajectory{T,T},
        λ::Union{AbstractVector{T}, Nothing} = nothing,
        v::Union{AbstractVector{T}, Nothing} = nothing,
    ) where {T}
        if isnothing(λ)
            λ = zeros(T, num_lagrange_multipliers(g))
        end
        if isnothing(v)
            v = zeros(T, num_lagrange_multipliers(h))
        end

        ng = num_lagrange_multipliers(g)
        @assert length(λ) == ng "inequality constraint lagrange multipliers vector must have length $(ng) but has $(length(λ))"
        nh = num_lagrange_multipliers(h)
        @assert length(v) == nh "equality constraint lagrange multipliers vector must have length $(nh) but has $(length(v))"

        new{T}(mechanism, f, g, h, x, λ, v)
    end
end

objectives(solver::SQPSolver) = solver.f

inequality_constraints(solver::SQPSolver) = solver.g

equality_constraints(solver::SQPSolver) = solver.h

inequality_duals(solver::SQPSolver) = solver.λ

equality_duals(solver::SQPSolver) = solver.v

primal(solver::SQPSolver) = solver.x

function initialize_trajectory(mechanism::Mechanism{T}, tf::T, Δt::T, nu::Int) where {T}
    ts, qs, vs = simulate_mechanism(mechanism, tf, Δt, zeros(T, 2), zeros(T, 2))

    N = length(ts)
    nx = num_positions(mechanism) + num_velocities(mechanism)
    knotpointsize = nx + nu
    num_decision_variables = N * knotpointsize
    zero_control_vector = zeros(nu)

    timesteps = fill(T(Δt), N)
    knotpoints = Vector{T}(undef, num_decision_variables)

    for i = 1:N
        idx₀ = (i - 1) * knotpointsize + 1
        idx₁ = idx₀ + knotpointsize - 1
        knotpoint = [qs[i]; vs[i]; zero_control_vector]
        knotpoints[idx₀:idx₁] = knotpoint
    end

    DiscreteTrajectory(ts, timesteps, knotpoints, knotpointsize, nx)
end

function num_lagrange_multipliers(constraints::AbstractVector{<:AdjacentKnotPointsFunction})
    sum(outputdim(c) * length(indices(c)) for c in constraints)
end

function evaluate_objective(
    objectives::AbstractVector{<:AdjacentKnotPointsFunction},
    Z::DiscreteTrajectory,
)
    sum(objective(Val(Sum), Z) for objective in objectives)
end

function super_gradient(
    objectives::AbstractVector{<:AdjacentKnotPointsFunction},
    Z::DiscreteTrajectory,
)
    z = knotpoints(Z)
    # Rest assured, no copying happening here
    fwrapped(z) = evaluate_objective(
        objectives,
        DiscreteTrajectory(time(Z), timesteps(Z), z, knotpointsize(Z), nstates(Z)),
    )
    ForwardDiff.gradient(fwrapped, z)
end

function evaluate_constraints(
    constraints::AbstractVector{<:AdjacentKnotPointsFunction},
    Z::DiscreteTrajectory{Ts,Tk},
) where {Ts,Tk}
    # TODO: preallocate before here
    result = Vector{Tk}()
    for constraint in constraints
        val = constraint(Val(Stack), Z)
        append!(result, val)
    end
    return result
end

function super_jacobian(
    constraints::AbstractVector{<:AdjacentKnotPointsFunction},
    Z::DiscreteTrajectory{Ts,Tk},
) where {Ts,Tk}
    z = knotpoints(Z)
    # Rest assured, no copying happening here
    fwrapped(z) = evaluate_constraints(
        constraints,
        DiscreteTrajectory(time(Z), timesteps(Z), z, knotpointsize(Z), nstates(Z)),
    )
    ForwardDiff.jacobian(fwrapped, z)
end

negate!(x::AbstractArray) = x .*= -1

"""
Solve QP using Clarabel

minimize   1⁄2𝒙ᵀ𝑷𝒙 + 𝒒ᵀ𝒙
subject to  𝑨𝒙 + 𝒔 = 𝒃
                 𝒔 ∈ 𝑲
with decision variables 𝒙 ∈ ℝⁿ, 𝒔 ∈ 𝑲 and data matrices 𝑷 = 𝑷ᵀ ≥ 0,
𝒒 ∈ ℝⁿ, 𝑨 ∈ ℝᵐˣⁿ, and b ∈ ℝᵐ. The convext set 𝑲 is a composition of convex cones.
"""
function solve_qp(
    g::AbstractVector{T},
    Jg::AbstractMatrix{T},
    h::AbstractVector{T},
    Jh::AbstractMatrix{T},
    ▽L::AbstractVector{T},
    ▽²L::AbstractMatrix{T},
) where {T}
    P = sparse(▽²L)
    q = ▽L
    A = sparse([
        Jg;
        Jh;
    ])
    b = [
        g;
        h
    ]
    K = [Clarabel.ZeroConeT(length(h)), Clarabel.NonnegativeConeT(length(g))]

    println("P $(size(P)): ", P)
    println("q $(size(q)): ", q)
    println("A $(size(A)): ", A)
    println("b $(size(b)): ", b)
    println("K $(size(K)): ", K)

    settings = Clarabel.Settings()
    solver = Clarabel.Solver(P, q, A, b, K, settings)
    solution = Clarabel.solve!(solver)
    # solution.x → primal solution
    # solution.z → dual solution
    # solution.s → slacks
    (solution.x, solution.z)
end

function solve!(solver::SQPSolver{T}) where {T}
    for k = 1:1
        x = primal(solver)
        println("primal x: ", x)

        λ = inequality_duals(solver)
        println("dual λ: ", λ)

        v = equality_duals(solver)
        println("dual v: ", v)

        f = evaluate_objective(objectives(solver), primal(solver))
        println("f: ", f)

        g = evaluate_constraints(inequality_constraints(solver), primal(solver))
        println("g $(size(g)): ", g)

        h = evaluate_constraints(equality_constraints(solver), primal(solver))
        println("h $(size(h)): ", h)

        ▽f = gradient(Val(Sum), objectives(solver), primal(solver))
        println("▽f $(size(▽f)): ", ▽f)

        Jg = jacobian(inequality_constraints(solver), primal(solver))
        println("Jg $(size(Jg)): ", Jg)

        Jh = jacobian(equality_constraints(solver), primal(solver))
        println("Jh $(size(Jh)): ", Jh)

        L = f + λ' * g + v' * h
        println("L $(size(L)): ", L)

        ▽L = ▽f + Jg' * λ + Jh' * v
        println("▽L $(size(▽L)): ", ▽L)

        ▽²f = hessian(objectives(solver), primal(solver))
        println("▽²f $(size(▽²f)): ", ▽²f)

        ▽²g = vector_hessian(inequality_constraints(solver), primal(solver), λ)
        println("▽²g $(size(▽²g)): ", ▽²g)

        ▽²h = vector_hessian(equality_constraints(solver), primal(solver), v)
        println("▽²h $(size(▽²h)): ", ▽²h)

        ▽²L = ▽²f + ▽²g + ▽²h
        println("▽²L: ", ▽²L)

        negate!(Jg)
        negate!(Jh)
        pₖ, lₖ = solve_qp(g, Jg, h, Jh, ▽L, ▽²L)
        println("QP primal pₖ $(length(pₖ)): ", pₖ)
        println("QP dual lₖ $(length(lₖ)): ", lₖ)

        # solution step
        knotpoints(primal(solver)) .+= pₖ
        inequality_duals(solver) .+= @view lₖ[1:length(g)]
        equality_duals(solver) .+= @view lₖ[length(g)+1:end]

        #        𝚫𝒙ₖ₊₁, 𝒗ₖ₊₁, 𝝀ₖ₊₁ = unpack_result(result)
        #        nudge_𝒙!(solver, 𝚫𝒙ₖ₊₁)
        #        set_𝒗!(solver, 𝒗ₖ₊₁)
        #        set_𝝀!(solver, 𝝀ₖ₊₁)
    end
end
