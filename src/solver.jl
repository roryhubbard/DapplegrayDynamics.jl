@kwdef struct OuterSettings{T <: AbstractFloat}
    max_iter::UInt32    	= 10
    time_limit::Float64     = Inf
    verbose::Bool           = true
    max_step_fraction::T    = 0.99
end

OuterSettings(args...) = OuterSettings{Float64}(args...)

struct SQPSolver{T}
    mechanism::Mechanism{T}
    f::AbstractVector{<:AdjacentKnotPointsFunction}
    g::AbstractVector{<:AdjacentKnotPointsFunction}
    h::AbstractVector{<:AdjacentKnotPointsFunction}
    x::DiscreteTrajectory{T,T}
    λ::AbstractVector{T}
    v::AbstractVector{T}
    inner_settings::Clarabel.Settings{T}
    outer_settings::OuterSettings{T}
    guts::Dict{Symbol,Any}

    function SQPSolver(
        mechanism::Mechanism{T},
        f::AbstractVector{<:AdjacentKnotPointsFunction},
        g::AbstractVector{<:AdjacentKnotPointsFunction},
        h::AbstractVector{<:AdjacentKnotPointsFunction},
        x::DiscreteTrajectory{T,T},
        λ::Union{AbstractVector{T},Nothing} = nothing,
        v::Union{AbstractVector{T},Nothing} = nothing,
        inner_settings::Union{Clarabel.Settings{T},Nothing} = nothing,
        outer_settings::Union{OuterSettings{T},Nothing} = nothing,
    ) where {T}
        if isnothing(λ)
            λ = zeros(T, num_lagrange_multipliers(g))
        end
        if isnothing(v)
            v = zeros(T, num_lagrange_multipliers(h))
        end
        if isnothing(inner_settings)
            inner_settings = Clarabel.Settings()
        end
        if isnothing(outer_settings)
            outer_settings = OuterSettings()
        end

        ng = num_lagrange_multipliers(g)
        @assert length(λ) == ng "inequality constraint lagrange multipliers vector must have length $(ng) but has $(length(λ))"
        nh = num_lagrange_multipliers(h)
        @assert length(v) == nh "equality constraint lagrange multipliers vector must have length $(nh) but has $(length(v))"

        new{T}(mechanism, f, g, h, x, λ, v, inner_settings, outer_settings, Dict{Symbol,Any}())
    end
end

objectives(solver::SQPSolver) = solver.f

inequality_constraints(solver::SQPSolver) = solver.g

equality_constraints(solver::SQPSolver) = solver.h

inequality_duals(solver::SQPSolver) = solver.λ

equality_duals(solver::SQPSolver) = solver.v

primal(solver::SQPSolver) = solver.x

get_inner_settings(solver::SQPSolver) = solver.inner_settings

get_outer_settings(solver::SQPSolver) = solver.outer_settings

function initialize_trajectory(
    mechanism::Mechanism{T},
    N::Int,
    tf::T,
    nu::Int,
    q₀::AbstractVector{T},
    q₁::AbstractVector{T},
    v₀::AbstractVector{T},
    v₁::AbstractVector{T},
) where {T}
    nq = num_positions(mechanism)
    nv = num_velocities(mechanism)

    ts, qs, vs = straight_line_trajectory(N, tf, q₀, q₁, v₀, v₁)

    N = length(ts)
    nx = nq + nv
    knotpointsize = nx + nu
    num_decision_variables = N * knotpointsize
    zero_control_vector = zeros(nu)

    timesteps = diff(ts)
    # timesteps needs to be the same length as timestamps
    push!(timesteps, last(timesteps))

    knotpoints = Vector{T}(undef, num_decision_variables)

    for i = 1:N
        idx₀ = (i - 1) * knotpointsize + 1
        idx₁ = idx₀ + knotpointsize - 1
        knotpoints[idx₀:idx₁] = [qs[i]; vs[i]; zero_control_vector]
    end

    DiscreteTrajectory(ts, timesteps, knotpoints, knotpointsize, nx)
end

function num_lagrange_multipliers(constraints::AbstractVector{<:AdjacentKnotPointsFunction})
    sum(outputdim(c) * length(indices(c)) for c ∈ constraints)
end

function evaluate_objective(
    objectives::AbstractVector{<:AdjacentKnotPointsFunction},
    Z::DiscreteTrajectory,
)
    sum(objective(Val(Sum), Z) for objective ∈ objectives)
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
    for constraint ∈ constraints
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

function super_hessian_objective(
    objectives::AbstractVector{<:AdjacentKnotPointsFunction},
    Z::DiscreteTrajectory,
)
    z = knotpoints(Z)
    # Rest assured, no copying happening here
    fwrapped(z) = evaluate_objective(
        objectives,
        DiscreteTrajectory(time(Z), timesteps(Z), z, knotpointsize(Z), nstates(Z)),
    )
    result = DiffResults.HessianResult(z)
    result = ForwardDiff.hessian!(result, fwrapped, z)
    DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
end

function super_hessian_constraints(
    constraints::AbstractVector{<:AdjacentKnotPointsFunction},
    Z::DiscreteTrajectory,
    λ::AbstractVector{T},
) where {T}
    z = knotpoints(Z)
    n = length(z)
    m = sum(length(indices(con)) * outputdim(con) for con ∈ constraints)
    y = zeros(T, m * n)
    H = DiffResults.JacobianResult(y, z)

    fwrapped(z) = evaluate_constraints(
        constraints,
        DiscreteTrajectory(time(Z), timesteps(Z), z, knotpointsize(Z), nstates(Z)),
    )

    # TODO: can't use ForwardDiff.jacobian! for innner jacobian
    # https://github.com/JuliaDiff/ForwardDiff.jl/issues/393
    H = ForwardDiff.jacobian!(H, z -> ForwardDiff.jacobian(fwrapped, z), z)

    # Outer Jacobian as 3-tensor: (i,j,k) = (output, ∂/∂z_j, ∂/∂z_k)
    G3 = reshape(DiffResults.jacobian(H), m, n, n)
    H3 = PermutedDimsArray(G3, (2, 3, 1))  # (n, n, m), one Hessian per output

    @assert length(λ) == size(H3, 3) "length(λ)=$(length(λ)) ≠ Hessian stack depth $(size(H3, 3))"

    ∑H = zeros(T, n, n)
    # λ-weighted sum of Hessians (no mutation of H3)
    @inbounds @views for k = 1:length(λ)
        ∑H .+= λ[k] .* H3[:, :, k]
    end
    # numeric symmetrization before wrapping to handle autodiff noise, maybe not
    # necessary?
    #    ∑H .= (∑H .+ ∑H') .* T(0.5)

    cval = evaluate_constraints(constraints, Z)
    Jmn = reshape(DiffResults.value(H), m, n)

    cval, Jmn, Symmetric(∑H)
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
    inner_settings::Clarabel.Settings{T},
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
    K = [Clarabel.NonnegativeConeT(length(g)), Clarabel.ZeroConeT(length(h))]

    if inner_settings.verbose
        println("P $(size(P)): ", P)
        println("q $(size(q)): ", q)
        println("A $(size(A)): ", A)
        println("b $(size(b)): ", b)
        println("K $(size(K)): ", K)
    end

    solver = Clarabel.Solver(P, q, A, b, K, inner_settings)
    solution = Clarabel.solve!(solver)
    # solution.x → primal solution
    # solution.z → dual solution
    # solution.s → slacks
    (solution.x, solution.z)
end

function solve!(
    solver::SQPSolver{T},
    custom_gradients::Bool = false,
    expose_guts::Bool = true,
) where {T}
    inner_settings = get_inner_settings(solver)
    outer_settings = get_outer_settings(solver)
    for k = 1:outer_settings.max_iter
        x = primal(solver)
        λ = inequality_duals(solver)
        v = equality_duals(solver)

        if custom_gradients
            f = evaluate_objective(objectives(solver), primal(solver))
            ▽f = gradient(Val(Sum), objectives(solver), primal(solver))
            ▽²f = hessian(objectives(solver), primal(solver))

            g = evaluate_constraints(inequality_constraints(solver), primal(solver))
            Jg = jacobian(inequality_constraints(solver), primal(solver))
            ▽²g = vector_hessian(inequality_constraints(solver), primal(solver), λ)

            h = evaluate_constraints(equality_constraints(solver), primal(solver))
            Jh = jacobian(equality_constraints(solver), primal(solver))
            ▽²h = vector_hessian(equality_constraints(solver), primal(solver), v)
        else
            f, ▽f, ▽²f = super_hessian_objective(objectives(solver), primal(solver))
            g, Jg, ▽²g =
                super_hessian_constraints(inequality_constraints(solver), primal(solver), λ)
            h, Jh, ▽²h =
                super_hessian_constraints(equality_constraints(solver), primal(solver), v)
        end

        L = f + λ' * g + v' * h
        ▽L = ▽f + Jg' * λ + Jh' * v
        ▽²L = ▽²f + ▽²g + ▽²h

        # Add regularization to ensure positive definiteness
        ϵ = 1e-6
        ▽²L += ϵ * I

        negate!(Jg)
        negate!(Jh)

        pₖ, lₖ = solve_qp(g, Jg, h, Jh, ▽L, ▽²L, inner_settings)

        if expose_guts
            push!(
                get!(solver.guts, :primal, Vector{DiscreteTrajectory{T,T}}()),
                deepcopy(x),
            )
            push!(get!(solver.guts, :inequality_duals, Vector{Vector{T}}()), deepcopy(λ))
            push!(get!(solver.guts, :equality_duals, Vector{Vector{T}}()), deepcopy(v))
            push!(get!(solver.guts, :objective, Vector{T}()), deepcopy(f))
            push!(get!(solver.guts, :lagrangian, Vector{T}()), deepcopy(L))
        end

        # solution step
        α = outer_settings.max_step_fraction
        knotpoints(primal(solver)) .+= α .* pₖ
        inequality_duals(solver) .+= α .* @view lₖ[1:length(g)]
        equality_duals(solver) .+= α .* @view lₖ[(length(g)+1):end]

        if expose_guts && k == outer_settings.max_iter
            push!(get!(solver.guts, :primal, Vector{DiscreteTrajectory{T,T}}()), x)
            push!(get!(solver.guts, :inequality_duals, Vector{Vector{T}}()), λ)
            push!(get!(solver.guts, :equality_duals, Vector{Vector{T}}()), v)
            push!(get!(solver.guts, :objective, Vector{T}()), f)
            push!(get!(solver.guts, :lagrangian, Vector{T}()), L)
        end

        if outer_settings.verbose
            println("primal x $(length(knotpoints(x))): ", x)
            println("dual λ $(length(λ)): ", λ)
            println("dual v $(length(v)): ", v)

            println("f $(length(f)): ", f)
            println("▽f $(size(▽f)): ", ▽f)
            println("▽²f $(size(▽²f)): ", ▽²f)

            println("g $(size(g)): ", g)
            println("Jg $(size(Jg)): ", Jg)
            println("▽²g $(size(▽²g)): ", ▽²g)

            println("h $(size(h)): ", h)
            println("Jh $(size(Jh)): ", Jh)
            println("▽²h $(size(▽²h)): ", ▽²h)

            println("L $(size(L)): ", L)
            println("▽L $(size(▽L)): ", ▽L)
            println("▽²L $(size(▽²L)): ", ▽²L)

            println("QP primal pₖ $(length(pₖ)): ", pₖ)
            println("QP dual lₖ $(length(lₖ)): ", lₖ)
        end
    end
end
