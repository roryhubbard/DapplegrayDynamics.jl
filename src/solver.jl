struct SQPSolver{T}
    mechanism::Mechanism{T}
    f::AbstractVector{<:AdjacentKnotPointsFunction}
    g::AbstractVector{<:AdjacentKnotPointsFunction}
    h::AbstractVector{<:AdjacentKnotPointsFunction}
    x::DiscreteTrajectory{T}
    λ::AbstractVector{T}
    v::AbstractVector{T}
    settings::Clarabel.Settings{T}

    function SQPSolver(
        mechanism::Mechanism{T},
        f::AbstractVector{<:AdjacentKnotPointsFunction},
        g::AbstractVector{<:AdjacentKnotPointsFunction},
        h::AbstractVector{<:AdjacentKnotPointsFunction},
        x::DiscreteTrajectory{T,T},
        λ::Union{AbstractVector{T},Nothing} = nothing,
        v::Union{AbstractVector{T},Nothing} = nothing,
        settings::Union{Clarabel.Settings{T},Nothing} = nothing,
    ) where {T}
        if isnothing(λ)
            λ = zeros(T, num_lagrange_multipliers(g))
        end
        if isnothing(v)
            v = zeros(T, num_lagrange_multipliers(h))
        end
        if isnothing(settings)
            settings = Clarabel.Settings(
                max_iter = 10,
                time_limit = 60,
                verbose = true,
                max_step_fraction = 0.99,
                tol_gap_abs = 1e-8,
                tol_gap_rel = 1e-8,
            )
        end

        ng = num_lagrange_multipliers(g)
        @assert length(λ) == ng "inequality constraint lagrange multipliers vector must have length $(ng) but has $(length(λ))"
        nh = num_lagrange_multipliers(h)
        @assert length(v) == nh "equality constraint lagrange multipliers vector must have length $(nh) but has $(length(v))"

        new{T}(mechanism, f, g, h, x, λ, v, settings)
    end
end

objectives(solver::SQPSolver) = solver.f

inequality_constraints(solver::SQPSolver) = solver.g

equality_constraints(solver::SQPSolver) = solver.h

inequality_duals(solver::SQPSolver) = solver.λ

equality_duals(solver::SQPSolver) = solver.v

primal(solver::SQPSolver) = solver.x

get_settings(solver::SQPSolver) = solver.settings

function initialize_trajectory(
    mechanism::Mechanism{T},
    N::Int,
    tf::T,
    nu::Int,
    q₀::AbstractVector{T},
    q₁::AbstractVector{T},
    v₀::AbstractVector{T},
    v₁::AbstractVector{T},
    add_midpoints::Bool = false,
) where {T}
    nq = num_positions(mechanism)
    nv = num_velocities(mechanism)

    ts, qs, vs = straight_line_trajectory(N, tf, q₀, q₁, v₀, v₁, add_midpoints)

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

    ∑H = zeros(T, n, n)
    H3 = reshape(DiffResults.jacobian(H)', n, n, :)
    @assert length(λ) == size(H3, 3) "length of dual variable vector $length(λ)) ≠ hessian stack depth $(size(H3, 3))"
    @inbounds for k = 1:length(λ)
        sliceₖ = @view H3[:, :, k]
        sliceₖ .*= λ[k]
        ∑H .+= sliceₖ
    end

    y, DiffResults.value(H), Symmetric(∑H)
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
    settings::Clarabel.Settings{T},
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

    solver = Clarabel.Solver(P, q, A, b, K, settings)
    solution = Clarabel.solve!(solver)
    # solution.x → primal solution
    # solution.z → dual solution
    # solution.s → slacks
    (solution.x, solution.z)
end

function solve!(solver::SQPSolver{T}, custom_gradients::Bool = false, debug::Bool = true) where {T}
    settings = get_settings(solver)
    for k = 1:settings.max_iter
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
            g, Jg, ▽²g = super_hessian_constraints(inequality_constraints(solver), primal(solver), λ)
            h, Jh, ▽²h = super_hessian_constraints(equality_constraints(solver), primal(solver), v)
        end

        L = f + λ' * g + v' * h
        ▽L = ▽f + Jg' * λ + Jh' * v
        ▽²L = ▽²f + ▽²g + ▽²h

        negate!(Jg)
        negate!(Jh)

        pₖ, lₖ = solve_qp(g, Jg, h, Jh, ▽L, ▽²L, settings)

        # solution step
        knotpoints(primal(solver)) .+= settings.max_step_fraction .* pₖ
        inequality_duals(solver) .+= settings.max_step_fraction .* @view lₖ[1:length(g)]
        equality_duals(solver) .+= settings.max_step_fraction .* @view lₖ[(length(g)+1):end]

        if debug
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
