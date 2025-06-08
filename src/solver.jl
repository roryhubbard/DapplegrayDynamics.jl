struct Problem{T}
    mechanism::Mechanism{T}
    objectives::AbstractVector{<:AdjacentKnotPointsFunction}
    equality_constraints::AbstractVector{<:AdjacentKnotPointsFunction}
    inequality_constraints::AbstractVector{<:AdjacentKnotPointsFunction}
    trajectory::DiscreteTrajectory{T,T}
end

objectives(problem::Problem) = problem.objectives

equality_constraints(problem::Problem) = problem.equality_constraints

inequality_constraints(problem::Problem) = problem.inequality_constraints

trajectory(problem::Problem) = problem.trajectory

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
    result = 0
    for constraint ∈ constraints
        result += outputdim(constraint) * length(indices(constraint))
    end
    result
end

function evaluate_objective(
    objectives::AbstractVector{<:AdjacentKnotPointsFunction},
    Z::DiscreteTrajectory,
)
    result = 0.0
    for objective ∈ objectives
        result += objective(Val(Sum), Z)
    end
    result
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

function evaluate_lagrangian(
    f::T,
    λ::AbstractVector{T},
    g::AbstractVector{T},
    v::AbstractVector{T},
    h::AbstractVector{T},
) where {T}
    f + λ' * g + v' * h
end

function ▽Lagrangian(
    ▽f::AbstractVector{T},
    λ::AbstractVector{T},
    Jg::AbstractMatrix{T},
    v::AbstractVector{T},
    Jh::AbstractMatrix{T},
) where {T}
    ▽f + Jg' * λ + Jh' * v
end

function solve!(problem::Problem{T}) where {T}
    v = zeros(num_lagrange_multipliers(equality_constraints(problem)))
    println("v: ", v)

    λ = zeros(num_lagrange_multipliers(inequality_constraints(problem)))
    println("λ: ", λ)

    for k = 1:1
        f = evaluate_objective(objectives(problem), trajectory(problem))
        println("f: ", f)

        h = evaluate_constraints(equality_constraints(problem), trajectory(problem))
        println("h $(size(h)): ", h)

        g = evaluate_constraints(inequality_constraints(problem), trajectory(problem))
        println("g $(size(g)): ", g)

        ▽f = gradient(Val(Sum), objectives(problem), trajectory(problem))
        println("▽f $(size(▽f)): ", ▽f)

        Jh = jacobian(equality_constraints(problem), trajectory(problem))
        println("Jh $(size(Jh)): ", Jh)

        Jg = jacobian(inequality_constraints(problem), trajectory(problem))
        println("Jg $(size(Jg)): ", Jg)

        L = evaluate_lagrangian(f, λ, g, v, h)
        println("L $(size(L)): ", L)

        ▽L = ▽Lagrangian(▽f, λ, Jg, v, Jh)
        println("▽L $(size(▽L)): ", ▽L)

        ▽²f = hessian(objectives(problem), trajectory(problem))
        println("▽²f $(size(▽²f)): ", ▽²f)

        ▽²h = vector_hessian(equality_constraints(problem), trajectory(problem))
        println("▽²h $(size(▽²h)): ", ▽²h)

        ▽²g = vector_hessian(inequality_constraints(problem), trajectory(problem))
        println("▽²g $(size(▽²g)): ", ▽²g)

        #        superg = super_gradient(objectives(problem), trajectory(problem))
        #        println("sg $(size(superg)): ", superg)
        #
        #        superJg = super_jacobian(inequality_constraints(problem), trajectory(problem))
        #        println("superJg $(size(superJg)): ", superJg)
        #
        #        superJh = super_jacobian(equality_constraints(problem), trajectory(problem))
        #        println("superJh: $(size(superJh))", superJh)

        #        ▽²L = ▽²Lagrangian()
        #        println("▽²L: ", ▽²L)
        #        """
        #        Solve QP using Clarabel
        #
        #        minimize   1⁄2𝒙ᵀ𝑷𝒙 + 𝒒ᵀ𝒙
        #        subject to  𝑨𝒙 + 𝒔 = 𝒃
        #                         𝒔 ∈ 𝑲
        #        with decision variables 𝒙 ∈ ℝⁿ, 𝒔 ∈ 𝑲 and data matrices 𝑷 = 𝑷ᵀ ≥ 0,
        #        𝒒 ∈ ℝⁿ, 𝑨 ∈ ℝᵐˣⁿ, and b ∈ ℝᵐ. The convext set 𝑲 is a composition of convex cones.
        #        """
        #        𝑷 = sparse(▽²ₓₓℒ)
        #        𝒒 = sparse(▽ₓℒ)
        #        𝑨 = sparse([𝑱ₓ𝒉;
        #                    𝑱ₓ𝒈;
        #                    ])
        #        𝒃 = [-𝒉;
        #             -𝒈]
        #        𝑲 = [
        #            Clarabel.ZeroConeT(length(𝒉)),
        #            Clarabel.NonnegativeConeT(length(𝒈))]
        #
        #        settings = Clarabel.Settings()
        #        solver   = Clarabel.Solver()
        #        Clarabel.setup!(solver, 𝑷, 𝒒, 𝑨, 𝒃, 𝑲, settings)
        #        result = Clarabel.solve!(solver)
        #        𝚫𝒙ₖ₊₁, 𝒗ₖ₊₁, 𝝀ₖ₊₁ = unpack_result(result)
        #
        #        nudge_𝒙!(solver, 𝚫𝒙ₖ₊₁)
        #        set_𝒗!(solver, 𝒗ₖ₊₁)
        #        set_𝝀!(solver, 𝝀ₖ₊₁)
    end
end
