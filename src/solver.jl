struct Problem{T}
    mechanism::Mechanism{T}
    objectives::AbstractVector{<:AdjacentKnotPointsFunction}
    inequality_constraints::AbstractVector{<:AdjacentKnotPointsFunction}
    equality_constraints::AbstractVector{<:AdjacentKnotPointsFunction}
    trajectory::DiscreteTrajectory{T,T}
end

objectives(problem::Problem) = problem.objectives

inequality_constraints(problem::Problem) = problem.inequality_constraints

equality_constraints(problem::Problem) = problem.equality_constraints

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

function solve!(problem::Problem{T}) where {T}
    λ = zeros(num_lagrange_multipliers(inequality_constraints(problem)))
    println("λ: ", λ)

    v = zeros(num_lagrange_multipliers(equality_constraints(problem)))
    println("v: ", v)

    for k = 1:1
        f = evaluate_objective(objectives(problem), trajectory(problem))
        println("f: ", f)

        g = evaluate_constraints(inequality_constraints(problem), trajectory(problem))
        println("g $(size(g)): ", g)

        h = evaluate_constraints(equality_constraints(problem), trajectory(problem))
        println("h $(size(h)): ", h)

        ▽f = gradient(Val(Sum), objectives(problem), trajectory(problem))
        println("▽f $(size(▽f)): ", ▽f)

        Jg = jacobian(inequality_constraints(problem), trajectory(problem))
        println("Jg $(size(Jg)): ", Jg)

        Jh = jacobian(equality_constraints(problem), trajectory(problem))
        println("Jh $(size(Jh)): ", Jh)

        L = f + λ' * g + v' * h
        println("L $(size(L)): ", L)

        ▽L = ▽f + Jg' * λ + Jh' * v
        println("▽L $(size(▽L)): ", ▽L)

        ▽²f = hessian(objectives(problem), trajectory(problem))
        println("▽²f $(size(▽²f)): ", ▽²f)

        ▽²g = vector_hessian(inequality_constraints(problem), trajectory(problem), λ)
        println("▽²g $(size(▽²g)): ", ▽²g)

        ▽²h = vector_hessian(equality_constraints(problem), trajectory(problem), v)
        println("▽²h $(size(▽²h)): ", ▽²h)

        ▽²L = ▽²f + ▽²g + ▽²h
        println("▽²L: ", ▽²L)

        """
        Solve QP using Clarabel

        minimize   1⁄2𝒙ᵀ𝑷𝒙 + 𝒒ᵀ𝒙
        subject to  𝑨𝒙 + 𝒔 = 𝒃
                         𝒔 ∈ 𝑲
        with decision variables 𝒙 ∈ ℝⁿ, 𝒔 ∈ 𝑲 and data matrices 𝑷 = 𝑷ᵀ ≥ 0,
        𝒒 ∈ ℝⁿ, 𝑨 ∈ ℝᵐˣⁿ, and b ∈ ℝᵐ. The convext set 𝑲 is a composition of convex cones.
        """
        P = sparse(▽²L)
        q = ▽L
        Jg .*= -1
        Jh .*= -1
        A = sparse([Jg;
                    Jh;
                    ])
        b = [g;
             h]
        K = [
            Clarabel.ZeroConeT(length(h)),
            Clarabel.NonnegativeConeT(length(g))]

        println("P $(size(P)): ", P)
        println("q $(size(q)): ", q)
        println("A $(size(A)): ", A)
        println("b $(size(b)): ", b)
        println("K $(size(K)): ", K)

        settings = Clarabel.Settings()
        solver   = Clarabel.Solver()
        Clarabel.setup!(solver, P, q, A, b, K, settings)
        result = Clarabel.solve!(solver)
        println("QP result ", result)
#        𝚫𝒙ₖ₊₁, 𝒗ₖ₊₁, 𝝀ₖ₊₁ = unpack_result(result)
#
#        nudge_𝒙!(solver, 𝚫𝒙ₖ₊₁)
#        set_𝒗!(solver, 𝒗ₖ₊₁)
#        set_𝝀!(solver, 𝝀ₖ₊₁)
    end
end
