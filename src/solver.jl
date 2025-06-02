struct DiscreteTrajectory{T}
    time::AbstractVector{T}
    timesteps::AbstractVector{T}
    knotpoints::AbstractVector{T}
    knotpointsize::Int
    nstates::Int
    function(time::AbstractVector{T}, timesteps::AbstractVector{T}, knotpoints::AbstractVector{T}, knotpointsize::Int, nstates::Int) where {T}
        nk = length(knotpoints) ÷ knotpointsize
        @assert length(time) == length(timesteps) == nk "lengths must match"
        new{T}(time, timesteps, knotpoints, knotpointsize, nstates)
    end
end

time(trajectory::DiscreteTrajectory{T}) where {T} = trajectory.time

timesteps(trajectory::DiscreteTrajectory{T}) where {T} = trajectory.timesteps

knotpoints(trajectory::DiscreteTrajectory{T}) where {T} = trajectory.knotpoints
function knotpoints(trajectory::DiscreteTrajectory{T}, idx::UnitRange{Int}; useview::Bool=true) where {T}
    irange = knotpointindices(trajectory, idx)
    return useview ? view(knotpoints(trajectory), irange) : knotpoints(trajectory)[irange]
end

function knotpointindices(trajectory::DiscreteTrajectory{T}, idx::UnitRange{Int}) where {T}
    ksize = knotpointsize(trajectory)
    idx₀ = (first(idx) - 1) * ksize + 1
    idx₁ = last(idx) * ksize
    idx₀:idx₁
end
knotpointindices(trajectory::DiscreteTrajectory{T}, idx::Int) where {T} = knotpointindices(trajectory, idx:idx)
knotpointindex(trajectory::DiscreteTrajectory{T}, idx::Int) where {T} = (idx - 1) * knotpointsize(trajectory) + 1

knotpointsize(trajectory::DiscreteTrajectory{T}) where {T} = trajectory.knotpointsize

nstates(trajectory::DiscreteTrajectory{T}) where {T} = trajectory.nstates

function Base.getindex(trajectory::DiscreteTrajectory{T}, idx::UnitRange{Int}) where {T}
    return DiscreteTrajectory(
        time(trajectory)[idx],
        timesteps(trajectory)[idx],
        knotpoints(trajectory, idx, useview=false),
        knotpointsize(trajectory),
        nstates(trajectory)
    )
end

function Base.view(trajectory::DiscreteTrajectory{T}, idx::UnitRange{Int}) where {T}
    return DiscreteTrajectory(
        view(time(trajectory), idx),
        view(timesteps(trajectory), idx),
        view(knotpoints(trajectory, idx), useview=true),
        knotpointsize(trajectory),
        nstates(trajectory)
    )
end

Base.getindex(trajectory::DiscreteTrajectory{T}, idx::Int) where {T} = trajectory[idx:idx]
Base.view(trajectory::DiscreteTrajectory{T}, idx::Int) where {T} = view(trajectory, idx:idx)

function state(trajectory::DiscreteTrajectory{T}, idx::Int) where {T}
    k = knotpoints(trajectory)
    i₀ = knotpointindex(trajectory, idx)
    i₁ = i₀ + nstates(trajectory) - 1
    x = @view k[i₀:i₁]
    x
end
function state(trajectory::DiscreteTrajectory{T}, ::Val{N}) where {T, N}
    k = knotpoints(trajectory)
    return ntuple(i -> state(trajectory, i), Val(N))
end

function control(trajectory::DiscreteTrajectory{T}, idx::Int) where {T}
    k = knotpoints(trajectory)
    xidx₀ = knotpointindex(trajectory, idx)
    i₀ = xidx₀ + nstates(trajectory)
    i₁ = xidx₀ + knotpointsize(trajectory) - 1
    u = @view k[i₀:i₁]
    u
end
function control(trajectory::DiscreteTrajectory{T}, ::Val{N}) where {T, N}
    k = knotpoints(trajectory)
    nx = nstates(trajectory)
    ksize = knotpointsize(trajectory)
    return ntuple(i -> control(trajectory, i), Val(N))
end

function initialize_trajectory(mechanism::Mechanism{T}, tf::T, Δt::T, nu::Int) where {T}
    ts, qs, vs = simulate_mechanism(mechanism, tf, Δt, zeros(T, 2), zeros(T, 2))

    N  = length(ts)
    nx = num_positions(con.mechanism) + num_velocities(con.mechanism)
    knotpointsize = nx + nu
    num_decision_variables = N * knotpointsize
    zero_control_vector = zeros(nu)

    timesteps = Vector{T}(Δt, N)
    knotpoints = Vector{T}(undef, num_decision_variables)

    for i in 1:N
        idx₀ = (i - 1) * knotpointsize + 1
        idx₁ = idx₀ + knotpointsize - 1
        knotpoint  = [qs[i]; vs[i]; zero_control_vector]
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

struct Problem{T}
    mechanism::Mechanism{T}
    objectives::AbstractVector{<:AdjacentKnotPointsFunction}
    equality_constraints::AbstractVector{<:AdjacentKnotPointsFunction}
    inequality_constraints::AbstractVector{<:AdjacentKnotPointsFunction}
    trajectory::DiscreteTrajectory{T}
end
function objectives(problem::Problem)
    problem.objectives
end
function equality_constraints(problem::Problem)
    problem.equality_constraints
end
function inequality_constraints(problem::Problem)
    problem.inequality_constraints
end
function trajectory(problem::Problem)
    problem.trajectory
end

function evaluate_objective(objectives::AbstractVector{<:AdjacentKnotPointsFunction}, trajectory::DiscreteTrajectory{T}) where {T}
    result = 0.0
    for objective ∈ objectives
        result += objective(::Sum, trajectory)
    end
    result
end

function evaluate_constraints(constraints::AbstractVector{<:AdjacentKnotPointsFunction}, trajectory::DiscreteTrajectory{T}) where {T}
    # TODO: preallocate before here
    result = Vector{T}()
    for constraint in constraints
        val = constraint(::Concatenate, trajectory)
        append!(result, val)
    end
    return result
end

struct SQP{T}
end

function solve!(solver::SQP{T}, problem::Problem{T}) where {T}
    v = zeros(num_lagrange_multipliers(equality_constraints(problem)))
    println("v: ", v)

    λ = zeros(num_lagrange_multipliers(inequality_constraints(problem)))
    println("λ: ", λ)

    for k = 1:1 # TODO: repeat until convergence criteria is met
        fₖ = evaluate_objective(objectives(problem), trajectory(problem))
        println("fₖ: ", fₖ)

        hₖ = evaluate_constraints(equality_constraints(problem), trajectory(problem))
        println("hₖ: ", hₖ)

        gₖ = evaluate_constraints(inequality_constraints(problem), trajectory(problem))
        println("gₖ: ", hₖ)

        ▽f_vstacked = gradient(objectives(problem), trajectory(problem))
        println("▽f_vstacked: ", Matrix(▽f_vstacked))

        Jh = jacobian(equality_constraints(problem), trajectory(problem))
        println("Jh: ", Jh)

        Jg = jacobian(inequality_constraints(problem), trajectory(problem))
        println("Jg: ", Matrix(Jg))

#        ℒ = build_lagrangian(𝒇, 𝒉, 𝒈, 𝒗, 𝝀)
#        ▽ₓ𝒇 = gradient(𝒇)
#        𝑱ₓ𝒉 = jacobian(𝒉)
#        𝑱ₓ𝒈 = jacobian(𝒈)
#        # ▽ₓℒ = gradiant(ℒ)
#        ▽ₓℒ = ▽ₓ𝒇 + 𝑱ₓ𝒉'𝒗 + 𝑱ₓ𝒈'𝝀
#        ▽²ₓₓℒ = hessian(▽ₓℒ)
#
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
