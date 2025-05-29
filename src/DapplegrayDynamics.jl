module DapplegrayDynamics

using Clarabel
using ForwardDiff
using LinearAlgebra
using RigidBodyDynamics
using SparseArrays
using StaticArrays

export swingup, doublependulum

include("knotpoint.jl")

#function build_lagrangian{T}(
#    𝒇::TrajectoryOptimization.AbstractObjective,
#    𝒉::Vector{TrajectoryOptimization.AbstractConstraint},
#    𝒈::Vector{TrajectoryOptimization.AbstractConstraint},
#    𝒗::Vector{T},
#    𝝀::Vector{T},
#)
#    𝒇 + 𝒗'𝒉 + 𝝀'𝒈
#end

function doublependulum()::Mechanism
    g = -9.81 # gravitational acceleration in z-direction
    world = RigidBody{Float64}("world")
    doublependulum = Mechanism(world; gravity = SVector(0, 0, g))

    axis = SVector(0., 1., 0.) # joint axis
    I_1 = 0.333 # moment of inertia about joint axis
    c_1 = -0.5 # center of mass location with respect to joint axis
    m_1 = 1. # mass
    frame1 = CartesianFrame3D("upper_link") # the reference frame in which the spatial inertia will be expressed
    inertia1 = SpatialInertia(frame1,
        moment=I_1 * axis * axis',
        com=SVector(0, 0, c_1),
        mass=m_1)

    upperlink = RigidBody(inertia1)
    shoulder = Joint("shoulder", Revolute(axis))
    before_shoulder_to_world = one(Transform3D,
        frame_before(shoulder), default_frame(world))
    attach!(doublependulum, world, upperlink, shoulder,
        joint_pose = before_shoulder_to_world)

    l_1 = -1. # length of the upper link
    I_2 = 0.333 # moment of inertia about joint axis
    c_2 = -0.5 # center of mass location with respect to joint axis
    m_2 = 1. # mass
    inertia2 = SpatialInertia(CartesianFrame3D("lower_link"),
        moment=I_2 * axis * axis',
        com=SVector(0, 0, c_2),
        mass=m_2)
    lowerlink = RigidBody(inertia2)
    elbow = Joint("elbow", Revolute(axis))
    before_elbow_to_after_shoulder = Transform3D(
        frame_before(elbow), frame_after(shoulder), SVector(0, 0, l_1))
    attach!(doublependulum, upperlink, lowerlink, elbow,
        joint_pose = before_elbow_to_after_shoulder)

    doublependulum
end

function simulate_mechanism(mechansim::Mechanism, finaltime::Real, Δt::Real, initial_configuration::AbstractVector, initial_velocity::AbstractVector)
    state = MechanismState(mechansim)
    set_configuration!(state, initial_configuration)
    set_velocity!(state, initial_velocity)

    simulate(state, finaltime, Δt=Δt);
end

abstract type FunctionInputs end
struct StateOnly <: FunctionInputs end
struct ControlOnly <: FunctionInputs end
struct StateControl <: FunctionInputs end

abstract type FunctionOutput end
struct ScalarOutput <: FunctionOutput end
struct VectorOutput <: FunctionOutput end

abstract type AbstractKnotPointsFunction end
indices(func::AbstractKnotPointsFunction) = func.idx
outputtype(func::AbstractKnotPointsFunction) = error("outputtype not defined")
outputdim() = error("outputdim not defined")
(::AbstractKnotPointsFunction)(knotpoints::AbstractVector{<:AbstractKnotPoint}) = error("call on knotpoint trajectory not implemented")

abstract type AdjacentKnotPointsFunction <: AbstractKnotPointsFunction end
function (func::AdjacentKnotPointsFunction)(::ScalarOutput, knotpoints::AbstractVector{<:AbstractKnotPoint})
    result = 0.0
    for idx ∈ indices(func)
        result += func(knotpoints[idx], knotpoints[idx+1])
    end
    result
end
function (func::AdjacentKnotPointsFunction)(::VectorOutput, knotpoints::AbstractVector{<:AbstractKnotPoint})
    vcat(map(idx -> func(knotpoints[idx], knotpoints[idx+1]), indices(func))...)
end
(func::AdjacentKnotPointsFunction)(knotpoints::AbstractVector{<:AbstractKnotPoint}) = func(outputtype(func), knotpoints)
function gradient(func::AdjacentKnotPointsFunction, zₖ::AbstractKnotPoint, zₖ₊₁::AbstractKnotPoint)
    z = [zₖ; zₖ₊₁]
    nₖ = length(zₖ)
    func_stacked_knots(z) = func(z[1:nₖ], z[nₖ+1:end])
    ForwardDiff.gradient(func_stacked_knots, z)
end
function jacobian(func::AdjacentKnotPointsFunction, zₖ::AbstractKnotPoint, zₖ₊₁::AbstractKnotPoint)
    z = [zₖ; zₖ₊₁]
    nₖ = length(zₖ)
    func_stacked_knots(z) = func(z[1:nₖ], z[nₖ+1:end])
    ForwardDiff.jacobian(func_stacked_knots, z)
end
function hessian(func::AdjacentKnotPointsFunction, zₖ::AbstractKnotPoint, zₖ₊₁::AbstractKnotPoint)
    z = [zₖ; zₖ₊₁]
    nₖ = length(zₖ)
    func_stacked_knots(z) = func(z[1:nₖ], z[nₖ+1:end])
    ForwardDiff.hessian(func_stacked_knots, z)
end

abstract type SingleKnotPointFunction <: AbstractKnotPointsFunction end
(::SingleKnotPointFunction)(_, _) = error("f(x, u) not implemented")
function (func::SingleKnotPointFunction)(::ScalarOutput, knotpoints::AbstractVector{<:AbstractKnotPoint})
    result = 0.0
    for idx ∈ indices(func)
        result += func(knotpoints[idx])
    end
    result
end
function (func::SingleKnotPointFunction)(::VectorOutput, knotpoints::AbstractVector{<:AbstractKnotPoint})
    vcat(map(idx -> func(knotpoints[idx]), indices(func))...)
end
function (func::SingleKnotPointFunction)(z::AbstractKnotPoint)
    x = state(z)
    u = control(z)
    func(x, u)
end
function gradient(func::SingleKnotPointFunction, z::AbstractKnotPoint)
    ForwardDiff.gradient(func, z)
end
function jacobian(func::SingleKnotPointFunction, z::AbstractKnotPoint)
    ForwardDiff.jacobian(func, z)
end
function hessian(func::SingleKnotPointFunction, z::AbstractKnotPoint)
    ForwardDiff.hessian(func, z)
end
function gradient(func::SingleKnotPointFunction, knotpoints::AbstractVector{<:AbstractKnotPoint})
    rows = indices(func)
    I, J, V = Int[], Int[], Float64[]

    for (i, idx) in enumerate(rows)
        g = gradient(func, knotpoints[idx])
        for (j, val) in pairs(g)
            push!(I, i)
            push!(J, j)
            push!(V, val)
        end
    end

    m = length(rows)
    n = length(knotpoints[1]) * length(knotpoints)
    return sparse(I, J, V, m, n)
end
function gradient(funcs::AbstractVector{<:SingleKnotPointFunction}, knotpoints::AbstractVector{<:AbstractKnotPoint})
    for func ∈ funcs
        r = gradient(func, knotpoints)
        print(r)
    end
end

abstract type StateFunction <: SingleKnotPointFunction end
function _juststatecall(func::StateFunction, x::AbstractVector)
    func(x, nothing)
end
function (func::StateFunction)(z::AbstractKnotPoint)
    x = state(z)
    _juststatecall(func, x)
end

abstract type ControlFunction <: SingleKnotPointFunction end
function _justcontrolcall(func::ControlFunction, u::AbstractVector)
    func(nothing, u)
end
function (func::ControlFunction)(z::AbstractKnotPoint)
    u = control(z)
    _justcontrolcall(func, u)
end

struct ClarabelKnotConstraint <: SingleKnotPointFunction
    A::AbstractMatrix
    b::AbstractVector
    cone::Clarabel.SupportedCone
    functioninputs::FunctionInputs
    idx::UnitRange{Int}
end
outputtype(::ClarabelKnotConstraint) = VectorOutput()
function (con::ClarabelKnotConstraint)(z::AbstractVector)
    A * z - b
end

# TODO: dynamics! assumes fully actuated systems, figure out a method for
# dealing with control vectors that are smaller in rank than the vector of
# "velocities" in mechanism state
function hermite_simpson_separated(mechanism::Mechanism, Δt::Real, xₖ::AbstractVector, uₖ::AbstractVector, xₖ₊₁::AbstractVector, uₖ₊₁::AbstractVector, xₘ::AbstractVector, uₘ::AbstractVector)
    mechanismstate = MechanismState(mechanism)
    dynamicsresult = DynamicsResult(mechanism)

    τₖ = vcat(0., uₖ)
    ẋₖ = similar(xₖ)
    dynamics!(ẋₖ, dynamicsresult, mechanismstate, xₖ, τₖ)

    τₖ₊₁ = vcat(0., uₖ₊₁)
    ẋₖ₊₁ = similar(xₖ₊₁)
    dynamics!(ẋₖ₊₁, dynamicsresult, mechanismstate, xₖ₊₁, τₖ₊₁)

    τₘ = vcat(0., uₘ)
    ẋₘ = similar(xₖ)
    dynamics!(ẋₘ, dynamicsresult, mechanismstate, xₘ, τₘ)

    c₁ = xₖ₊₁ - xₖ - Δt / 6 * (ẋₖ + 4 * ẋₘ + ẋₖ₊₁)
    c₂ = ẋₘ - 1 / 2 * (xₖ + xₖ₊₁) - Δt / 8 * (ẋₖ - ẋₖ₊₁)
    c₁, c₂
end
function hermite_simpson_compressed(mechanism::Mechanism, Δt::Real, xₖ::AbstractVector, uₖ::AbstractVector, xₖ₊₁::AbstractVector, uₖ₊₁::AbstractVector)
    mechanismstate = MechanismState(mechanism)
    dynamicsresult = DynamicsResult(mechanism)

    τₖ = vcat(0., uₖ)
    ẋₖ = similar(xₖ)
    dynamics!(ẋₖ, dynamicsresult, mechanismstate, xₖ, τₖ)

    τₖ₊₁ = vcat(0., uₖ₊₁)
    ẋₖ₊₁ = similar(xₖ₊₁)
    dynamics!(ẋₖ₊₁, dynamicsresult, mechanismstate, xₖ₊₁, τₖ₊₁)

    # We could add the collocation point as an extra decision varaible and
    # constraint. This would be "separated form". Here we are implementing
    # "compressed form" where we calculate `fcol` and jam it into the constraint
    # for the integral of the system dynamics.
    xₘ = 1 / 2 * (xₖ + xₖ₊₁) + Δt / 8 * (ẋₖ - ẋₖ₊₁)
    uₘ = 1 / 2 * (uₖ + uₖ₊₁)
    τₘ = vcat(0., uₘ)
    ẋₘ = similar(xₖ)
    dynamics!(ẋₘ, dynamicsresult, mechanismstate, xₘ, τₘ)

    # equality constraint: xₖ₊₁ - xₖ = (Δt / 6) * (fₖ + 4fcol + fₖ₊₁)
    xₖ₊₁ - xₖ - Δt / 6 * (ẋₖ + 4 * ẋₘ + ẋₖ₊₁)
end
struct HermiteSimpsonConstraint{T} <: AdjacentKnotPointsFunction
    mechanism::Mechanism{T}
    idx::UnitRange{Int}
end
outputtype(::HermiteSimpsonConstraint) = VectorOutput()
outputdim(con::HermiteSimpsonConstraint) = num_positions(con.mechanism) + num_velocities(con.mechanism)
function (con::HermiteSimpsonConstraint)(zₖ::AbstractKnotPoint, zₖ₊₁::AbstractKnotPoint)
    xₖ = state(zₖ)
    uₖ = control(zₖ)
    xₖ₊₁ = state(zₖ₊₁)
    uₖ₊₁ = control(zₖ₊₁)
    Δt = timestep(zₖ)
    hermite_simpson_compressed(con.mechanism, Δt, xₖ, uₖ, xₖ₊₁, uₖ₊₁)
end

struct ControlBound{T} <: ControlFunction
    upperbound::Union{AbstractVector{T}, Nothing}
    lowerbound::Union{AbstractVector{T}, Nothing}
    idx::UnitRange{Int}

    function ControlBound(upperbound::Union{AbstractVector{T}, Nothing},
                          lowerbound::Union{AbstractVector{T}, Nothing},
                          idx::UnitRange{Int}) where {T}
        if isnothing(upperbound) && isnothing(lowerbound)
            throw(ArgumentError("At least one of upperbound or lowerbound must be provided."))
        end
        new{T}(upperbound, lowerbound, idx)
    end
end
outputtype(::ControlBound) = VectorOutput()
outputdim(con::ControlBound) = isnothing(con.upperbound) ? length(con.lowerbound) : length(con.upperbound)
function (con::ControlBound)(::Union{AbstractVector, Nothing}, u::AbstractVector)
    ub = con.upperbound
    lb = con.lowerbound
    if isnothing(ub)
        return lb - u
    elseif isnothing(lb)
        return u - ub
    else
        return [lb - u; u - ub]
    end
end

struct StateEqualityConstraint <: StateFunction
    xd::AbstractVector
    idx::UnitRange{Int}
end
outputtype(::StateEqualityConstraint) = VectorOutput()
outputdim(con::StateEqualityConstraint) = length(con.xd)
function (con::StateEqualityConstraint)(x::AbstractVector, _)
    x - con.xd
end

struct LQRCost <: SingleKnotPointFunction
    Q::AbstractMatrix
    R::AbstractMatrix
    xd::AbstractVector
    ud::AbstractVector
    idx::UnitRange{Int}
end
outputtype(::LQRCost) = ScalarOutput()
outputdim(::LQRCost) = 1
function LQRCost(Q::AbstractMatrix, R::AbstractMatrix, xd::AbstractVector, idx::UnitRange{Int})
    ud = zeros(size(R, 2))
    LQRCost(Q, R, xd, ud, idx)
end
function (cost::LQRCost)(x::AbstractVector, u::AbstractVector)
    x̄ = (x - cost.xd)
    ū = (u - cost.ud)
    1 / 2 * (x̄' * cost.Q * x̄ + ū' * cost.R * ū)
end

struct StateCost <: StateFunction
    Q::AbstractMatrix
    xd::AbstractVector
    idx::UnitRange{Int}
end
outputtype(::StateCost) = ScalarOutput()
outputdim(::StateCost) = 1
function (cost::StateCost)(x::AbstractVector, _)
    x̄ = (x - cost.xd)
    x̄' * cost.Q * x̄
end

struct ControlCost <: ControlFunction
    R::AbstractMatrix
    idx::UnitRange{Int}
end
outputtype(::ControlCost) = ScalarOutput()
outputdim(::ControlCost) = 1
function (cost::ControlCost)(_, u::AbstractVector)
    u' * cost.R * u
end

function initialize_decision_variables(mechanism::Mechanism, tf::Real, Δt::Real, nu::Int)
    ts, qs, vs = simulate_mechanism(mechanism, tf, Δt, [0.0, 0.0], [0.0, 0.0])

    N  = length(ts)
    zero_u = zeros(nu)
    knotpoints = Vector{KnotPoint}(undef, N)

    for i in 1:N
        x  = [qs[i]; vs[i]]
        t  = ts[i]
        knotpoints[i] = KnotPoint(x, zero_u, t, Δt)
    end

    knotpoints
end

function num_lagrange_multipliers(constraints::AbstractVector{<:AbstractKnotPointsFunction})
    result = 0
    for constraint ∈ constraints
        result += outputdim(constraint) * length(indices(constraint))
    end
    result
end

struct Problem
    mechanism::Mechanism
    objectives::AbstractVector{<:AbstractKnotPointsFunction}
    equality_constraints::AbstractVector{<:AbstractKnotPointsFunction}
    inequality_constraints::AbstractVector{<:AbstractKnotPointsFunction}
    knotpoints::AbstractVector{<:AbstractKnotPoint}
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
function knotpoints(problem::Problem)
    problem.knotpoints
end

function evaluate_objective(objectives::AbstractVector{<:AbstractKnotPointsFunction}, knotpoints::AbstractVector{<:AbstractKnotPoint})
    result = 0.0
    for objective ∈ objectives
        result += objective(outputtype(objective), knotpoints)
    end
    result
end

function evaluate_constraints(constraints::AbstractVector{<:AbstractKnotPointsFunction}, knotpoints::AbstractVector{<:AbstractKnotPoint})
    result = Vector{Float64}()

    for constraint in constraints
        val = constraint(outputtype(constraint), knotpoints)

        if outputtype(constraint) isa ScalarOutput
            push!(result, val)  # scalar → 1-element appended
        elseif outputtype(constraint) isa VectorOutput
            append!(result, val)  # append all vector elements
        else
            error("Unknown output type")
        end
    end

    return result
end

struct SQP
end

function solve!(solver::SQP, problem::Problem)
    v = zeros(num_lagrange_multipliers(equality_constraints(problem)))
    println("v: ", v)

    λ = zeros(num_lagrange_multipliers(inequality_constraints(problem)))
    println("λ: ", λ)

    for k = 1:1 # TODO: repeat until convergence criteria is met
        fₖ = evaluate_objective(objectives(problem), knotpoints(problem))
        println("fₖ: ", fₖ)

        hₖ = evaluate_constraints(equality_constraints(problem), knotpoints(problem))
        println("hₖ: ", hₖ)

        gₖ = evaluate_constraints(inequality_constraints(problem), knotpoints(problem))
        println("gₖ: ", hₖ)

        ▽f = gradient(objectives(problem), knotpoints(problem))
#        Jₕ = gradient(equality_constraints(problem), knotpoints(problem))
#        Jg = gradient(equality_constraints(problem), knotpoints(problem))

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

function swingup(method::Symbol = :sqp)
    mechanism = doublependulum()
    nx = num_positions(mechanism) + num_velocities(mechanism)
    nu = 1 # control dimension

    N = 2
    tf = 1.0           # final time (sec)
    Δt = tf / (N - 1)  # time step (sec)

    x0 = zeros(nx)
    xf = [π, 0, 0, 0]  # swing up

    Q = 0.01 * I(nx) * Δt
    Qf = 100.0 * I(nx)
    R = 0.1 * I(nu) * Δt

    objectives = [
        LQRCost(Q, R, xf, 1:N-1),
    ]

    τbound = 3.0
    inequality_constraints = [
        ControlBound([τbound], [-τbound], 1:N-1),
    ]
    equality_constraints = [
        HermiteSimpsonConstraint(mechanism, 1:N-1),
        StateEqualityConstraint(x0, 1:1),
        StateEqualityConstraint(xf, N:N),
    ]

    knotpoints = initialize_decision_variables(mechanism, tf, Δt, nu)

    problem = Problem(mechanism, objectives, equality_constraints, inequality_constraints, knotpoints)

    solver = SQP()
    solve!(solver, problem)
end

end
