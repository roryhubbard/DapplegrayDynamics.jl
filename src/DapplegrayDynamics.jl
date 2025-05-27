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

#function apply_constraint(K, constraintindices, constraint)
#    println("################### CONSTRAINT ###################")
#    println(constraint)
#
#    T = typeof(constraint)
#    println("type: ", T)
#
#    p = RobotDynamics.output_dim(constraint)
#    println("output_dim: ", p)
#
#    input_dim = RobotDynamics.input_dim(constraint)
#    println("input_dim: ", input_dim)
#
#    input_type = RobotDynamics.functioninputs(constraint)
#    println("input_type: ", input_type)
#
#    sense = TrajectoryOptimization.sense(constraint)
#    println("sense: ", sense)
#
#    for j ∈ constraintindices
#        println("j: ", j)
#
#        k = K[j]
#        x = RobotDynamics.state(k)
#        n = RobotDynamics.state_dim(k)
#        u = RobotDynamics.control(k)
#        m = RobotDynamics.control_dim(k)
#        println("state: $n $x")
#        println("control: $m $u")
#
#        y = RobotDynamics.evaluate(constraint, k)
#        println("evaluate: ", y)
#
#        𝑱 = Matrix{Float64}(undef, p, input_dim)
#        y = Vector{Float64}(undef, p)
#        RobotDynamics.jacobian!(constraint, 𝑱, y, k)
#        println("jacobian: ", 𝑱)
#
#        𝑯 = Matrix{Float64}(undef, input_dim, input_dim)
#        𝝀 = zeros(p) # TODO: get this the right way
#        z_ref = RobotDynamics.getinput(input_type, k)  # this will be x, u, or [x; u]
#        f(zvec) = RobotDynamics.evaluate(constraint, zvec)
#        for i = 1:p
#            fᵢ(zvec) = f(zvec)[i]  # scalar function
#            Hᵢ = ForwardDiff.hessian(fᵢ, z_ref)
#            print("row hessian: ", Hᵢ)
#            𝑯 += 𝝀[i] .* Hᵢ
#        end
#        println("sum of hessians: ", 𝑯)
#    end
#end
#

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

abstract type AbstractKnotPointsFunction end
indices(func::AbstractKnotPointsFunction) = func.idx
(::AbstractKnotPointsFunction)(knotpoints) = error("call on knotpoint trajectory not implemented")

abstract type AdjacentKnotPointsFunction <: AbstractKnotPointsFunction end
function (func::AdjacentKnotPointsFunction)(knotpoints)
    for idx ∈ indices(func)
        func(knotpoints[idx], knotpoints[idx+1])
    end
end
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
function (func::SingleKnotPointFunction)(knotpoints)
    for idx ∈ indices(func)
        func(knotpoints[idx])
    end
end
function (func::SingleKnotPointFunction)(z::AbstractKnotPoint)
    x = state(z)
    u = control(u)
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
    u = control(u)
    _justcontrolcall(func, u)
end

struct ClarabelKnotConstraint <: SingleKnotPointFunction
    A::AbstractMatrix
    b::AbstractVector
    cone::SupportedCone
    functioninputs::FunctionInputs
    idx::UnitRange{Int}
end
function (con::ClarabelKnotConstraint)(z::AbstractVector)
    A * z - b
end

function hermite_simpson_separated(mechanism::Mechanism, Δt::Real, xₖ::AbstractVector, uₖ::AbstractVector, xₖ₊₁::AbstractVector, uₖ₊₁::AbstractVector, xₘ::AbstractVector, uₘ::AbstractVector)
    mechanismstate = MechanismState(mechanism)
    dynamicsresult = DynamicsResult(mechanism)

    ẋₖ = similar(xₖ)
    dynamics!(ẋₖ, dynamicsresult, mechanismstate, xₖ, uₖ)

    ẋₖ₊₁ = similar(xₖ₊₁)
    dynamics!(ẋₖ₊₁, dynamicsresult, mechanismstate, xₖ₊₁, uₖ₊₁)

    ẋₘ = similar(xₖ)
    dynamics!(ẋₘ, dynamicsresult, mechanismstate, xₘ, uₘ)

    c₁ = xₖ₊₁ - xₖ - Δt / 6 * (ẋₖ + 4 * ẋₘ + ẋₖ₊₁)
    c₂ = ẋₘ - 1 / 2 * (xₖ + xₖ₊₁) - Δt / 8 * (ẋₖ - ẋₖ₊₁)
    c₁, c₂
end
function hermite_simpson_compressed(mechanism::Mechanism, Δt::Real, xₖ::AbstractVector, uₖ::AbstractVector, xₖ₊₁::AbstractVector, uₖ₊₁::AbstractVector)
    mechanismstate = MechanismState(mechanism)
    dynamicsresult = DynamicsResult(mechanism)

    ẋₖ = similar(xₖ)
    dynamics!(ẋₖ, dynamicsresult, mechanismstate, xₖ, uₖ)

    ẋₖ₊₁ = similar(xₖ₊₁)
    dynamics!(ẋₖ₊₁, dynamicsresult, mechanismstate, xₖ₊₁, uₖ₊₁)

    # We could add the collocation point as an extra decision varaible and
    # constraint. This would be "separated form". Here we are implementing
    # "compressed form" where we calculate `fcol` and jam it into the constraint
    # for the integral of the system dynamics.
    xₘ = 1 / 2 * (xₖ + xₖ₊₁) + Δt / 8 * (ẋₖ - ẋₖ₊₁)
    uₘ = 1 / 2 * (uₖ + uₖ₊₁)
    ẋₘ = similar(xₖ)
    dynamics!(ẋₘ, dynamicsresult, mechanismstate, xₘ, uₘ)

    # equality constraint: xₖ₊₁ - xₖ = (Δt / 6) * (fₖ + 4fcol + fₖ₊₁)
    xₖ₊₁ - xₖ - Δt / 6 * (ẋₖ + 4 * ẋₘ + ẋₖ₊₁)
end
struct HermiteSimpsonConstraint{T} <: AdjacentKnotPointsFunction
    mechanism::Mechanism{T}
    idx::UnitRange{Int}
end
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
function (con::ControlBound)(_, u::AbstractVector)
    ub = con.upperbound
    lb = con.lowerbound
    if isnothing(ub)
        return lb .- u
    elseif isnothing(lb)
        return u .- ub
    else
        return [lb .- u; u .- ub]
    end
end

struct StateEqualityConstraint <: StateFunction
    xd::AbstractVector
    idx::UnitRange{Int}
end
function (con::StateEqualityConstraint)(x::AbstractVector, _)
    x - xd
end

struct LQRCost <: SingleKnotPointFunction
    Q::AbstractMatrix
    R::AbstractMatrix
    xd::AbstractVector
    ud::AbstractVector
    idx::UnitRange{Int}
end
function LQRCost(Q::AbstractMatrix, R::AbstractMatrix, xd::AbstractVector, idx::UnitRange{Int})
    ud = zeros(size(R, 2))
    return LQRCost(Q, R, xd, ud, idx)
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
function (cost::StateCost)(x::AbstractVector, _)
    x̄ = (x - xd)
    x̄' * cost.Q * x̄
end

struct ControlCost <: ControlFunction
    R::AbstractMatrix
    idx::UnitRange{Int}
end
function (cost::ControlCost)(_, u::AbstractVector)
    u' * cost.R * u
end

## hessian w.r.t to only state even though function sig takes u too
#dummy_u = nothing              # can also be `SVector{0,Float64}()`
#f_x = x -> cost(x, dummy_u)    # Rⁿ → ℝ  (u is fixed & unused)
## current state at knot-point k
#x_k = @SVector randn(n)        # replace with your real state
## Hessian w.r.t. x ONLY:   n × n
#H = ForwardDiff.hessian(f_x, x_k)

function initialize_decision_variables(mechanism::Mechanism, tf::Real, Δt::Real, nu::Int)
    ts, qs, vs = simulate_mechanism(mechanism, tf, Δt, [0.0, 0.0], [0.0, 0.0])

    N  = length(ts) # number of knot points
    zero_u = zeros(nu)
    knotpoints = Vector{KnotPoint}(undef, N)

    for i in 1:N
        x  = [qs[i]; vs[i]]
        t  = ts[i]
        dt = (i == N) ? 0.0 : Δt # terminal point → dt = 0

        knotpoints[i] = KnotPoint(x, zero_u, t, dt)
    end

    knotpoints
end

struct Problem
    mechanism::Mechanism
    objective::AbstractVector{<:AbstractKnotPointsFunction}
    inequality_constraints::AbstractVector{<:AbstractKnotPointsFunction}
    equality_constraints::AbstractVector{<:AbstractKnotPointsFunction}
    knotpoints::AbstractVector{<:AbstractKnotPoint}
end

struct SQP
end

#function solve!(solver::SQP, problem::Problem)
#    for _ = 1:1 # TODO: repeat until convergence criteria is met
#        𝒇 = get_objective(problem)
#        constraints = get_constraints(problem)
#
#        Z = get_trajectory(solver.problem)
#        println("trajectory: ", Z)
#
#        K = RobotDynamics.getdata(Z)
#        println("K: ", K)
#
#        X = states(Z)
#        println("X: ", X)
#
#        U = controls(Z)
#        println("U: ", U)
#
#        times = gettimes(Z)
#        println("times: ", times)
#        println()
#
#        for (constraintindices, constraint) ∈ zip(constraints)
#            apply_constraint(Z, constraintindices, constraint)
#            println()
#        end
#
#        return
#
#        𝒉 = equality_constraints(constraints)
#        𝒈 = inequality_constraints(constraints)
#        𝒗 = equality_dual_vector(solver)
#        𝝀 = inequality_dual_vector(solver)
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
#    end
#end

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

    objective = [
        LQRCost(Q, R, xf, 1:N-1),
    ]

    τbound = 3.0
    constraints = [
        HermiteSimpsonConstraint(mechanism, 1:N),
        ControlBound([τbound], [-τbound], 1:N-1),
        StateEqualityConstraint(x0, 1:1),
        StateEqualityConstraint(xf, N:N),
    ]

    knotpoints = initialize_decision_variables(mechanism, tf, Δt, nu)

#    problem = Problem(mechanism, objective, constraints, knotpoints)

    solver = SQP()
#    solve!(solver)
end

end
