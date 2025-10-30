module DapplegrayDynamics

using Clarabel
using ForwardDiff
using LinearAlgebra
using RigidBodyDynamics
using SparseArrays
using StaticArrays

# Re-export from RigidBodyDynamics.jl
export Mechanism, parse_urdf

include("trajectory.jl")
include("knotpointsfunction.jl")
include("constraints.jl")
include("objective.jl")
include("rigidbodydynamics.jl")
include("solver.jl")

export acrobot_swingup, df, pendulum_swingup, kj

function acrobot_swingup(mechanism::Mechanism, N::Int, tf::AbstractFloat)
    nq = num_positions(mechanism)
    nv = num_velocities(mechanism)
    nx = nq + nv
    nu = 1 # control dimension
    knotpointsize = nx + nu

    Δt = tf / (N - 1)  # time step (sec)

    x0 = zeros(nx)
    xf = [π, 0, 0, 0]  # swing up

    Q = 0.01 * I(nx) * Δt
    Qf = 100.0 * I(nx)
    R = 0.1 * I(nu) * Δt

    objectives = [LQRCost(Q, R, xf, 1:N)]

    τbound = 3.0
    inequality_constraints =
        [control_bound_constraint(knotpointsize, 1:(N-1), [τbound], [-τbound])]
    equality_constraints = [
        CompressedHermiteSimpsonConstraint(mechanism, 1:(N-1), [2]),
        state_equality_constraint(x0, knotpointsize, 1),
        state_equality_constraint(xf, knotpointsize, N),
    ]

    initial_solution = initialize_trajectory(
        mechanism,
        N,
        tf,
        nu,
        zeros(typeof(tf), nq),
        [π, 0.0],
        zeros(typeof(tf), nv),
        zeros(typeof(tf), nv),
    )

    solver = SQPSolver(
        mechanism,
        objectives,
        equality_constraints,
        inequality_constraints,
        initial_solution,
    )

    solve!(solver)

    solver
end

function load_acrobot()::Mechanism
    srcdir = dirname(pathof(DapplegrayDynamics))
    urdf = joinpath(srcdir, "..", "test", "urdf", "Acrobot.urdf")
    parse_urdf(urdf)
end

function df(urdf::Bool = true)
    mechanism = urdf ? load_acrobot() : doublependulum()
    acrobot_swingup(mechanism, 50, 10.0)
end

function pendulum_swingup(mechanism::Mechanism, N::Int, tf::AbstractFloat)
    nq = num_positions(mechanism)
    nv = num_velocities(mechanism)
    nx = nq + nv
    nu = 1 # control dimension
    knotpointsize = nx + nu

    Δt = tf / (N - 1)  # time step (sec)

    x0 = zeros(nx)
    xf = [π, 0]  # swing up

    Q = 0.01 * I(nx) * Δt
    Qf = 100.0 * I(nx)
    R = 0.1 * I(nu) * Δt

    objectives = [LQRCost(Q, R, xf, 1:N)]

    τbound = 3.0
    inequality_constraints =
        [control_bound_constraint(knotpointsize, 1:(N-1), [τbound], [-τbound])]
    equality_constraints = [
        SeparatedHermiteSimpsonConstraint(mechanism, 1:(N-1), [1]),
        state_equality_constraint(x0, knotpointsize, 1),
        state_equality_constraint(xf, knotpointsize, N),
    ]

    initial_solution = initialize_trajectory(
        mechanism,
        N,
        tf,
        nu,
        zeros(typeof(tf), nq),
        [Float64(π)],
        zeros(typeof(tf), nv),
        zeros(typeof(tf), nv),
    )

    solver = SQPSolver(
        mechanism,
        objectives,
        equality_constraints,
        inequality_constraints,
        initial_solution,
    )

    solve!(solver)

    solver
end

function load_pendulum()::Mechanism
    srcdir = dirname(pathof(DapplegrayDynamics))
    urdf = joinpath(srcdir, "..", "test", "urdf", "pendulum.urdf")
    parse_urdf(urdf)
end

function kj()
    mechanism = load_pendulum()
    pendulum_swingup(mechanism, 50, 10.0)
end

end
