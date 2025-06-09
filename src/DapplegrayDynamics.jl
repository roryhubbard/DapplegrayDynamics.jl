module DapplegrayDynamics

using Clarabel
using ForwardDiff
using LinearAlgebra
using RigidBodyDynamics
using SparseArrays
using StaticArrays

export df

include("trajectory.jl")
include("knotpointsfunction.jl")
include("constraints.jl")
include("objective.jl")
include("rigidbodydynamics.jl")
include("solver.jl")

function df(method::Symbol = :sqp)
    mechanism = doublependulum()
    nx = num_positions(mechanism) + num_velocities(mechanism)
    nu = 1 # control dimension
    knotpointsize = nx + nu

    N = 2
    tf = 1.0           # final time (sec)
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
        HermiteSimpsonConstraint(mechanism, 1:(N-1)),
        state_equality_constraint(x0, knotpointsize, 1),
        state_equality_constraint(xf, knotpointsize, N),
    ]

    initial_solution = initialize_trajectory(mechanism, tf, Δt, nu)

    solver = SQPSolver(mechanism, objectives, equality_constraints, inequality_constraints, initial_solution)

    solve!(solver)
end

end
