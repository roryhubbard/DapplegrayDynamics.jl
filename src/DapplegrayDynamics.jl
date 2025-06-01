module DapplegrayDynamics

using Clarabel
using ForwardDiff
using LinearAlgebra
using RigidBodyDynamics
using SparseArrays
using StaticArrays

export df, doublependulum

include("constraint.jl")
include("knotpointfunction.jl")
include("objective.jl")
include("rigidbodydynamics.jl")
include("solver.jl")

function df(method::Symbol = :sqp)
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
#        StateEqualityConstraint(x0, 1:1),
#        StateEqualityConstraint(xf, N:N),
    ]

    knotpoint_trajectory = initialize_trajectory(mechanism, tf, Δt, nu)

    problem = Problem(mechanism, objectives, equality_constraints, inequality_constraints, knotpoint_trajectory)

    solver = SQP()
    solve!(solver, problem)
end

end
