module DapplegrayDynamics

import RobotZoo.Pendulum
using Altro
using LinearAlgebra
using RobotDynamics
using StaticArrays
using TrajectoryOptimization

export swingup


function swingup()
    model = Pendulum()
    n = state_dim(model)
    m = control_dim(model)

    tf = 3.0           # final time (sec)
    N = 21             # number of knot points
    dt = tf / (N - 1)  # time step (sec)

    # Objective
    x0 = SA[0, 0]  # initial state
    xf = SA[π, 0]  # final state

    Q = Diagonal(@SVector ones(n))
    Qf = Diagonal(@SVector ones(n))
    R = Diagonal(@SVector ones(m))
    objective = LQRObjective(Q, R, Qf, xf, N);

    # Create our list of constraints
    constraints = ConstraintList(n, m, N)

    # Create the goal constraint
    goalcon = GoalConstraint(xf)
    add_constraint!(constraints, goalcon, N)  # add to the last time step

    # Create control limits
    ubnd = 3
    bnd = BoundConstraint(n, m, u_min=-ubnd, u_max=ubnd)
    add_constraint!(constraints, bnd, 1:N-1)  # add to all but the last time step

    prob = Problem(model, objective, x0, tf, xf=xf, constraints=constraints)

    # Initialization
    u0 = @SVector fill(0.01,m)
    U0 = [u0 for k = 1:N-1]
    initial_controls!(prob, U0)
    rollout!(prob);

    opts = SolverOptions(
        cost_tolerance_intermediate=1e-2,
        penalty_scaling=10.,
        penalty_initial=1.0
    )

    altro = ALTROSolver(prob, opts)
    set_options!(altro, show_summary=true)
    solve!(altro);

    # Extract the solution
    X = states(altro)
    U = controls(altro)
end

end
