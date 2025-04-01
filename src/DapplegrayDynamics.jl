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

    N = 101
    tf = 5.0           # final time (sec)
    dt = tf / (N - 1)  # time step (sec)

    # Objective
    x0 = @SVector zeros(n)
    xf = @SVector [π, 0]  # swing up

    Q = 0.01 * Diagonal(@SVector ones(n)) * dt
    Qf = 100.0 * Diagonal(@SVector ones(n))
    R = 0.1 * Diagonal(@SVector ones(m)) * dt
    objective = LQRObjective(Q, R, Qf, xf, N);

    # Create our list of constraints
    constraints = ConstraintList(n, m, N)

    # Create the goal constraint
    goalcon = GoalConstraint(xf)
    add_constraint!(constraints, goalcon, N)  # add to the last time step

    # Create control limits
    ubnd = 3.0
    bnd = BoundConstraint(n, m, u_min=-ubnd, u_max=ubnd)
    add_constraint!(constraints, bnd, 1:N-1)  # add to all but the last time step

    prob = Problem(model, objective, x0, tf, xf=xf, constraints=constraints)

    # Initialization
    u0 = @SVector fill(0.01,m)
    U0 = [u0 for k = 1:N-1]
    initial_controls!(prob, U0)
#    rollout!(prob)

    opts = SolverOptions(
        cost_tolerance_intermediate=1e-2,
        penalty_scaling=10.0,
        penalty_initial=1.0
    )

    altro = ALTROSolver(prob, opts)
#    set_options!(altro, show_summary=true)
    solve!(altro);

    # Get some info on the solve
    max_violation(altro)  # 5.896e-7
    cost(altro)           # 1.539
    iterations(altro)     # 44

    # Extract the solution
    X = states(altro)
    U = controls(altro)

    # Extract the solver statistics
    stats = Altro.stats(altro)   # alternatively, solver.stats
    stats.iterations             # 44, equivalent to iterations(solver)
    stats.iterations_outer       # 4 (number of Augmented Lagrangian iterations)
    stats.iterations_pn          # 1 (number of projected newton iterations)
    stats.cost[end]              # terminal cost
    stats.c_max[end]             # terminal constraint satisfaction
    stats.gradient[end]          # terminal gradient of the Lagrangian
    dstats = Dict(stats)         # get the per-iteration stats as a dictionary (can be converted to DataFrame)
end

end
