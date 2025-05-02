module DapplegrayDynamics

import RobotZoo.Pendulum
using Altro
using LinearAlgebra
using RobotDynamics
using StaticArrays
using TrajectoryOptimization

export swingup


function swingup(method::Symbol = :rk4)
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
    objective = LQRObjective(Q, R, Qf, xf, N)

    # Create constraints
    constraints = ConstraintList(n, m, N)

    # Terminal goal constraint
    goalcon = GoalConstraint(xf)
    add_constraint!(constraints, goalcon, N)

    # Control bounds
    ubnd = 3.0
    bnd = BoundConstraint(n, m, u_min=-ubnd, u_max=ubnd)
    add_constraint!(constraints, bnd, 1:N-1)

    # Construct problem depending on method
    prob = if method == :rk4
        Problem(model, objective, x0, tf; constraints=constraints)
    elseif method == :hermite_simpson
        # Placeholder: HermiteSimpsonConstraint needs to be implemented separately
        hs_constraint = HermiteSimpsonConstraint(model, dt, N)
        Problem(model, objective, x0, tf; constraints=constraints, dynamics=hs_constraint)
    else
        error("Unsupported method: $method. Choose :rk4 or :hermite_simpson.")
    end

    # Initialization
    u0 = @SVector fill(0.01, m)
    U0 = [u0 for _ in 1:N-1]
    initial_controls!(prob, U0)
    rollout!(prob)

    # Solver options
    opts = SolverOptions(
        cost_tolerance_intermediate=1e-2,
        penalty_scaling=10.0,
        penalty_initial=1.0
    )

    # Solve
    altro = ALTROSolver(prob, opts)
    set_options!(altro, show_summary=true)
    solve!(altro)

    # Access result
    X = states(altro)
    U = controls(altro)
    return altro
end

end
