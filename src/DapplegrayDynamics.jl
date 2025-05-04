module DapplegrayDynamics

import RobotZoo.Pendulum
using Altro
using LinearAlgebra
using RobotDynamics
using StaticArrays
using TrajectoryOptimization

export swingup

"""
Interface: Any constraint must implement the following interface:

n = RobotDynamics.state_dim(::MyCon)
m = RobotDynamics.control_dim(::MyCon)
p = RobotDynamics.output_dim(::MyCon)
TrajectoryOptimization.sense(::MyCon)::ConstraintSense
c = RobotDynamics.evaluate(::MyCon, x, u)
RobotDynamics.evaluate!(::MyCon, c, x, u)
"""
struct HermiteSimpsonConstraint{M,T} <: TrajectoryOptimization.StageConstraint
    model::M
    dt::T
end
# State and control dimensions
RobotDynamics.state_dim(con::HermiteSimpsonConstraint) = state_dim(con.model)
RobotDynamics.control_dim(con::HermiteSimpsonConstraint) = control_dim(con.model)
RobotDynamics.output_dim(con::HermiteSimpsonConstraint) = state_dim(con.model)

# Constraint sense: this is an equality constraint
TrajectoryOptimization.sense(::HermiteSimpsonConstraint) = ZeroCone() # ↔ Equality()

function hermite_simpson_compressed(model, dt, xₖ, uₖ, xₖ₊₁, uₖ₊₁)
    fₖ = RobotDynamics.evaluate(model, xₖ, uₖ)
    fₖ₊₁ = RobotDynamics.evaluate(model, xₖ₊₁, uₖ₊₁)

    # We could add the collocation point as an extra decision varaible and
    # constraint. This would be "separated form". Here we are implementing
    # "compressed form" where we calculate `fcol` and jam it into the constraint
    # for the integral of the system dynamics.
    xcol = 0.5 * (xₖ + xₖ₊₁) + dt / 8 * (fₖ - fₖ₊₁)
    ucol = 0.5 * (uₖ + uₖ₊₁)
    fcol = RobotDynamics.evaluate(model, xcol, ucol)

    # equality constraint: xₖ₊₁ - xₖ = (dt / 6) * (fₖ + 4fcol + fₖ₊₁)
    SVector{length(xₖ)}(xₖ₊₁ - xₖ - (dt / 6) * (fₖ + 4fcol + fₖ₊₁))
end

function RobotDynamics.evaluate(
    con::HermiteSimpsonConstraint,
    xₖ::AbstractVector,
    uₖ::AbstractVector,
    xₖ₊₁::AbstractVector,
    uₖ₊₁::AbstractVector,
)
    hermite_simpson_compressed(con.model, con.dt, xₖ, uₖ, xₖ₊₁, uₖ₊₁)
end

function RobotDynamics.evaluate!(
    con::HermiteSimpsonConstraint,
    c::AbstractVector,
    xₖ::AbstractVector,
    uₖ::AbstractVector,
    xₖ₊₁::AbstractVector,
    uₖ₊₁::AbstractVector,
)
    copyto!(c, hermite_simpson_compressed(con.model, con.dt, xₖ, uₖ, xₖ₊₁, uₖ₊₁))
    c
end

struct DapplegraySQP{T} <: ConstrainedSolver{T}
    opts::SolverOptions{T}
    stats::SolverStats{T}
    problem::Problem
end

function swingup(method::Symbol = :altro)
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
    bnd = BoundConstraint(n, m, u_min = -ubnd, u_max = ubnd)
    add_constraint!(constraints, bnd, 1:N-1)

    # Construct problem depending on method
    prob = if method == :altro
        Problem(model, objective, x0, tf; constraints = constraints)
    elseif method == :sqp
        collocation_constraints = HermiteSimpsonConstraint(model, dt)
        add_constraint!(constraints, collocation_constraints, 1:N-1)
        Problem(model, objective, x0, tf; constraints = constraints)
    else
        error("Unsupported method: $method. Choose :altro or :sqp.")
    end

    # Construct solver depending on method
    solver = if method == :altro
        opts = SolverOptions(
            cost_tolerance_intermediate = 1e-2,
            penalty_scaling = 10.0,
            penalty_initial = 1.0,
        )
        ALTROSolver(prob, opts)
    elseif method == :sqp
        DapplegraySQP(prob)
    else
        error("Unsupported method: $method. Choose :altro or :sqp.")
    end

    # Initialization
    u0 = @SVector fill(0.01, m)
    U0 = [u0 for _ = 1:N-1]
    initial_controls!(prob, U0)
    rollout!(prob)

    set_options!(solver, show_summary = true)
    solve!(solver)

    prob
end

end
