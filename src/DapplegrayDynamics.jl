module DapplegrayDynamics

using Clarabel
using DiffResults
using ForwardDiff
using LinearAlgebra
using ProgressMeter
using RigidBodyDynamics
using SparseArrays
using StaticArrays

# potentially remove later
using GLMakie
# https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/
# https://github.com/jump-dev/NLopt.jl
using NLopt
# https://github.com/cvanaret/Uno
# https://arxiv.org/pdf/2406.13454
using UnoSolver

# Re-export from RigidBodyDynamics.jl
export Mechanism, parse_urdf

include("trajectory.jl")
include("knotpointsfunction.jl")
include("constraints.jl")
include("objective.jl")
include("rigidbodydynamics.jl")
include("solver.jl")

export acrobot_swingup, df, pendulum_swingup, pendulum_swingup_nlopt, kj, nl

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

    @assert isodd(N) "N needs to be odd for SeparatedHermiteSimpsonConstraint but it is $n"
    equality_constraints = [
        [SeparatedHermiteSimpsonConstraint(mechanism, i, [1]) for i = 1:2:(N-2)]...,
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

function pendulum_swingup_nlopt(mechanism::Mechanism, N::Int, tf::AbstractFloat, maxeval::Int)
    nq = num_positions(mechanism)
    nv = num_velocities(mechanism)
    nx = nq + nv
    nu = 1 # control dimension
    knotpointsize = nx + nu

    Δt = tf / (N - 1)  # time step (sec)

    # Initial and final states
    x0 = zeros(nx)
    xf = [π, 0.0]  # swing up

    # LQR cost weights
    Q = 0.01 * I(nx) * Δt
    R = 0.1 * I(nu) * Δt

    # Control bounds
    τbound = 3.0

    # Total number of decision variables
    num_vars = N * knotpointsize

    # Create initial trajectory to get time/timestep structure
    initial_traj = initialize_trajectory(
        mechanism,
        N,
        tf,
        nu,
        zeros(typeof(tf), nq),
        [Float64(π)],
        zeros(typeof(tf), nv),
        zeros(typeof(tf), nv),
    )

    # Create cost and constraint objects
    lqr_cost = LQRCost(Q, R, xf, 1:N)
    objective = [lqr_cost]

    @assert isodd(N) "N needs to be odd for SeparatedHermiteSimpsonConstraint but it is $n"
    equality_constraints = [
        [SeparatedHermiteSimpsonConstraint(mechanism, i, [1]) for i = 1:2:(N-2)]...,
        state_equality_constraint(x0, knotpointsize, 1),
        state_equality_constraint(xf, knotpointsize, N),
    ]
    # ignored but required by super_hessian_constraints
    v = zeros(Float64, num_lagrange_multipliers(equality_constraints))

    inequality_constraints = [
        control_bound_constraint(knotpointsize, 1:(N-1), [τbound], [-τbound])
    ]
    # ignored but required by super_hessian_constraints
    λ = zeros(Float64, num_lagrange_multipliers(inequality_constraints))

    # Trace iterations (following NLopt-README pattern)
    trace = Any[]

    pbar = Progress(maxeval; desc="Running solver")
    function objective_fn(z::Vector, grad::Vector)
        Z = DiscreteTrajectory(time(initial_traj), timesteps(initial_traj), z, knotpointsize, nx)
        f, ▽f, ▽²f = super_hessian_objective(objective, Z)
        if length(grad) > 0
            grad[:] = ▽f
        end
        push!(trace, copy(z) => f)
        next!(pbar)
        return f
    end

    function equality_constraints_fn(result::Vector, z::Vector, grad::Matrix)
        Z = DiscreteTrajectory(time(initial_traj), timesteps(initial_traj), z, knotpointsize, nx)
        h, ▽h, ▽²h = super_hessian_constraints(equality_constraints, Z, v)
        result[:] = h
        if length(grad) > 0
            grad[:, :] = ▽h'
        end
    end

    function inequality_constraints_fn(result::Vector, z::Vector, grad::Matrix)
        Z = DiscreteTrajectory(time(initial_traj), timesteps(initial_traj), z, knotpointsize, nx)
        g, ▽g, ▽²g = super_hessian_constraints(inequality_constraints, Z, λ)
        result[:] = g
        if length(grad) > 0
            grad[:, :] = ▽g'
        end
    end

    # Create NLopt optimizer with NLOPT_LD_SLSQP algorithm
    opt = NLopt.Opt(:LD_SLSQP, num_vars)

    # Set objective
    NLopt.min_objective!(opt, objective_fn)

    # Add equality constraints
    NLopt.equality_constraint!(opt, equality_constraints_fn, fill(1e-8, length(v)))

    # Add inequality constraints (control bounds)
    NLopt.inequality_constraint!(opt, inequality_constraints_fn, fill(1e-8, length(λ)))

    # Set stopping criteria
#    NLopt.xtol_rel!(opt, 1e-6)
    NLopt.maxeval!(opt, maxeval)

    # Initial guess
    z0 = knotpoints(initial_traj)

    # Optimize
    println("Starting NLopt optimization with NLOPT_LD_SLSQP...")
    min_f, min_z, ret = NLopt.optimize(opt, z0)
    num_evals = NLopt.numevals(opt)

    println("""
    ================================
    NLopt Pendulum Swing-Up Results
    ================================
    Objective value       : $min_f
    Solution status       : $ret
    Function evaluations  : $num_evals
    ================================
    """)

    solution_traj = DiscreteTrajectory(
        time(initial_traj),
        timesteps(initial_traj),
        min_z,
        knotpointsize,
        nx
    )

    primal_solutions = [
        DiscreteTrajectory(
            time(initial_traj),
            timesteps(initial_traj),
            z_vec,
            knotpointsize,
            nx
        )
        for (z_vec, obj_val) in trace
    ]

    return (
        optimizer = opt,
        solution = solution_traj,
        objective_value = min_f,
        return_code = ret,
        num_evals = num_evals,
        trace = trace,
        primal_solutions = primal_solutions
    )
end

function load_pendulum()::Mechanism
    srcdir = dirname(pathof(DapplegrayDynamics))
    urdf = joinpath(srcdir, "..", "test", "urdf", "pendulum.urdf")
    parse_urdf(urdf)
end

function plot_pendulum_iterations(primal_solutions::Vector; max_iterations::Int = 10)
    # Create a figure for plotting all trajectories
    fig = Figure(size = (800, 800))
    ax1 = Axis(
        fig[1, 1],
        xlabel = "θ (theta) [deg]",
        ylabel = "θ̇ (thetadot) [deg/s]",
        title = "Pendulum Phase Portrait",
    )

    ax2 = Axis(
        fig[2, 1],
        xlabel = "Time [s]",
        ylabel = "Control (τ) [Nm]",
        title = "Control Trajectories",
    )

    # Subsample iterations if there are too many
    n_total = length(primal_solutions)
    if n_total <= max_iterations
        indices_to_plot = 1:n_total
    else
        # Plot first, last, and evenly spaced intermediate iterations
        indices_to_plot = unique([1; round.(Int, LinRange(2, n_total-1, max_iterations-2)); n_total])
    end

    # Plot each solution trajectory
    for idx ∈ indices_to_plot
        solution_trajectory = primal_solutions[idx]
        ts = time(solution_trajectory)
        qs = position_trajectory(solution_trajectory)
        vs = velocity_trajectory(solution_trajectory)
        us = control_trajectory(solution_trajectory)

        # Extract theta and thetadot for plotting (convert to degrees)
        theta = [rad2deg(first(q)) for q ∈ qs]
        thetadot = [rad2deg(first(v)) for v ∈ vs]

        # Extract controls
        controls = [first(u) for u ∈ us]

        # Plot the phase portrait
        scatterlines!(ax1, theta, thetadot, label = "Iteration $idx")

        # Plot the control trajectory (note: controls have length N-1)
        lines!(ax2, ts[1:length(controls)], controls, label = "Iteration $idx")
    end

    axislegend(ax1, position = :rt)
    axislegend(ax2, position = :rt)
    display(fig)

    return fig
end

function nl()
    mechanism = load_pendulum()
    result = pendulum_swingup_nlopt(mechanism, 51, 10.0, 10)
    plot_pendulum_iterations(result.primal_solutions)
    result
end

function kj()
    mechanism = load_pendulum()
    solver = pendulum_swingup(mechanism, 51, 10.0)

    println(
        "********************************** PRINT GUTS **********************************",
    )
    println(
        "********************************************************************************",
    )
    for (k, _v) in solver.guts
        println(k)
    end
    primal_solutions = solver.guts[:primal]

    plot_pendulum_iterations(primal_solutions)

    solver
end

end
