module DapplegrayDynamics

using Clarabel
using DiffResults
using ForwardDiff
using LinearAlgebra
using RigidBodyDynamics
using SparseArrays
using StaticArrays

# potentially remove later
using GLMakie
using Ipopt
using JuMP
# https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/
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

export acrobot_swingup, df, pendulum_swingup, pendulum_swingup_nlopt, kj, trynlopt

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
        SeparatedHermiteSimpsonConstraint(mechanism, 1:(N-2), [1]),
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

function pendulum_swingup_nlopt(mechanism::Mechanism, N::Int, tf::AbstractFloat)
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

    # Create cost and constraint objects using existing functions
    lqr_cost = LQRCost(Q, R, xf, 1:N)

    equality_constraints = [
        CompressedHermiteSimpsonConstraint(mechanism, 1:(N-1), [1]),
        state_equality_constraint(x0, knotpointsize, 1),
        state_equality_constraint(xf, knotpointsize, N),
    ]

    inequality_constraints = [
        control_bound_constraint(knotpointsize, 1:(N-1), [τbound], [-τbound])
    ]

    # Trace iterations (following NLopt-README pattern)
    trace = Any[]

    # Objective function using existing LQRCost (following your fwrapped pattern)
    function objective_fn_raw(z::Vector)
        Z = DiscreteTrajectory(time(initial_traj), timesteps(initial_traj), z, knotpointsize, nx)
        return evaluate_objective([lqr_cost], Z)
    end

    # Wrap with ForwardDiff for automatic gradient computation (from NLopt-README)
    function objective_fn(z::Vector, grad::Vector)
        if length(grad) > 0
            ForwardDiff.gradient!(grad, objective_fn_raw, z)
        end
        value = objective_fn_raw(z)
        # Store iteration trace (following NLopt-README section on trace iterations)
        push!(trace, copy(z) => value)
        return value
    end

    # Equality constraints function (following your fwrapped pattern)
    function equality_constraints_raw(z::Vector)
        Z = DiscreteTrajectory(time(initial_traj), timesteps(initial_traj), z, knotpointsize, nx)
        return evaluate_constraints(equality_constraints, Z)
    end

    n_eq_constraints = sum(outputdim(con) * length(indices(con)) for con in equality_constraints)

    function equality_constraints_fn(result::Vector, z::Vector, grad::Matrix)
        result[:] = equality_constraints_raw(z)
        if length(grad) > 0
            jac = ForwardDiff.jacobian(equality_constraints_raw, z)
            grad[:, :] = jac'
        end
        return
    end

    # Inequality constraints function
    function inequality_constraints_raw(z::Vector)
        Z = DiscreteTrajectory(time(initial_traj), timesteps(initial_traj), z, knotpointsize, nx)
        return evaluate_constraints(inequality_constraints, Z)
    end

    n_ineq_constraints = sum(outputdim(con) * length(indices(con)) for con in inequality_constraints)

    function inequality_constraints_fn(result::Vector, z::Vector, grad::Matrix)
        result[:] = inequality_constraints_raw(z)
        if length(grad) > 0
            jac = ForwardDiff.jacobian(inequality_constraints_raw, z)
            grad[:, :] = jac'
        end
        return
    end

    # Create NLopt optimizer with NLOPT_LD_SLSQP algorithm
    opt = NLopt.Opt(:LD_SLSQP, num_vars)

    # Set objective
    NLopt.min_objective!(opt, objective_fn)

    # Add equality constraints
    NLopt.equality_constraint!(opt, equality_constraints_fn, fill(1e-8, n_eq_constraints))

    # Add inequality constraints (control bounds)
    NLopt.inequality_constraint!(opt, inequality_constraints_fn, fill(1e-8, n_ineq_constraints))

    # Set stopping criteria
#    NLopt.xtol_rel!(opt, 1e-6)
#    NLopt.maxeval!(opt, 10000)

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

    # Return as DiscreteTrajectory for compatibility with existing plotting code
    solution_traj = DiscreteTrajectory(
        time(initial_traj),
        timesteps(initial_traj),
        min_z,
        knotpointsize,
        nx
    )

    # Convert trace to DiscreteTrajectory objects for plotting (like kj() uses solver.guts[:primal])
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

function trynlopt()
    mechanism = load_pendulum()
    result = pendulum_swingup_nlopt(mechanism, 50, 10.0)
    plot_pendulum_iterations(result.primal_solutions)
    result
end

function kj()
    mechanism = load_pendulum()
    solver = pendulum_swingup(mechanism, 50, 10.0)

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
