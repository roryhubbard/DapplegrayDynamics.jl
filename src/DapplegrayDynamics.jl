module DapplegrayDynamics

using Clarabel
using DiffResults
using ForwardDiff
using LinearAlgebra
using ProgressMeter
using RigidBodyDynamics
using SparseArrays
using StaticArrays

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

export acrobot_swingup, pendulum_swingup, pendulum_swingup_nlopt

function setup_swingup_problem(
    mechanism::Mechanism,
    N::Int,
    tf::AbstractFloat;
    nu::Int = 1,
    Q_weight::Float64 = 0.1,
    R_weight::Float64 = 1.0,
    τbound::Float64 = 3.0,
)
    nq = num_positions(mechanism)
    nv = num_velocities(mechanism)
    nx = nq + nv
    knotpointsize = nx + nu
    Δt = tf / (N - 1)

    x0 = zeros(nx)
    Q = Q_weight * I(nx) * Δt
    R = R_weight * I(nu) * Δt

    return (
        nq = nq,
        nv = nv,
        nx = nx,
        nu = nu,
        knotpointsize = knotpointsize,
        Δt = Δt,
        x0 = x0,
        Q = Q,
        R = R,
        τbound = τbound,
    )
end

function create_boundary_constraints(x0, xf, knotpointsize, N)
    return [
        state_equality_constraint(x0, knotpointsize, 1),
        state_equality_constraint(xf, knotpointsize, N),
    ]
end

function create_control_bounds(knotpointsize, N, τbound)
    return [control_bound_constraint(knotpointsize, 1:(N-1), [τbound], [-τbound])]
end

function acrobot_swingup(mechanism::Mechanism, N::Int, tf::AbstractFloat)
    prob = setup_swingup_problem(mechanism, N, tf)

    xf = [π, 0, 0, 0]  # swing up

    objectives = [LQRCost(prob.Q, prob.R, xf, 1:N)]

    inequality_constraints = create_control_bounds(prob.knotpointsize, N, prob.τbound)

    equality_constraints = [
        CompressedHermiteSimpsonConstraint(mechanism, 1:(N-1), [2]),
        create_boundary_constraints(prob.x0, xf, prob.knotpointsize, N)...,
    ]

    initial_solution = initialize_trajectory(
        mechanism,
        N,
        tf,
        prob.nu,
        zeros(typeof(tf), prob.nq),
        [π, 0.0],
        zeros(typeof(tf), prob.nv),
        zeros(typeof(tf), prob.nv),
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

function pendulum_swingup(mechanism::Mechanism, N::Int, tf::AbstractFloat, maxeval::Int)
    prob = setup_swingup_problem(mechanism, N, tf)

    xf = [π, 0]  # swing up

    objectives = [LQRCost(prob.Q, prob.R, xf, 1:N)]

    inequality_constraints = create_control_bounds(prob.knotpointsize, N, prob.τbound)

    @assert isodd(N) "N needs to be odd for SeparatedHermiteSimpsonConstraint but it is $N"
    equality_constraints = [
        [SeparatedHermiteSimpsonConstraint(mechanism, i, [1]) for i = 1:2:(N-2)]...,
        create_boundary_constraints(prob.x0, xf, prob.knotpointsize, N)...,
    ]

    initial_solution = initialize_trajectory(
        mechanism,
        N,
        tf,
        prob.nu,
        zeros(typeof(tf), prob.nq),
        [Float64(π)],
        zeros(typeof(tf), prob.nv),
        zeros(typeof(tf), prob.nv),
    )

    solver = SQPSolver(
        mechanism,
        objectives,
        equality_constraints,
        inequality_constraints,
        initial_solution,
        nothing,
        nothing,
        nothing,
        OuterSettings(max_iter=maxeval),
    )

    solve!(solver)

    solver
end

function pendulum_swingup_nlopt(
    mechanism::Mechanism,
    N::Int,
    tf::AbstractFloat,
    maxeval::Int,
)
    prob = setup_swingup_problem(mechanism, N, tf)

    # Final state
    xf = [π, 0.0]  # swing up

    # Total number of decision variables
    num_vars = N * prob.knotpointsize

    # Create initial trajectory to get time/timestep structure
    initial_traj = initialize_trajectory(
        mechanism,
        N,
        tf,
        prob.nu,
        zeros(typeof(tf), prob.nq),
        [Float64(π)],
        zeros(typeof(tf), prob.nv),
        zeros(typeof(tf), prob.nv),
    )

    # Create cost and constraint objects
    lqr_cost = LQRCost(prob.Q, prob.R, xf, 1:N)
    objective = [lqr_cost]

    @assert isodd(N) "N needs to be odd for SeparatedHermiteSimpsonConstraint but it is $N"
    equality_constraints = [
        [SeparatedHermiteSimpsonConstraint(mechanism, i, [1]) for i = 1:2:(N-2)]...,
        create_boundary_constraints(prob.x0, xf, prob.knotpointsize, N)...,
    ]
    # ignored but required by super_hessian_constraints
    v = zeros(Float64, num_lagrange_multipliers(equality_constraints))

    inequality_constraints = create_control_bounds(prob.knotpointsize, N, prob.τbound)
    # ignored but required by super_hessian_constraints
    λ = zeros(Float64, num_lagrange_multipliers(inequality_constraints))

    # Trace iterations (following NLopt README pattern)
    trace = Any[]

    pbar = Progress(maxeval; desc = "Running solver")
    function objective_fn(z::Vector, grad::Vector)
        Z = DiscreteTrajectory(
            time(initial_traj),
            timesteps(initial_traj),
            z,
            prob.knotpointsize,
            prob.nx,
        )
        f, ▽f, ▽²f = super_hessian_objective(objective, Z)
        if length(grad) > 0
            grad[:] = ▽f
        end
        push!(trace, copy(z) => f)
        next!(pbar)
        return f
    end

    function equality_constraints_fn(result::Vector, z::Vector, grad::Matrix)
        Z = DiscreteTrajectory(
            time(initial_traj),
            timesteps(initial_traj),
            z,
            prob.knotpointsize,
            prob.nx,
        )
        h, ▽h, ▽²h = super_hessian_constraints(equality_constraints, Z, v)
        result[:] = h
        if length(grad) > 0
            grad[:, :] = ▽h'
        end
    end

    function inequality_constraints_fn(result::Vector, z::Vector, grad::Matrix)
        Z = DiscreteTrajectory(
            time(initial_traj),
            timesteps(initial_traj),
            z,
            prob.knotpointsize,
            prob.nx,
        )
        g, ▽g, ▽²g = super_hessian_constraints(inequality_constraints, Z, λ)
        result[:] = g
        if length(grad) > 0
            grad[:, :] = ▽g'
        end
    end

    # Create NLopt optimizer with NLOPT_LD_SLSQP algorithm
    opt = NLopt.Opt(:LD_SLSQP, num_vars)

    NLopt.min_objective!(opt, objective_fn)

    NLopt.equality_constraint!(opt, equality_constraints_fn, fill(1e-8, length(v)))

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
        prob.knotpointsize,
        prob.nx,
    )

    primal_solutions = [
        DiscreteTrajectory(
            time(initial_traj),
            timesteps(initial_traj),
            z_vec,
            prob.knotpointsize,
            prob.nx,
        ) for (z_vec, obj_val) in trace
    ]

    return (
        optimizer = opt,
        solution = solution_traj,
        objective_value = min_f,
        return_code = ret,
        num_evals = num_evals,
        trace = trace,
        primal_solutions = primal_solutions,
    )
end

function load_pendulum()::Mechanism
    srcdir = dirname(pathof(DapplegrayDynamics))
    urdf = joinpath(srcdir, "..", "test", "urdf", "pendulum.urdf")
    parse_urdf(urdf)
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

end
