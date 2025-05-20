module DapplegrayDynamics

using Clarabel
using ForwardDiff
using LinearAlgebra
using RigidBodyDynamics
using SparseArrays
using StaticArrays

export swingup, doublependulum


function doublependulum()::Mechanism
    g = -9.81 # gravitational acceleration in z-direction
    world = RigidBody{Float64}("world")
    doublependulum = Mechanism(world; gravity = SVector(0, 0, g))

    axis = SVector(0., 1., 0.) # joint axis
    I_1 = 0.333 # moment of inertia about joint axis
    c_1 = -0.5 # center of mass location with respect to joint axis
    m_1 = 1. # mass
    frame1 = CartesianFrame3D("upper_link") # the reference frame in which the spatial inertia will be expressed
    inertia1 = SpatialInertia(frame1,
        moment=I_1 * axis * axis',
        com=SVector(0, 0, c_1),
        mass=m_1)

    upperlink = RigidBody(inertia1)
    shoulder = Joint("shoulder", Revolute(axis))
    before_shoulder_to_world = one(Transform3D,
        frame_before(shoulder), default_frame(world))
    attach!(doublependulum, world, upperlink, shoulder,
        joint_pose = before_shoulder_to_world)

    l_1 = -1. # length of the upper link
    I_2 = 0.333 # moment of inertia about joint axis
    c_2 = -0.5 # center of mass location with respect to joint axis
    m_2 = 1. # mass
    inertia2 = SpatialInertia(CartesianFrame3D("lower_link"),
        moment=I_2 * axis * axis',
        com=SVector(0, 0, c_2),
        mass=m_2)
    lowerlink = RigidBody(inertia2)
    elbow = Joint("elbow", Revolute(axis))
    before_elbow_to_after_shoulder = Transform3D(
        frame_before(elbow), frame_after(shoulder), SVector(0, 0, l_1))
    attach!(doublependulum, upperlink, lowerlink, elbow,
        joint_pose = before_elbow_to_after_shoulder)

    doublependulum
end

function simulate_doublependulum()
    doublependulum = doublependulum()
    state = MechanismState(doublependulum)
    set_configuration!(state, shoulder, 0.3)
    set_configuration!(state, elbow, 0.4)
    set_velocity!(state, shoulder, 1.)
    set_velocity!(state, elbow, 2.)

    ts, qs, vs = simulate(state, 5., Δt = 1e-3);
end

struct HermiteSimpsonConstraint{M,T}
    model::M
    Δt::T
end

#function hermite_simpson_compressed(model::Mechanism, Δt, xₖ, uₖ, xₖ₊₁, uₖ₊₁)
#    fₖ = RobotDynamics.evaluate(model, xₖ, uₖ)
#    fₖ₊₁ = RobotDynamics.evaluate(model, xₖ₊₁, uₖ₊₁)
#
#    # We could add the collocation point as an extra decision varaible and
#    # constraint. This would be "separated form". Here we are implementing
#    # "compressed form" where we calculate `fcol` and jam it into the constraint
#    # for the integral of the system dynamics.
#    xcol = 0.5 * (xₖ + xₖ₊₁) + Δt / 8 * (fₖ - fₖ₊₁)
#    ucol = 0.5 * (uₖ + uₖ₊₁)
#    fcol = RobotDynamics.evaluate(model, xcol, ucol)
#
#    # equality constraint: xₖ₊₁ - xₖ = (Δt / 6) * (fₖ + 4fcol + fₖ₊₁)
#    SVector{length(xₖ)}(xₖ₊₁ - xₖ - (Δt / 6) * (fₖ + 4fcol + fₖ₊₁))
#end

function evaluate(
    con::HermiteSimpsonConstraint,
    xₖ::AbstractVector,
    uₖ::AbstractVector,
    xₖ₊₁::AbstractVector,
    uₖ₊₁::AbstractVector,
)
    hermite_simpson_compressed(con.model, con.Δt, xₖ, uₖ, xₖ₊₁, uₖ₊₁)
end

function evaluate!(
    con::HermiteSimpsonConstraint,
    c::AbstractVector,
    xₖ::AbstractVector,
    uₖ::AbstractVector,
    xₖ₊₁::AbstractVector,
    uₖ₊₁::AbstractVector,
)
    copyto!(c, hermite_simpson_compressed(con.model, con.Δt, xₖ, uₖ, xₖ₊₁, uₖ₊₁))
    c
end

struct DapplegraySQP
#    opts::SolverOptions{T}
#    stats::SolverStats{T}
#    problem::Problem
end

#function build_lagrangian{T}(
#    𝒇::TrajectoryOptimization.AbstractObjective,
#    𝒉::Vector{TrajectoryOptimization.AbstractConstraint},
#    𝒈::Vector{TrajectoryOptimization.AbstractConstraint},
#    𝒗::Vector{T},
#    𝝀::Vector{T},
#)
#    𝒇 + 𝒗'𝒉 + 𝝀'𝒈
#end

#function apply_constraint(K, constraintindices, constraint)
#    println("################### CONSTRAINT ###################")
#    println(constraint)
#
#    T = typeof(constraint)
#    println("type: ", T)
#
#    p = RobotDynamics.output_dim(constraint)
#    println("output_dim: ", p)
#
#    input_dim = RobotDynamics.input_dim(constraint)
#    println("input_dim: ", input_dim)
#
#    input_type = RobotDynamics.functioninputs(constraint)
#    println("input_type: ", input_type)
#
#    sense = TrajectoryOptimization.sense(constraint)
#    println("sense: ", sense)
#
#    for j ∈ constraintindices
#        println("j: ", j)
#
#        k = K[j]
#        x = RobotDynamics.state(k)
#        n = RobotDynamics.state_dim(k)
#        u = RobotDynamics.control(k)
#        m = RobotDynamics.control_dim(k)
#        println("state: $n $x")
#        println("control: $m $u")
#
#        y = RobotDynamics.evaluate(constraint, k)
#        println("evaluate: ", y)
#
#        𝑱 = Matrix{Float64}(undef, p, input_dim)
#        y = Vector{Float64}(undef, p)
#        RobotDynamics.jacobian!(constraint, 𝑱, y, k)
#        println("jacobian: ", 𝑱)
#
#        𝑯 = Matrix{Float64}(undef, input_dim, input_dim)
#        𝝀 = zeros(p) # TODO: get this the right way
#        z_ref = RobotDynamics.getinput(input_type, k)  # this will be x, u, or [x; u]
#        f(zvec) = RobotDynamics.evaluate(constraint, zvec)
#        for i = 1:p
#            fᵢ(zvec) = f(zvec)[i]  # scalar function
#            Hᵢ = ForwardDiff.hessian(fᵢ, z_ref)
#            print("row hessian: ", Hᵢ)
#            𝑯 += 𝝀[i] .* Hᵢ
#        end
#        println("sum of hessians: ", 𝑯)
#    end
#end
#
#
#function solve!(solver::DapplegraySQP)
#    for _ = 1:10 # TODO: repeat until convergence criteria is met
#        𝒇 = get_objective(solver.problem)
#        constraints = get_constraints(solver.problem)
#
#        Z = get_trajectory(solver.problem)
#        println("trajectory: ", Z)
#
#        K = RobotDynamics.getdata(Z)
#        println("K: ", K)
#
#        X = states(Z)
#        println("X: ", X)
#
#        U = controls(Z)
#        println("U: ", U)
#
#        times = gettimes(Z)
#        println("times: ", times)
#        println()
#
#        for (constraintindices, constraint) ∈ zip(constraints)
#            apply_constraint(Z, constraintindices, constraint)
#            println()
#        end
#
#        return
#
#        𝒉 = equality_constraints(constraints)
#        𝒈 = inequality_constraints(constraints)
#        𝒗 = equality_dual_vector(solver)
#        𝝀 = inequality_dual_vector(solver)
#        ℒ = build_lagrangian(𝒇, 𝒉, 𝒈, 𝒗, 𝝀)
#        ▽ₓ𝒇 = gradient(𝒇)
#        𝑱ₓ𝒉 = jacobian(𝒉)
#        𝑱ₓ𝒈 = jacobian(𝒈)
#        # ▽ₓℒ = gradiant(ℒ)
#        ▽ₓℒ = ▽ₓ𝒇 + 𝑱ₓ𝒉'𝒗 + 𝑱ₓ𝒈'𝝀
#        ▽²ₓₓℒ = hessian(▽ₓℒ)
#
#        """
#        Solve QP using Clarabel
#
#        minimize   1⁄2𝒙ᵀ𝑷𝒙 + 𝒒ᵀ𝒙
#        subject to  𝑨𝒙 + 𝒔 = 𝒃
#                         𝒔 ∈ 𝑲
#        with decision variables 𝒙 ∈ ℝⁿ, 𝒔 ∈ 𝑲 and data matrices 𝑷 = 𝑷ᵀ ≥ 0,
#        𝒒 ∈ ℝⁿ, 𝑨 ∈ ℝᵐˣⁿ, and b ∈ ℝᵐ. The convext set 𝑲 is a composition of convex cones.
#        """
#        𝑷 = sparse(▽²ₓₓℒ)
#        𝒒 = sparse(▽ₓℒ)
#        𝑨 = sparse([𝑱ₓ𝒉;
#                    𝑱ₓ𝒈;
#                    ])
#        𝒃 = [-𝒉;
#             -𝒈]
#        𝑲 = [
#            Clarabel.ZeroConeT(length(𝒉)),
#            Clarabel.NonnegativeConeT(length(𝒈))]
#
#        settings = Clarabel.Settings()
#        solver   = Clarabel.Solver()
#        Clarabel.setup!(solver, 𝑷, 𝒒, 𝑨, 𝒃, 𝑲, settings)
#        result = Clarabel.solve!(solver)
#        𝚫𝒙ₖ₊₁, 𝒗ₖ₊₁, 𝝀ₖ₊₁ = unpack_result(result)
#
#        nudge_𝒙!(solver, 𝚫𝒙ₖ₊₁)
#        set_𝒗!(solver, 𝒗ₖ₊₁)
#        set_𝝀!(solver, 𝝀ₖ₊₁)
#    end
#end

lqr_cost(Q,R) = (x,u) -> x'Q*x + u'R*u
terminal_cost(Q) = (x,_) -> x'Q*x
## hessian w.r.t to only state even though function sig takes u too
#dummy_u = nothing              # can also be `SVector{0,Float64}()`
#f_x = x -> cost(x, dummy_u)    # Rⁿ → ℝ  (u is fixed & unused)
## current state at knot-point k
#x_k = @SVector randn(n)        # replace with your real state
## Hessian w.r.t. x ONLY:   n × n
#H = ForwardDiff.hessian(f_x, x_k)

function evaluatecosts(costs, x, u)
    for (f, idx) in costs
        # TODO: this indexing is going to cause issues with how the cost
        # functions are currently defined
        f(x[idx], u[idx])
    end
end

function swingup(method::Symbol = :sqp)
    model = doublependulum()
    n = 4 # state dimension
    m = 1 # control dimension

    N = 2
    tf = 2.0           # final time (sec)
    dt = tf / (N - 1)  # time step (sec)

    # Objective
    x0 = @SVector zeros(n)
    xf = @SVector [π, 0, 0, 0]  # swing up

    Q = 0.01 * Diagonal(@SVector ones(n)) * dt
    Qf = 100.0 * Diagonal(@SVector ones(n))
    R = 0.1 * Diagonal(@SVector ones(m)) * dt

    objective = [
        (lqr_cost(Q,R),   1:N-1),
        (terminal_cost(Qf), N),
    ]

    # Create constraints
    constraints = [] 

#    # Terminal goal constraint
#    goalcon = GoalConstraint(xf)
#    add_constraint!(constraints, goalcon, N)
#
#    # Control bounds
#    ubnd = 3.0
#    bnd = ControlBound(m, u_min = -ubnd, u_max = ubnd)
##    bnd = BoundConstraint(n, m, u_min = -ubnd, u_max = ubnd)
#    add_constraint!(constraints, bnd, 1:N-1)
#
#    ################## GREAT BARRIER ##################
#
#    # Construct problem depending on method
#    prob = if method == :altro
#        Problem(model, objective, x0, tf; constraints = constraints)
#    elseif method == :sqp
#        collocation_constraints = HermiteSimpsonConstraint(model, dt)
#        add_constraint!(constraints, collocation_constraints, 1:N-1)
#        Problem(model, objective, x0, tf; constraints = constraints)
#    else
#        error("Unsupported method: $method. Choose :altro or :sqp.")
#    end
#
#    # Construct solver depending on method
#    solver = if method == :altro
#        opts = SolverOptions(
#            cost_tolerance_intermediate = 1e-2,
#            penalty_scaling = 10.0,
#            penalty_initial = 1.0,
#        )
#        ALTROSolver(prob, opts)
#    elseif method == :sqp
#        DapplegraySQP(prob)
#    else
#        error("Unsupported method: $method. Choose :altro or :sqp.")
#    end
#
#    # Initialization
#    u0 = @SVector fill(0.01, m)
#    U0 = [u0 for _ = 1:N-1]
#    initial_controls!(prob, U0)
#    rollout!(prob)
#
##    set_options!(solver, show_summary = true)
#    solve!(solver)
#
#    prob
end

end
