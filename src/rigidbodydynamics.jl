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

function simulate_mechanism(mechansim::Mechanism, finaltime::Real, Δt::Real, initial_configuration::AbstractVector, initial_velocity::AbstractVector)
    state = MechanismState(mechansim)
    set_configuration!(state, initial_configuration)
    set_velocity!(state, initial_velocity)

    simulate(state, finaltime, Δt=Δt);
end
