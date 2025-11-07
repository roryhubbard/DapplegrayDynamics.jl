using DapplegrayDynamics
using Test

@testset "DapplegrayDynamics.jl" begin
    acrobot = DD.load_acrobot()
    acrobot_swingup(acrobot, 50, 10.0)
    pendulum = DD.load_pendulum()
    pendulum_swingup(pendulum, 51, 10.0)
    pendulum_swingup_nlopt(pendulum, 51, 10.0, 10)
end
