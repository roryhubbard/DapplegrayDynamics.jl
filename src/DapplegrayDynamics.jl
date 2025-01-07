module DapplegrayDynamics

using DifferentialEquations: ODEProblem, solve, Tsit5

export pendulum, simulate_pendulum

struct Pendulum{T}
    mass::T
    len::T
    b::T
    lc::T
    I::T
    g::T
end
function Pendulum(;mass=1., len=0.5, b=0.1, lc=0.5, I=0.25, g=9.81)
    T = eltype(promote(mass, len, b, lc, I, g))
    Pendulum{T}(mass, len, b, lc, I, g)
end

pendulum = Pendulum()
u0 = [π/4; 0.0]
tspan = (0.0, 4π)

function dynamics!(du, u, p, t)
    pm = pendulum
    m = pm.mass * pm.lc^2
    θ = u[1]
    dθ = u[2]
    du[1] = dθ
    du[2] = θ/m - pm.g * sin(θ) / pm.len - pm.b * dθ / m
end

function simulate_pendulum()
    prob = ODEProblem(dynamics!, u0, tspan)
    sol = solve(prob, Tsit5())
    sol
end

end
