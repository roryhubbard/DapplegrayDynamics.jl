struct LQRCost <: SingleKnotPointFunction
    Q::AbstractMatrix
    R::AbstractMatrix
    xd::AbstractVector
    ud::AbstractVector
    idx::UnitRange{Int}
end
nknots(::LQRCost) = 1
outputdim(::LQRCost) = 1
outputtype(::LQRCost) = ScalarOutput()
function LQRCost(Q::AbstractMatrix, R::AbstractMatrix, xd::AbstractVector, idx::UnitRange{Int})
    ud = zeros(size(R, 2))
    LQRCost(Q, R, xd, ud, idx)
end
function (cost::LQRCost)(z::AbstractVector, nstates::Int)
    x = @view z[1:nstates]
    u = @view z[nstates+1:end]
    x̄ = (x - cost.xd)
    ū = (u - cost.ud)
    1 / 2 * (x̄' * cost.Q * x̄ + ū' * cost.R * ū)
end

struct StateCost <: StateFunction
    Q::AbstractMatrix
    xd::AbstractVector
    idx::UnitRange{Int}
end
outputtype(::StateCost) = ScalarOutput()
outputdim(::StateCost) = 1
function (cost::StateCost)(z::AbstractVector, nstates::Int)
    x = @view z[1:nstates]
    x̄ = (x - cost.xd)
    x̄' * cost.Q * x̄
end

struct ControlCost <: ControlFunction
    R::AbstractMatrix
    idx::UnitRange{Int}
end
outputtype(::ControlCost) = ScalarOutput()
outputdim(::ControlCost) = 1
function (cost::ControlCost)(z::AbstractVector, nstates::Int)
    u = @view z[nstates+1:end]
    u' * cost.R * u
end
