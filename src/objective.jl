struct LQRCost <: SingleKnotPointFunction
    Q::AbstractMatrix
    R::AbstractMatrix
    xd::AbstractVector
    ud::AbstractVector
    idx::UnitRange{Int}
    nknots::Int
    outputdim::Int
end
function LQRCost(Q::AbstractMatrix, R::AbstractMatrix, xd::AbstractVector, idx::UnitRange{Int})
    ud = zeros(size(R, 2))
    LQRCost(Q, R, xd, ud, idx, 1, 1)
end
function (cost::LQRCost)(z::AbstractVector, nstates::Int)
    x = @view z[1:nstates]
    u = @view z[nstates+1:end]
    x̄ = (x - cost.xd)
    ū = (u - cost.ud)
    1 / 2 * (x̄' * cost.Q * x̄ + ū' * cost.R * ū)
end
