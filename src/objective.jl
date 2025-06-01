struct LQRCost{T} <: SingleKnotPointFunction
    Q::AbstractMatrix{T}
    R::AbstractMatrix{T}
    xd::AbstractVector{T}
    ud::AbstractVector{T}
    idx::UnitRange{Int}
    nknots::Int
    outputdim::Int

    function LQRCost(Q::AbstractMatrix{T}, R::AbstractMatrix{T},
                     xd::AbstractVector{T}, ud::AbstractVector{T},
                     idx::UnitRange{Int}) where {T}
        @assert size(Q, 1) == size(Q, 2) == length(xd) "Q must have dimensions == length of xd"
        @assert size(R, 2) == length(ud) "columns of R must == length of ud"
        @assert size(R, 1) == length(xd) "rows of R must == length of ud"
        return new{T}(Q, R, xd, ud, idx, 1, 1)
    end

    function LQRCost(Q::AbstractMatrix{T}, R::AbstractMatrix{T},
                     xd::AbstractVector{T}, idx::UnitRange{Int}) where {T}
        @assert size(Q, 1) == size(Q, 2) == length(xd) "Q must have dimensions == length of xd"
        ud = zeros(T, size(R, 2))
        return new{T}(Q, R, xd, ud, idx, 1, 1)
    end
end

function (cost::LQRCost)(z::AbstractVector, nstates::Int)
    x = @view z[1:nstates]
    u = @view z[nstates+1:end]
    x̄ = (x - cost.xd)
    ū = (u - cost.ud)
    1 / 2 * (x̄' * cost.Q * x̄ + ū' * cost.R * ū)
end
