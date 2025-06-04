struct DiscreteTrajectory{T}
    time::AbstractVector{T}
    timesteps::AbstractVector{T}
    knotpoints::AbstractVector{T}
    knotpointsize::Int
    nstates::Int
    function DiscreteTrajectory(time::AbstractVector{T}, timesteps::AbstractVector{T}, knotpoints::AbstractVector{T}, knotpointsize::Int, nstates::Int) where {T}
        nk = length(knotpoints) ÷ knotpointsize
        @assert length(time) == length(timesteps) == nk "lengths must match"
        new{T}(time, timesteps, knotpoints, knotpointsize, nstates)
    end
end

time(trajectory::DiscreteTrajectory{T}) where {T} = trajectory.time

timesteps(trajectory::DiscreteTrajectory{T}) where {T} = trajectory.timesteps

knotpoints(trajectory::DiscreteTrajectory{T}) where {T} = trajectory.knotpoints
function knotpoints(trajectory::DiscreteTrajectory{T}, idx::UnitRange{Int}; useview::Bool=true) where {T}
    irange = knotpointindices(trajectory, idx)
    return useview ? view(knotpoints(trajectory), irange) : knotpoints(trajectory)[irange]
end

function knotpointindices(trajectory::DiscreteTrajectory{T}, idx::UnitRange{Int}) where {T}
    ksize = knotpointsize(trajectory)
    idx₀ = (first(idx) - 1) * ksize + 1
    idx₁ = last(idx) * ksize
    idx₀:idx₁
end
knotpointindices(trajectory::DiscreteTrajectory{T}, idx::Int) where {T} = knotpointindices(trajectory, idx:idx)
knotpointindex(trajectory::DiscreteTrajectory{T}, idx::Int) where {T} = (idx - 1) * knotpointsize(trajectory) + 1

knotpointsize(trajectory::DiscreteTrajectory{T}) where {T} = trajectory.knotpointsize

nstates(trajectory::DiscreteTrajectory{T}) where {T} = trajectory.nstates

function Base.getindex(trajectory::DiscreteTrajectory{T}, idx::UnitRange{Int}) where {T}
    return DiscreteTrajectory(
        time(trajectory)[idx],
        timesteps(trajectory)[idx],
        knotpoints(trajectory, idx, useview=false),
        knotpointsize(trajectory),
        nstates(trajectory)
    )
end

function Base.view(trajectory::DiscreteTrajectory{T}, idx::UnitRange{Int}) where {T}
    return DiscreteTrajectory(
        view(time(trajectory), idx),
        view(timesteps(trajectory), idx),
        knotpoints(trajectory, idx, useview=true),
        knotpointsize(trajectory),
        nstates(trajectory)
    )
end

Base.getindex(trajectory::DiscreteTrajectory{T}, idx::Int) where {T} = trajectory[idx:idx]
Base.view(trajectory::DiscreteTrajectory{T}, idx::Int) where {T} = view(trajectory, idx:idx)

function state(trajectory::DiscreteTrajectory{T}, idx::Int) where {T}
    k = knotpoints(trajectory)
    i₀ = knotpointindex(trajectory, idx)
    i₁ = i₀ + nstates(trajectory) - 1
    x = @view k[i₀:i₁]
    x
end
function state(trajectory::DiscreteTrajectory{T}, ::Val{N}) where {T, N}
    k = knotpoints(trajectory)
    return ntuple(i -> state(trajectory, i), Val(N))
end

function control(trajectory::DiscreteTrajectory{T}, idx::Int) where {T}
    k = knotpoints(trajectory)
    xidx₀ = knotpointindex(trajectory, idx)
    i₀ = xidx₀ + nstates(trajectory)
    i₁ = xidx₀ + knotpointsize(trajectory) - 1
    u = @view k[i₀:i₁]
    u
end
function control(trajectory::DiscreteTrajectory{T}, ::Val{N}) where {T, N}
    k = knotpoints(trajectory)
    nx = nstates(trajectory)
    ksize = knotpointsize(trajectory)
    return ntuple(i -> control(trajectory, i), Val(N))
end
