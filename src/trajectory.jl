struct DiscreteTrajectory{Ts,Tk}
    time::AbstractVector{Ts}
    timesteps::AbstractVector{Ts}
    knotpoints::AbstractVector{Tk}
    knotpointsize::Int
    nstates::Int

    function DiscreteTrajectory(
        time::AbstractVector{Ts},
        timesteps::AbstractVector{Ts},
        knotpoints::AbstractVector{Tk},
        knotpointsize::Int,
        nstates::Int,
    ) where {Ts,Tk}
        nk = length(knotpoints) ÷ knotpointsize
        @assert length(time) == length(timesteps) == nk "lengths must match"
        new{Ts,Tk}(time, timesteps, knotpoints, knotpointsize, nstates)
    end
end

time(trajectory::DiscreteTrajectory) = trajectory.time

timesteps(trajectory::DiscreteTrajectory) = trajectory.timesteps

knotpoints(trajectory::DiscreteTrajectory) = trajectory.knotpoints
function knotpoints(
    trajectory::DiscreteTrajectory,
    idx::UnitRange{Int};
    useview::Bool = true,
)
    irange = knotpointindices(trajectory, idx)
    return useview ? view(knotpoints(trajectory), irange) : knotpoints(trajectory)[irange]
end

function knotpointindices(trajectory::DiscreteTrajectory, idx::UnitRange{Int})
    ksize = knotpointsize(trajectory)
    idx₀ = (first(idx) - 1) * ksize + 1
    idx₁ = last(idx) * ksize
    idx₀:idx₁
end
knotpointindices(trajectory::DiscreteTrajectory, idx::Int) =
    knotpointindices(trajectory, idx:idx)
knotpointindex(trajectory::DiscreteTrajectory, idx::Int) =
    (idx - 1) * knotpointsize(trajectory) + 1

knotpointsize(trajectory::DiscreteTrajectory) = trajectory.knotpointsize

nstates(trajectory::DiscreteTrajectory) = trajectory.nstates

function Base.getindex(trajectory::DiscreteTrajectory, idx::UnitRange{Int})
    return DiscreteTrajectory(
        time(trajectory)[idx],
        timesteps(trajectory)[idx],
        knotpoints(trajectory, idx, useview = false),
        knotpointsize(trajectory),
        nstates(trajectory),
    )
end

function Base.view(trajectory::DiscreteTrajectory, idx::UnitRange{Int})
    return DiscreteTrajectory(
        view(time(trajectory), idx),
        view(timesteps(trajectory), idx),
        knotpoints(trajectory, idx, useview = true),
        knotpointsize(trajectory),
        nstates(trajectory),
    )
end

Base.getindex(trajectory::DiscreteTrajectory, idx::Int) = trajectory[idx:idx]
Base.view(trajectory::DiscreteTrajectory, idx::Int) = view(trajectory, idx:idx)

function state(trajectory::DiscreteTrajectory, idx::Int)
    k = knotpoints(trajectory)
    i₀ = knotpointindex(trajectory, idx)
    i₁ = i₀ + nstates(trajectory) - 1
    x = @view k[i₀:i₁]
    x
end
function state(trajectory::DiscreteTrajectory, ::Val{N}) where {N}
    k = knotpoints(trajectory)
    return ntuple(i -> state(trajectory, i), Val(N))
end

function control(trajectory::DiscreteTrajectory, idx::Int)
    k = knotpoints(trajectory)
    xidx₀ = knotpointindex(trajectory, idx)
    i₀ = xidx₀ + nstates(trajectory)
    i₁ = xidx₀ + knotpointsize(trajectory) - 1
    u = @view k[i₀:i₁]
    u
end
function control(trajectory::DiscreteTrajectory, ::Val{N}) where {N}
    k = knotpoints(trajectory)
    nx = nstates(trajectory)
    ksize = knotpointsize(trajectory)
    return ntuple(i -> control(trajectory, i), Val(N))
end
