abstract type AdjacentKnotPointsFunction end
indices(func::AdjacentKnotPointsFunction) = func.idx
nknots(func::AdjacentKnotPointsFunction) = func.nknots
outputdim(func::AdjacentKnotPointsFunction) = func.outputdim
(func::AdjacentKnotPointsFunction)(Z::DiscreteTrajectory{T}) = error("call on knotpoint(s) not implemented")

abstract type ResultAccumulationMethod end
struct Sum <: ResultAccumulationMethod end
struct Concatenate <: ResultAccumulationMethod end

# Evaluation
function (func::AdjacentKnotPointsFunction)(::Sum, Z::DiscreteTrajectory{T}) where {T}
    result = zero(T)
    for i₀ ∈ indices(func)
        i₁ = i₀ + (nknots(func) - 1)
        z = @view Z[i₀:i₁]
        result += func(z, nstates(Z))
    end
    result
end

function (func::AdjacentKnotPointsFunction)(::Concatenate, Z::DiscreteTrajectory{T}) where {T}
    outputs = map(indices(func)) do i₀
        i₁ = i₀ + (nknots(func) - 1)
        z = @view Z[i₀:i₁]
        func(z)
    end
    vcat(outputs...)
end

# Gradient
function gradient!(▽f::AbstractVector{T}, func::SingleKnotPointFunction, Z::DiscreteTrajectory{T}) where {T}
    z = knotpoints(Z)
    # Rest assured, no copying happening here
    fwrapped(z) = func(DiscreteTrajectory(time(Z), timesteps(Z), z, knotpointsize(Z), nstates(Z))
    ForwardDiff.gradient!(▽f, fwrapped, z)
end

function gradient!(▽f_vstacked::SubArray, func::SingleKnotPointFunction, Z::DiscreteTrajectory{T}) where {T}
    for (i₀, idx) in enumerate(indices(func))
        i₁ = i₀ + nknots(func) - 1
        colrange = knotpointindices(Z, i₀:i₁)
        z = @view Z[i₀:i₁]
        rowview = view(▽f_vstacked, i, colrange)
        gradient!(rowview, func, z)
    end
end

function gradient(funcs::AbstractVector{<:SingleKnotPointFunction}, Z::DiscreteTrajectory{T}) where {T}
    m = sum(length(indices(func)) for func in funcs)
    n = knotpointsize(Z) * length(Z)
    ▽f_vstacked = zeros(Float64, m, n)

    current_row_idx = 1
    for func ∈ funcs
        band_height = length(indices(func))
        band_view = view(▽f_vstacked, current_row_idx:band_height, :)
        gradient!(band_view, func, Z)
        current_row_idx += band_height + 1
    end

    sparse(▽f_vstacked)
end

# Jacobian
function jacobian!(J::AbstractMatrix{T}, func::AdjacentKnotPointsFunction, Z::DiscreteTrajectory{T}) where {T}
    z = knotpoints(Z)
    fwrapped(z) = func(DiscreteTrajectory(time(Z), timesteps(Z), z, knotpointsize(Z), nstates(Z))
    ForwardDiff.jacobian!(J, fwrapped, z)
end

function jacobian!(J_vstacked::AbstractMatrix{T}, func::AdjacentKnotPointsFunction, Z::DiscreteTrajectory{T}) where {T}
    slice_length = nknots(func) * knotpointsize(Z)
    Jheight = outputdim(func)

    for (i₀, idx) in enumerate(indices(func))
        row₀ = (i - 1) * Jheight + 1
        row₁ = row₀ + Jheight - 1
        i₁ = i₀ + nknots(func) - 1
        colrange = knotpointindices(Z, i₀:i₁)
        z = @view Z[i₀:i₁]
        band = @view J_vstacked[row₀:row₁, colrange]
        jacobian!(band, func, z)
    end
end

function jacobian(funcs::AbstractVector{<:AdjacentKnotPointsFunction}, Z::DiscreteTrajectory{T}) where {T}
    m = sum(length(indices(func)) * outputdim(func) for func in funcs)
    n = length(knotpoints(Z))
    J_vstacked = zeros(T, m, n)

    current_row_idx = 1
    for func ∈ funcs
        band_height = length(indices(func)) * outputdim(func)
        band = @view J_vstacked[current_row_idx:band_height, :]
        jacobian!(band, func, Z)
        current_row_idx += band_height
    end

    sparse(J_vstacked)
end

# Hessian
function hessian(H::AbstractMatrix{T}, func::AdjacentKnotPointsFunction,, Z::DiscreteTrajectory{T}) where {T}
    z = knotpoints(Z)
    fwrapped(z) = func(DiscreteTrajectory(time(Z), timesteps(Z), z, knotpointsize(Z), nstates(Z))
    ForwardDiff.hessian!(H, fwrapped, z)
end
