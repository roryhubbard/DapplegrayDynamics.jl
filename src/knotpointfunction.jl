abstract type AdjacentKnotPointsFunction end
indices(func::AdjacentKnotPointsFunction) = func.idx
nknots(func::AdjacentKnotPointsFunction) = func.nknots
outputdim(func::AdjacentKnotPointsFunction) = func.outputdim
(::AdjacentKnotPointsFunction)(::AbstractVector) = error("call on knotpoint(s) not implemented")

abstract type ResultAccumulationMethod end
struct Sum <: ResultAccumulationMethod end
struct Concatenate <: ResultAccumulationMethod end

function (func::AdjacentKnotPointsFunction)(::Sum, Z::DiscreteTrajectory)
    result = 0.0
    for i ∈ indices(func)
        idx₀ = (i - 1) * knotpointsize(Z) + 1
        slice_length = nknots(func) * knotpointsize(Z)
        result += func(Z[idx₀:idx₀ + slice_length - 1], nstates(Z))
    end
    result
end
function (func::AdjacentKnotPointsFunction)(::Concatenate, Z::DiscreteTrajectory)
    outputs = map(indices(func)) do i
        idx₀ = (i - 1) * knotpointsize(Z) + 1
        slice_length = nknots(func) * knotpointsize(Z)
        z = @view Z[idx₀:idx₀ + slice_length - 1]
        func(z, nstates(Z))
    end
    vcat(outputs...)
end
function gradient(func::AdjacentKnotPointsFunction, z::AbstractVector, nstates::Int)
    fwrapped(z) = func(z, nstates)
    ForwardDiff.gradient(fwrapped, z)
end
function jacobian!(J::SubArray, func::AdjacentKnotPointsFunction, z::AbstractVector, nstates::Int)
    fwrapped(z) = func(z, nstates)
    ForwardDiff.jacobian!(J, fwrapped, z)
end
function jacobian!(J_vstacked::SubArray, func::AdjacentKnotPointsFunction, Z::DiscreteTrajectory)
    Zindices = indices(func)
    slice_length = nknots(func) * knotpointsize(Z)
    nstates = nstates(Z)
    Jheight = outputdim(func)

    for (i, idx) in enumerate(Zindices)
        row₀ = (i - 1) * Jheight + 1
        col₀ = kdim * (idx - 1) + 1
        band_view = view(J_vstacked, row₀:row₀ + Jheight - 1, col₀:col₀ + slice_length - 1)
        z = view(Z[col₀:col₀ + slice_length - 1])
        jacobian!(band_view, func, z, nstates)
    end
end
function jacobian(funcs::AbstractVector{<:AdjacentKnotPointsFunction}, Z::DiscreteTrajectory) where {T}
    ksize = knotpointsize(Z)
    m = sum(length(indices(func)) * outputdim(func) for func in funcs)
    n = length(knotpoints(Z))
    J_vstacked = zeros(T, m, n)

    current_row_idx = 1
    for func ∈ funcs
        band_height = length(indices(func)) * outputdim(func)
        band_view = view(J_vstacked, current_row_idx:band_height, :)
        jacobian!(band_view, func, knotpoints)
        current_row_idx += band_height
    end

    sparse(J_vstacked)
end
function hessian(func::AdjacentKnotPointsFunction, z::AbstractVector, nstates::Int)
    fwrapped(z) = func(z, nstates)
    ForwardDiff.hessian(fwrapped, z)
end
