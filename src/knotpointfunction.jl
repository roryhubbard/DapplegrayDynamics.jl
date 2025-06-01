abstract type FunctionOutput end
struct ScalarOutput <: FunctionOutput end
struct VectorOutput <: FunctionOutput end

abstract type AdjacentKnotPointsFunction end
indices(func::AdjacentKnotPointsFunction) = func.idx
nknots(::AdjacentKnotPointsFunction) = error("nknots not defined")
outputtype(::AdjacentKnotPointsFunction) = error("outputtype not defined")
outputdim() = error("outputdim not defined")
(::AdjacentKnotPointsFunction)(::AbstractVector) = error("call on knotpoint(s) not implemented")

function (func::AdjacentKnotPointsFunction)(::ScalarOutput, Z::DiscreteTrajectory)
    result = 0.0
    for i ∈ indices(func)
        result += func(Z[i:i + knotpointsize(Z)])
    end
    result
end
function (func::AdjacentKnotPointsFunction)(::VectorOutput, Z::DiscreteTrajectory)
    vcat(map(idx -> func(Z[idx:i + knotpointsize(Z)]), indices(func))...)
end
(func::AdjacentKnotPointsFunction)(Z::DiscreteTrajectory) = func(outputtype(func), Z)
function gradient(func::AdjacentKnotPointsFunction, z::AbstractVector)
    ForwardDiff.gradient(func, z)
end
function jacobian!(J::SubArray, func::AdjacentKnotPointsFunction, zₖ::AbstractKnotPoint, zₖ₊₁::AbstractKnotPoint)
    z = [getdata(zₖ); getdata(zₖ₊₁)]
    nₖ = length(zₖ)
    func_stacked_knots(z) = func(view(z, 1:nₖ), view(z, nₖ+1:length(z)))
    ForwardDiff.jacobian!(J, func_stacked_knots, z)
end
function jacobian!(J_vstacked::SubArray, func::AdjacentKnotPointsFunction, knotpoints::AbstractVector{<:AbstractKnotPoint})
    trajectory_indices = indices(func)
    kdim = 2 * length(knotpoints[1])
    Jheight = outputdim(func)

    for (i, idx) in enumerate(trajectory_indices)
        row₀ = (i - 1) * Jheight + 1
        col₀ = kdim * (idx - 1) + 1
        band_view = view(J_vstacked, row₀:row₀ + Jheight - 1, col₀:col₀ + kdim - 1)
        jacobian!(band_view, func, knotpoints[idx], knotpoints[idx+1])
    end
end
function jacobian(funcs::AbstractVector{<:AdjacentKnotPointsFunction}, knotpoints::AbstractVector{<:AbstractKnotPoint})
    kdim = length(knotpoints[1])
    m = sum(length(indices(func)) * outputdim(func) for func in funcs)
    n = kdim * length(knotpoints)
    J_vstacked = zeros(Float64, m, n)

    current_row_idx = 1
    for func ∈ funcs
        band_height = length(indices(func)) * outputdim(func)
        band_view = view(J_vstacked, current_row_idx:band_height, :)
        jacobian!(band_view, func, knotpoints)
        current_row_idx += band_height
    end

    sparse(J_vstacked)
end
function hessian(func::AdjacentKnotPointsFunction, zₖ::AbstractKnotPoint, zₖ₊₁::AbstractKnotPoint)
    z = [zₖ; zₖ₊₁]
    nₖ = length(zₖ)
    func_stacked_knots(z) = func(z[1:nₖ], z[nₖ+1:end])
    ForwardDiff.hessian(func_stacked_knots, z)
end
