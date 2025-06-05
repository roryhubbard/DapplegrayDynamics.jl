abstract type AdjacentKnotPointsFunction end
indices(func::AdjacentKnotPointsFunction) = func.idx
nknots(func::AdjacentKnotPointsFunction) = func.nknots
outputdim(func::AdjacentKnotPointsFunction) = func.outputdim
(func::AdjacentKnotPointsFunction)(Z::DiscreteTrajectory) =
    error("call on knotpoint(s) not implemented")

abstract type ResultAccumulationMethod end
struct Sum <: ResultAccumulationMethod end
struct Stack <: ResultAccumulationMethod end

# Evaluation
function (func::AdjacentKnotPointsFunction)(::Val{Sum}, Z::DiscreteTrajectory{T}) where {T}
    result = zero(T)
    for i₀ ∈ indices(func)
        i₁ = i₀ + (nknots(func) - 1)
        z = @view Z[i₀:i₁]
        result += func(z)
    end
    result
end

function (func::AdjacentKnotPointsFunction)(::Val{Stack}, Z::DiscreteTrajectory)
    # TOOD: preallocate
    outputs = map(indices(func)) do i₀
        i₁ = i₀ + (nknots(func) - 1)
        z = @view Z[i₀:i₁]
        func(z)
    end
    vcat(outputs...)
end

# Gradient
function gradient_impl!(
    ▽f::AbstractVector{T},
    func::AdjacentKnotPointsFunction,
    Z::DiscreteTrajectory,
) where {T}
    z = knotpoints(Z)
    # Rest assured, no copying happening here
    fwrapped(z) =
        func(DiscreteTrajectory(time(Z), timesteps(Z), z, knotpointsize(Z), nstates(Z)))
    ForwardDiff.gradient!(▽f, fwrapped, z)
end

function gradient_singlef!(
    ▽f_vstacked::AbstractMatrix{T},
    func::AdjacentKnotPointsFunction,
    Z::DiscreteTrajectory,
) where {T}
    for (i, col₀) in enumerate(indices(func))
        col₁ = col₀ + nknots(func) - 1
        colrange = knotpointindices(Z, col₀:col₁)
        z = @view Z[col₀:col₁]
        rowview = @view ▽f_vstacked[i, colrange]
        gradient_impl!(rowview, func, z)
    end
end

function gradient(
    ::Val{Stack},
    funcs::AbstractVector{<:AdjacentKnotPointsFunction},
    Z::DiscreteTrajectory{T},
) where {T}
    m = sum(length(indices(func)) for func in funcs)
    n = length(knotpoints(Z))
    ▽f_vstacked = zeros(T, m, n)

    current_row_idx = 1
    for func ∈ funcs
        band_height = length(indices(func))
        band_view = @view ▽f_vstacked[current_row_idx:band_height, :]
        gradient_singlef!(band_view, func, Z)
        current_row_idx += band_height + 1
    end

    ▽f_vstacked
end

function gradient(
    ::Val{Sum},
    funcs::AbstractVector{<:AdjacentKnotPointsFunction},
    Z::DiscreteTrajectory{T},
) where {T}
    ▽f_vstacked = gradient(Val(Stack), funcs, Z)
    vec(sum(▽f_vstacked, dims=1))
end

# Jacobian
function jacobian_impl!(
    J::AbstractMatrix{T},
    func::AdjacentKnotPointsFunction,
    Z::DiscreteTrajectory,
) where {T}
    z = knotpoints(Z)
    fwrapped(z) =
        func(DiscreteTrajectory(time(Z), timesteps(Z), z, knotpointsize(Z), nstates(Z)))
    ForwardDiff.jacobian!(J, fwrapped, z)
end

function jacobian_singlef!(
    J_vstacked::AbstractMatrix{T},
    func::AdjacentKnotPointsFunction,
    Z::DiscreteTrajectory,
) where {T}
    Jheight = outputdim(func)

    for (i, col₀) in enumerate(indices(func))
        row₀ = (i - 1) * Jheight + 1
        row₁ = row₀ + Jheight - 1
        col₁ = col₀ + nknots(func) - 1
        colrange = knotpointindices(Z, col₀:col₁)
        z = @view Z[col₀:col₁]
        band = @view J_vstacked[row₀:row₁, colrange]
        jacobian_impl!(band, func, z)
    end
end

function jacobian(
    funcs::AbstractVector{<:AdjacentKnotPointsFunction},
    Z::DiscreteTrajectory{T},
) where {T}
    m = sum(length(indices(func)) * outputdim(func) for func in funcs)
    n = length(knotpoints(Z))
    J_vstacked = zeros(T, m, n)

    for (i, func) ∈ enumerate(funcs)
        band_height = length(indices(func)) * outputdim(func)
        row₀ = (i - 1) * band_height + 1
        row₁ = row₀ + band_height - 1
        band = @view J_vstacked[row₀:row₁, :]
        jacobian_singlef!(band, func, Z)
    end

    J_vstacked
end

# Hessian
function hessian(
    H::AbstractMatrix{T},
    func::AdjacentKnotPointsFunction,
    Z::DiscreteTrajectory,
) where {T}
    z = knotpoints(Z)
    fwrapped(z) =
        func(DiscreteTrajectory(time(Z), timesteps(Z), z, knotpointsize(Z), nstates(Z)))
    ForwardDiff.hessian!(H, fwrapped, z)
end
