abstract type AdjacentKnotPointsFunction end
indices(func::AdjacentKnotPointsFunction) = func.idx
nknots(func::AdjacentKnotPointsFunction) = func.nknots
outputdim(func::AdjacentKnotPointsFunction) = func.outputdim
(func::AdjacentKnotPointsFunction)(Z::DiscreteTrajectory) =
    error("call on knotpoint(s) not implemented")

abstract type ResultAccumulationMethod end
struct Sum <: ResultAccumulationMethod end
struct Stack <: ResultAccumulationMethod end

function unary_func(func::AdjacentKnotPointsFunction, Z::DiscreteTrajectory)
    # Rest assured, no copying happening here
    z -> func(DiscreteTrajectory(time(Z), timesteps(Z), z, knotpointsize(Z), nstates(Z)))
end

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
    f = unary_func(func, Z)
    ForwardDiff.gradient!(▽f, f, z)
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
    m = sum(length(indices(func)) for func ∈ funcs)
    n = length(knotpoints(Z))
    ▽f_vstacked = zeros(T, m, n)
    for (i, func) ∈ enumerate(funcs)
        band_height = length(indices(func))
        row₀ = (i - 1) * band_height + 1
        row₁ = row₀ + band_height - 1
        band_view = @view ▽f_vstacked[row₀:row₁, :]
        gradient_singlef!(band_view, func, Z)
    end
    ▽f_vstacked
end

function gradient(
    ::Val{Sum},
    funcs::AbstractVector{<:AdjacentKnotPointsFunction},
    Z::DiscreteTrajectory{T},
) where {T}
    ▽f_vstacked = gradient(Val(Stack), funcs, Z)
    vec(sum(▽f_vstacked, dims = 1))
end

# Jacobian
function jacobian_impl!(
    J::AbstractMatrix{T},
    func::AdjacentKnotPointsFunction,
    Z::DiscreteTrajectory,
) where {T}
    z = knotpoints(Z)
    f = unary_func(func, Z)
    ForwardDiff.jacobian!(J, f, z)
end

function jacobian_singlef!(
    J_vstacked::AbstractMatrix{T},
    func::AdjacentKnotPointsFunction,
    Z::DiscreteTrajectory,
) where {T}
    Jheight = outputdim(func)
    for (i, col₀) ∈ enumerate(indices(func))
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
function hessian_impl!(
    H::AbstractMatrix{T},
    func::AdjacentKnotPointsFunction,
    Z::DiscreteTrajectory,
) where {T}
    z = knotpoints(Z)
    f = unary_func(func, Z)
    ForwardDiff.hessian!(H, f, z)
end

function hessian_singlef!(
    H::AbstractMatrix{T},
    func::AdjacentKnotPointsFunction,
    Z::DiscreteTrajectory,
) where {T}
    for (i, col₀) ∈ enumerate(indices(func))
        col₁ = col₀ + nknots(func) - 1
        colrange = knotpointindices(Z, col₀:col₁)
        z = @view Z[col₀:col₁]
        # Hessian is square symmetric, colrange = rowrange
        block = @view H[colrange, colrange]
        hessian_impl!(block, func, z)
    end
end

function hessian(
    funcs::AbstractVector{<:AdjacentKnotPointsFunction},
    Z::DiscreteTrajectory{T},
) where {T}
    n = length(knotpoints(Z))
    H = zeros(T, n, n)
    for (i, func) ∈ enumerate(funcs)
        # TODO: we're just overwriting the same Hessian matrix for every function. Fix this!
        hessian_singlef!(H, func, Z)
    end
    H
end

function vector_hessian_impl!(
    H::AbstractMatrix{T},
    J::AbstractMatrix{T},
    func::AdjacentKnotPointsFunction,
    Z::DiscreteTrajectory,
) where {T}
    z = knotpoints(Z)
    function inner_jacobian!(z::AbstractVector{T}) where {T}
        z = knotpoints(Z)
        f = unary_func(func, Z)
        ForwardDiff.jacobian!(J, f, z)
        vec(J)  # does not copy
    end
    ForwardDiff.jacobian!(H, inner_jacobian!, z)
end

function vector_hessian_singlef!(
    H::AbstractMatrix{T},
    J::AbstractMatrix{T},
    func::AdjacentKnotPointsFunction,
    Z::DiscreteTrajectory,
) where {T}
    for (i, col₀) ∈ enumerate(indices(func))
        col₁ = col₀ + nknots(func) - 1
        colrange = knotpointindices(Z, col₀:col₁)
        z = @view Z[col₀:col₁]

        band_height = outputdim(func) * length(colrange)
        row₀ = (i - 1) * band_height + 1
        row₁ = row₀ + band_height - 1

        block = @view H[row₀:row₁, colrange]
        vector_hessian_impl!(block, J, func, z)
    end
end

function vector_hessian(
    funcs::AbstractVector{<:AdjacentKnotPointsFunction},
    Z::DiscreteTrajectory{T},
    λ::AbstractVector{T},
) where {T}
    n = length(knotpoints(Z))
    m = sum(length(indices(func)) * outputdim(func) * n for func in funcs)
    H = zeros(T, m, n)

    for (i, func) ∈ enumerate(funcs)
        Jheight = outputdim(func)
        Jwidth = nknots(func) * knotpointsize(Z)
        J = zeros(T, Jheight, Jwidth)
        vector_hessian_singlef!(H, J, func, Z)
    end

    ∑H = zeros(T, n, n)
    H3 = reshape(H', n, n, :)

    @assert length(λ) == size(H3, 3) "length of dual variable vector $length(λ)) ≠ hessian stack depth $(size(H3, 3))"

    @inbounds for k = 1:length(λ)
        sliceₖ = @view H3[:, :, k]
        sliceₖ .*= λ[k]
        ∑H .+= sliceₖ
    end

    Symmetric(∑H)
end
