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
    for col₀ ∈ indices(func)
        col₁ = col₀ + nknots(func) - 1
        colrange = knotpointindices(Z, col₀:col₁)
        z = @view Z[col₀:col₁]
        # Hessian is square symmetric, colrange = rowrange
        block = @view H[colrange, colrange]
        hessian_impl!(block, func, z)
    end
end

# Collect the H indices a single objective will touch
@inline function _touched_indices(Z, f)::Vector{Int}
    acc = Int[]
    for col₀ in indices(f)
        col₁ = col₀ + nknots(f) - 1
        append!(acc, knotpointindices(Z, col₀:col₁))
    end
    unique!(acc)                 # allow internal repeats within f
    return acc
end

function hessian(
    funcs::AbstractVector{<:AdjacentKnotPointsFunction},
    Z::DiscreteTrajectory{T},
) where {T}
    n = length(knotpoints(Z))
    H = zeros(T, n, n)

    # TODO: check for this outside of the solver routine. Better if it was done
    # in the construction of the optimization problem
    # ---- detect overlapping supports exactly as the blocks are formed ----
    used = Dict{Int,Int}()  # H-index -> first func id that claimed it
    for (i, f) in enumerate(funcs)
        for k in _touched_indices(Z, f)
            if haskey(used, k)
                error(
                    "Hessian assembly conflict: objective $(i) overlaps with objective $(used[k]) at H-index $k. " *
                    "This violates the assumption that each objective acts on disjoint knotpoints and would overwrite Hessian blocks",
                )
            end
            used[k] = i
        end
    end

    foreach(f -> hessian_singlef!(H, f, Z), funcs)
    Symmetric(H)
end

function vector_hessian_impl!(
    H::AbstractMatrix{T},
    func::AdjacentKnotPointsFunction,
    Z::DiscreteTrajectory,
) where {T}
    z = knotpoints(Z)
    f = unary_func(func, Z)
    # TODO: can't use ForwardDiff.jacobian! for innner jacobian
    # https://github.com/JuliaDiff/ForwardDiff.jl/issues/393
    ForwardDiff.jacobian!(H, z -> ForwardDiff.jacobian(f, z), z)
end

function vector_hessian_singlef!(
    H::AbstractMatrix{T},
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
        vector_hessian_impl!(block, func, z)
    end
end

function vector_hessian(
    funcs::AbstractVector{<:AdjacentKnotPointsFunction},
    Z::DiscreteTrajectory,
    λ::AbstractVector{T},
) where {T}
    n = length(knotpoints(Z))
    m = sum(length(indices(func)) * outputdim(func) * n for func in funcs)
    H = zeros(T, m, n)

    foreach(f -> vector_hessian_singlef!(H, f, Z), funcs)

    ∑H = zeros(T, n, n)
    H3 = reshape(H', n, n, :)

    @assert length(λ) == size(H3, 3) "length of dual variable vector $length(λ)) ≠ hessian stack depth $(size(H3, 3))"

    @inbounds @views for k = 1:length(λ)
        ∑H .+= λ[k] .* H3[:, :, k]
    end
    # numeric symmetrization before wrapping to handle autodiff noise, maybe not
    # necessary?
    #    ∑H .= (∑H .+ ∑H') .* T(0.5)

    Symmetric(∑H)
end
