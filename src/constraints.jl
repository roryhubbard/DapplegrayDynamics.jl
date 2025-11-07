struct ConicConstraint{T} <: AdjacentKnotPointsFunction
    A::SparseMatrixCSC{T,Int}
    b::AbstractVector{T}
    cone::Clarabel.SupportedCone
    idx::UnitRange{Int}
    nknots::Int
    outputdim::Int
    function ConicConstraint(
        A::SparseMatrixCSC{T,Int},
        b::AbstractVector{T},
        conetype::Type{<:Clarabel.SupportedCone},
        idx::UnitRange{Int},
        nknots::Int,
    ) where {T}
        coneN = length(b)
        new{T}(A, b, conetype(coneN), idx, nknots, coneN)
    end
end
cone(::ConicConstraint) = error("cone not defined")
function (con::ConicConstraint{T})(z::DiscreteTrajectory) where {T}
    x = knotpoints(z)
    @assert length(x) == knotpointsize(z) * nknots(con) "ConicConstraint expected knotpoint vector length $(knotpointsize(z) * nknots(con)) but received $(length(x))"
    con.A * x - con.b
end

function conic_lowerbound_Ab(lowerbound::AbstractVector{T}, knotpointsize::Int) where {T}
    outputdim = length(lowerbound)
    A = spzeros(T, outputdim, knotpointsize)
    col₀ = knotpointsize - outputdim + 1
    A[:, col₀:end] .= -I(outputdim)
    b = -lowerbound
    A, b
end

function conic_upperbound_Ab(upperbound::AbstractVector{T}, knotpointsize::Int) where {T}
    outputdim = length(upperbound)
    A = spzeros(T, outputdim, knotpointsize)
    col₀ = knotpointsize - outputdim + 1
    A[:, col₀:end] .= I(outputdim)
    b = upperbound
    A, b
end

function control_bound_constraint(
    knotpointsize::Int,
    idx::UnitRange{Int},
    upperbound::Union{AbstractVector{T},Nothing} = nothing,
    lowerbound::Union{AbstractVector{T},Nothing} = nothing,
)::ConicConstraint{T} where {T}
    if isnothing(upperbound) && isnothing(lowerbound)
        throw(ArgumentError("At least one of upperbound or lowerbound must be provided."))
    end

    if isnothing(upperbound)
        A, b = conic_lowerbound_Ab(lowerbound, knotpointsize)
    elseif isnothing(lowerbound)
        A, b = conic_upperbound_Ab(lowerbound, knotpointsize)
    else
        if length(upperbound) != length(lowerbound)
            throw(
                ArgumentError(
                    "Bounds must have equal length (got upperbound length = $(length(upperbound)), lowerbound length = $(length(lowerbound)))",
                ),
            )
        end
        Aₗ, bₗ = conic_lowerbound_Ab(lowerbound, knotpointsize)
        Aᵤ, bᵤ = conic_upperbound_Ab(lowerbound, knotpointsize)
        A = [Aₗ; Aᵤ]
        b = [bₗ; bᵤ]
    end

    ConicConstraint(A, b, Clarabel.NonnegativeConeT, idx, 1)
end

function control_bound_constraint(
    knotpointsize::Int,
    idx::Int,
    upperbound::Union{AbstractVector{T},Nothing} = nothing,
    lowerbound::Union{AbstractVector{T},Nothing} = nothing,
)::ConicConstraint{T} where {T}
    control_bound_constraint(knotpointsize, idx:idx, upperbound, lowerbound)
end

function state_equality_constraint(
    xd::AbstractVector{T},
    knotpointsize::Int,
    idx::UnitRange{Int},
)::ConicConstraint{T} where {T}
    outputdim = length(xd)
    A = spzeros(outputdim, knotpointsize)
    A[:, 1:outputdim] .= I(outputdim)
    b = zeros(outputdim)
    ConicConstraint(A, b, Clarabel.ZeroConeT, idx, 1)
end

function state_equality_constraint(
    xd::AbstractVector{T},
    knotpointsize::Int,
    idx::Int,
)::ConicConstraint{T} where {T}
    state_equality_constraint(xd, knotpointsize, idx:idx)
end

function separated_hermite_simpson(
    mechanism::Mechanism{Tm},
    Δt::Real,
    xₖ::AbstractVector{Tk},
    uₖ::AbstractVector{Tk},
    xₘ::AbstractVector{Tk},
    uₘ::AbstractVector{Tk},
    xₖ₊₁::AbstractVector{Tk},
    uₖ₊₁::AbstractVector{Tk},
    active_joint_indices::Vector{Int},
) where {Tm,Tk}
    mechanismstate = MechanismState{Tk}(mechanism)
    dynamicsresult = DynamicsResult{Tk}(mechanism)

    # TODO: remove memory allocation
    τₖ = zeros(Tk, num_positions(mechanism))
    τₖ[active_joint_indices] = uₖ
    ẋₖ = similar(xₖ)
    dynamics!(ẋₖ, dynamicsresult, mechanismstate, xₖ, τₖ)

    # TODO: remove memory allocation
    τₖ₊₁ = zeros(Tk, num_positions(mechanism))
    τₖ₊₁[active_joint_indices] = uₖ₊₁
    ẋₖ₊₁ = similar(xₖ₊₁)
    dynamics!(ẋₖ₊₁, dynamicsresult, mechanismstate, xₖ₊₁, τₖ₊₁)

    # TODO: remove memory allocation
    τₘ = zeros(Tk, num_positions(mechanism))
    τₘ[active_joint_indices] = uₘ
    ẋₘ = similar(xₖ)
    dynamics!(ẋₘ, dynamicsresult, mechanismstate, xₘ, τₘ)

    c₁ = xₖ₊₁ - xₖ - Δt / 6 * (ẋₖ + 4 * ẋₘ + ẋₖ₊₁)
    c₂ = xₘ - 1 / 2 * (xₖ + xₖ₊₁) - Δt / 8 * (ẋₖ - ẋₖ₊₁)
    # TODO: remove memory allocation
    vcat(c₁, c₂)
end

function compressed_hermite_simpson(
    mechanism::Mechanism{Tm},
    Δt::Real,
    xₖ::AbstractVector{Tk},
    uₖ::AbstractVector{Tk},
    xₖ₊₁::AbstractVector{Tk},
    uₖ₊₁::AbstractVector{Tk},
    active_joint_indices::Vector{Int},
) where {Tm,Tk}
    mechanismstate = MechanismState{Tk}(mechanism)
    dynamicsresult = DynamicsResult{Tk}(mechanism)

    # TODO: remove memory allocation
    τₖ = zeros(Tk, num_positions(mechanism))
    τₖ[active_joint_indices] = uₖ
    ẋₖ = similar(xₖ)
    dynamics!(ẋₖ, dynamicsresult, mechanismstate, xₖ, τₖ)

    # TODO: remove memory allocation
    τₖ₊₁ = zeros(Tk, num_positions(mechanism))
    τₖ₊₁[active_joint_indices] = uₖ₊₁
    ẋₖ₊₁ = similar(xₖ₊₁)
    dynamics!(ẋₖ₊₁, dynamicsresult, mechanismstate, xₖ₊₁, τₖ₊₁)

    # We could add the collocation point as an extra decision variable and
    # constraint. This would be "separated form". Here we are implementing
    # "compressed form" where we calculate `fcol` and jam it into the constraint
    # for the integral of the system dynamics.
    xₘ = 1 / 2 * (xₖ + xₖ₊₁) + Δt / 8 * (ẋₖ - ẋₖ₊₁)
    uₘ = 1 / 2 * (uₖ + uₖ₊₁)
    τₘ = zeros(Tk, num_positions(mechanism))
    τₘ[active_joint_indices] = uₘ
    ẋₘ = similar(xₖ)
    dynamics!(ẋₘ, dynamicsresult, mechanismstate, xₘ, τₘ)

    # equality constraint: xₖ₊₁ - xₖ = (Δt / 6) * (fₖ + 4fcol + fₖ₊₁)
    xₖ₊₁ - xₖ - Δt / 6 * (ẋₖ + 4 * ẋₘ + ẋₖ₊₁)
end

abstract type HermiteSimpsonConstraint <: AdjacentKnotPointsFunction end

struct CompressedHermiteSimpsonConstraint{T} <: HermiteSimpsonConstraint
    mechanism::Mechanism{T}
    idx::UnitRange{Int}
    nknots::Int
    outputdim::Int
    active_joint_indices::Vector{Int}
    function CompressedHermiteSimpsonConstraint(
        mechanism::Mechanism{T},
        idx::UnitRange{Int},
        active_joint_indices::Vector{Int},
    ) where {T}
        outputdim = num_positions(mechanism) + num_velocities(mechanism)
        new{T}(mechanism, idx, 2, outputdim, active_joint_indices)
    end
end
function (con::CompressedHermiteSimpsonConstraint)(z::DiscreteTrajectory)
    Δt = timesteps(z)
    @assert length(Δt) == 2 "CompressedHermiteSimpsonConstraint expects two knotpoints and therefore 2 timesteps, but received $(length(Δt))"
    @assert length(knotpoints(z)) == knotpointsize(z) * nknots(con) "CompressedHermiteSimpsonConstraint expects knotpoint vector length $(knotpointsize(z) * nknots(con)) but received $(length(knotpoints(z)))"
    xₖ, xₖ₊₁ = state(z, Val(2))
    uₖ, uₖ₊₁ = control(z, Val(2))
    compressed_hermite_simpson(
        con.mechanism,
        first(Δt),
        xₖ,
        uₖ,
        xₖ₊₁,
        uₖ₊₁,
        con.active_joint_indices,
    )
end

struct SeparatedHermiteSimpsonConstraint{T} <: HermiteSimpsonConstraint
    mechanism::Mechanism{T}
    idx::UnitRange{Int}
    nknots::Int
    outputdim::Int
    active_joint_indices::Vector{Int}
    function SeparatedHermiteSimpsonConstraint(
        mechanism::Mechanism{T},
        idx::Int,
        active_joint_indices::Vector{Int},
    ) where {T}
        outputdim = num_positions(mechanism) + num_velocities(mechanism)
        # TODO: add arbitrary stride lengths so that we can apply separated
        # hermite simpson constraints with a knotpoint index range like 1:2:N-1
        # instead of having to apply 1 at a time
        new{T}(mechanism, idx:idx, 3, 2*outputdim, active_joint_indices)
    end
end
function (con::SeparatedHermiteSimpsonConstraint)(z::DiscreteTrajectory)
    Δt = timesteps(z)
    @assert length(Δt) == 3 "SeparatedHermiteSimpsonConstraint expects three knotpoints and therefore 3 timesteps, but received $(length(Δt))"
    @assert length(knotpoints(z)) == knotpointsize(z) * nknots(con) "SeparatedHermiteSimpsonConstraint expects knotpoint vector length $(knotpointsize(z) * nknots(con)) but received $(length(knotpoints(z)))"
    xₖ, xₘ, xₖ₊₁ = state(z, Val(3))
    uₖ, uₘ, uₖ₊₁ = control(z, Val(3))
    h = Δt[1] + Δt[2]
    separated_hermite_simpson(
        con.mechanism,
        h,
        xₖ,
        uₖ,
        xₘ,
        uₘ,
        xₖ₊₁,
        uₖ₊₁,
        con.active_joint_indices,
    )
end
