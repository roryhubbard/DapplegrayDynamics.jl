abstract type AdjacentKnotPointsConstraint <: AdjacentKnotPointsFunction end

struct ConicConstraint{T} <: AdjacentKnotPointsConstraint
    A::SparseMatrixCSC{T, Int}
    b::AbstractVector{T}
    cone::Clarabel.SupportedCone
    idx::UnitRange{Int}
    nknots::Int
    outputdim::Int
    function ConicConstraint(A::SparseMatrixCSC{T, Int}, b::AbstractVector{T},
                             conetype::Type{<:Clarabel.SupportedCone}, idx::UnitRange{Int},
                             nknots::Int) where {T}
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
    upperbound::Union{AbstractVector{T}, Nothing},
    lowerbound::Union{AbstractVector{T}, Nothing},
    knotpointsize::Int,
    idx::UnitRange{Int},
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
            throw(ArgumentError("Bounds must have equal length (got upperbound length = $(length(upperbound)), lowerbound length = $(length(lowerbound)))"))
        end
        Aₗ, bₗ = conic_lowerbound_Ab(lowerbound, knotpointsize)
        Aᵤ, bᵤ = conic_upperbound_Ab(lowerbound, knotpointsize)
        A = [Aₗ; Aᵤ]
        b = [bₗ; bᵤ]
    end

    ConicConstraint(A, b, Clarabel.NonnegativeConeT, idx, 1)
end

function control_bound_constraint(
    upperbound::Union{AbstractVector{T}, Nothing},
    lowerbound::Union{AbstractVector{T}, Nothing},
    knotpointsize::Int,
    idx::Int,
)::ConicConstraint{T} where {T}
    control_bound_constraint(upperbound, lowerbound, knotpointsize, idx:idx)
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

# TODO: dynamics! assumes fully actuated systems, figure out a method for
# dealing with control vectors that are smaller in rank than the vector of
# "velocities" in mechanism state
function hermite_simpson_separated(mechanism::Mechanism{T}, Δt::Real, xₖ::AbstractVector{T}, uₖ::AbstractVector{T}, xₖ₊₁::AbstractVector{T}, uₖ₊₁::AbstractVector{T}, xₘ::AbstractVector{T}, uₘ::AbstractVector{T}) where {T}
    mechanismstate = MechanismState(mechanism)
    dynamicsresult = DynamicsResult(mechanism)

    # TODO: remove this hardcode nullification of one of the actuators, see TODO
    # above
    τₖ = vcat(0., uₖ)
    ẋₖ = similar(xₖ)
    dynamics!(ẋₖ, dynamicsresult, mechanismstate, xₖ, τₖ)

    τₖ₊₁ = vcat(0., uₖ₊₁)
    ẋₖ₊₁ = similar(xₖ₊₁)
    dynamics!(ẋₖ₊₁, dynamicsresult, mechanismstate, xₖ₊₁, τₖ₊₁)

    τₘ = vcat(0., uₘ)
    ẋₘ = similar(xₖ)
    dynamics!(ẋₘ, dynamicsresult, mechanismstate, xₘ, τₘ)

    c₁ = xₖ₊₁ - xₖ - Δt / 6 * (ẋₖ + 4 * ẋₘ + ẋₖ₊₁)
    c₂ = ẋₘ - 1 / 2 * (xₖ + xₖ₊₁) - Δt / 8 * (ẋₖ - ẋₖ₊₁)
    c₁, c₂
end
function hermite_simpson_compressed(mechanism::Mechanism{Tm}, Δt::Real, xₖ::AbstractVector{Tk}, uₖ::AbstractVector{Tk}, xₖ₊₁::AbstractVector{Tk}, uₖ₊₁::AbstractVector{Tk}) where {Tm, Tk}
    mechanismstate = MechanismState{Tk}(mechanism)
    dynamicsresult = DynamicsResult{Tk}(mechanism)

    τₖ = vcat(0., uₖ)
    ẋₖ = similar(xₖ)
    dynamics!(ẋₖ, dynamicsresult, mechanismstate, xₖ, τₖ)

    τₖ₊₁ = vcat(0., uₖ₊₁)
    ẋₖ₊₁ = similar(xₖ₊₁)
    dynamics!(ẋₖ₊₁, dynamicsresult, mechanismstate, xₖ₊₁, τₖ₊₁)

    # We could add the collocation point as an extra decision varaible and
    # constraint. This would be "separated form". Here we are implementing
    # "compressed form" where we calculate `fcol` and jam it into the constraint
    # for the integral of the system dynamics.
    xₘ = 1 / 2 * (xₖ + xₖ₊₁) + Δt / 8 * (ẋₖ - ẋₖ₊₁)
    uₘ = 1 / 2 * (uₖ + uₖ₊₁)
    τₘ = vcat(0., uₘ)
    ẋₘ = similar(xₖ)
    dynamics!(ẋₘ, dynamicsresult, mechanismstate, xₘ, τₘ)

    # equality constraint: xₖ₊₁ - xₖ = (Δt / 6) * (fₖ + 4fcol + fₖ₊₁)
    xₖ₊₁ - xₖ - Δt / 6 * (ẋₖ + 4 * ẋₘ + ẋₖ₊₁)
end

struct HermiteSimpsonConstraint{T} <: AdjacentKnotPointsFunction
    mechanism::Mechanism{T}
    idx::UnitRange{Int}
    nknots::Int
    outputdim::Int
    function HermiteSimpsonConstraint(mechanism::Mechanism{T}, idx::UnitRange{Int}) where {T}
        outputdim = num_positions(mechanism) + num_velocities(mechanism)
        new{T}(mechanism, idx, 2, outputdim)
    end
end
function (con::HermiteSimpsonConstraint)(z::DiscreteTrajectory)
    Δt = timesteps(z)
    @assert length(Δt) == 2 "HermiteSimpsonConstraint expects two knotpoints and therefore 2 timesteps, but received $(length(Δt))"
    @assert length(knotpoints(z)) == knotpointsize(z) * nknots(con) "HermiteSimpsonConstraint expects knotpoint vector length $(knotpointsize(z) * nknots(con)) but received $(length(knotpoints(z)))"
    xₖ, xₖ₊₁ = state(z, Val(2))
    uₖ, uₖ₊₁ = control(z, Val(2))
    hermite_simpson_compressed(con.mechanism, first(Δt), xₖ, uₖ, xₖ₊₁, uₖ₊₁)
end
