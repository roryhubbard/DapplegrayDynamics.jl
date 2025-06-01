abstract type AdjacentKnotPointsConstraint <: AdjacentKnotPointsFunction end

struct ConicConstraint{T} <: AdjacentKnotPointsConstraint
    A::SparseMatrixCSC{T, Int}
    b::AbstractVector{T}
    cone::Clarabel.SupportedCone
    idx::UnitRange{Int}
    nknots::Int
    outputdim::Int
end
cone(::ConicConstraint) = error("cone not defined")
function (con::ConicConstraint{T})(z::DiscreteTrajectory{T}) where {T}
    x = knotpoints(z)
    @assert length(x) == knotpointsize(z) * nknots(con) "ConicConstraint expected knotpoint vector length $(knotpointsize(z) * nknots(con)) but received $(length(x))"
    A * x - b
end

function control_bound_constraint(
    upperbound::Union{AbstractVector{T}, Nothing},
    lowerbound::Union{AbstractVector{T}, Nothing},
    knotpointsize::Int,
    idx::UnitRange{Int},
)::ConicConstraint{T} where {T}
    if isnothing(upperbound) && isnothing(lowerbound)
        throw(ArgumentError("At least one of upperbound or lowerbound must be provided."))
    elseif length(upperbound) != length(lowerbound)
        throw(ArgumentError("Bounds must have equal length (got upperbound length = $(length(upperbound)), lowerbound length = $(length(lowerbound)))"))
    end

    if isnothing(con.upperbound)
        outputdim = length(con.lowerbound)
        A = spzeros(outputdim, knotpointsize)
        col₀ = knotpointsize - outputdim + 1
        A[:, col₀:end] .= -I(outputdim)
        b = -lb
    elseif isnothing(con.lowerbound)
        outputdim = length(con.upperbound)
        A = spzeros(outputdim, knotpointsize)
        col₀ = knotpointsize - outputdim + 1
        A[:, col₀:end] .= I(outputdim)
        b = ub
    end

    ConicConstraint(A, b, Clarabel.NonnegativeCone(outputdim), idx, 1, outputdim)
end

function state_equality_constraint(
    xd::AbstractVector{T}
    knotpointsize::Int,
    idx::UnitRange{Int},
)::ConicConstraint{T} where {T}
    outputdim = length(xd)
    A = spzeros(outputdim, knotpointsize)
    A[:, 1:outputdim] .= I(outputdim)
    b = zeros(outputdim)
    ConicConstraint(A, b, Clarabel.ZeroConeT(outputdim), idx, 1, outputdim)
end

# TODO: dynamics! assumes fully actuated systems, figure out a method for
# dealing with control vectors that are smaller in rank than the vector of
# "velocities" in mechanism state
function hermite_simpson_separated(mechanism::Mechanism, Δt::Real, xₖ::AbstractVector, uₖ::AbstractVector, xₖ₊₁::AbstractVector, uₖ₊₁::AbstractVector, xₘ::AbstractVector, uₘ::AbstractVector)
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
function hermite_simpson_compressed(mechanism::Mechanism, Δt::Real, xₖ::AbstractVector, uₖ::AbstractVector, xₖ₊₁::AbstractVector, uₖ₊₁::AbstractVector)
    mechanismstate = MechanismState(mechanism)
    dynamicsresult = DynamicsResult(mechanism)

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
function (con::HermiteSimpsonConstraint)(z::DiscreteTrajectory{T}) where {T}
    xₖ = state(zₖ)
    uₖ = control(zₖ)
    xₖ₊₁ = state(zₖ₊₁)
    uₖ₊₁ = control(zₖ₊₁)

    k = knotpoints(z)
    nx = nstates(z)
    x = @view k[1:nx]
    u = @view k[nx+1:end]
    Δt = timesteps(zₖ)
    @assert length(x) == knotpointsize(z) * nknots(con) "ConicConstraint expected knotpoint vector length $(knotpointsize(z) * nknots(con)) but received $(length(x))"
    hermite_simpson_compressed(con.mechanism, Δt, xₖ, uₖ, xₖ₊₁, uₖ₊₁)
end
