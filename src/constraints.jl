abstract type AdjacentKnotPointsConstraint <: AdjacentKnotPointsFunction end
abstract type AbstractConicConstraint <: AdjacentKnotPointsConstraint end

struct ConicConstraint{T} <: AdjacentKnotPointsConstraint
    A::AbstractMatrix{T}
    b::AbstractVector{T}
    cone::Clarabel.SupportedCone
    functioninputs::FunctionInputs
    idx::UnitRange{Int}
end
cone(::ConicConstraint) = error("cone not defined")
function (con::ConicConstraint)(z::AbstractVector)
    A * z - b
end

function control_bound_constraint(
    upperbound::Union{AbstractVector{T}, Nothing},
    lowerbound::Union{AbstractVector{T}, Nothing},
    idx::UnitRange{Int}) where {T}
)
    if isnothing(upperbound) && isnothing(lowerbound)
        throw(ArgumentError("At least one of upperbound or lowerbound must be provided."))
    elseif length(upperbound) != length(lowerbound)
        throw(ArgumentError("Bounds must have equal length (got upperbound length = $(length(upperbound)), lowerbound length = $(length(lowerbound)))"))
    end
end

struct ControlBound{T} <: ConicConstraint
    upperbound::Union{AbstractVector{T}, Nothing}
    lowerbound::Union{AbstractVector{T}, Nothing}
    idx::UnitRange{Int}

    function ControlBound(upperbound::Union{AbstractVector{T}, Nothing},
                          lowerbound::Union{AbstractVector{T}, Nothing},
                          idx::UnitRange{Int}) where {T}
        if isnothing(upperbound) && isnothing(lowerbound)
            throw(ArgumentError("At least one of upperbound or lowerbound must be provided."))
        elseif length(upperbound) != length(lowerbound)
            throw(ArgumentError("Bounds must have equal length (got upperbound length = $(length(upperbound)), lowerbound length = $(length(lowerbound)))"))
        end
        new{T}(upperbound, lowerbound, idx)
    end
end
outputtype(::ControlBound) = VectorOutput()
cone(con::ConicConstraint) = Clarabel.NonnegativeCone(outputdim(con))
function outputdim(con::ControlBound)
    if isnothing(con.upperbound)
        return length(con.lowerbound)
    elseif isnothing(con.lowerbound)
        return length(con.upperbound)
    end
    return length(con.upperbound) + length(con.lowerbound)
end
function (con::ControlBound)(z::AbstractVector)
    ub = con.upperbound
    lb = con.lowerbound
    if isnothing(ub)
        return lb - u
    elseif isnothing(lb)
        return u - ub
    else
        return [lb - u; u - ub]
    end
end

struct StateEqualityConstraint <: StateFunction
    xd::AbstractVector
    idx::UnitRange{Int}
end
outputtype(::StateEqualityConstraint) = VectorOutput()
outputdim(con::StateEqualityConstraint) = length(con.xd)
function (con::StateEqualityConstraint)(x::AbstractVector, _)
    x - con.xd
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
end
outputtype(::HermiteSimpsonConstraint) = VectorOutput()
outputdim(con::HermiteSimpsonConstraint) = num_positions(con.mechanism) + num_velocities(con.mechanism)
# TODO: remove this function
function (con::HermiteSimpsonConstraint)(zₖ::SubArray, zₖ₊₁::SubArray)
    num_controls = 1
    xₖ = zₖ[1:outputdim(con)]
    uₖ = [zₖ[outputdim(con)+1]]
    xₖ₊₁ = zₖ₊₁[1:outputdim(con)]
    uₖ₊₁ = [zₖ₊₁[outputdim(con)+1]]
    Δt = 1.0
    hermite_simpson_compressed(con.mechanism, Δt, xₖ, uₖ, xₖ₊₁, uₖ₊₁)
end
function (con::HermiteSimpsonConstraint)(zₖ::AbstractKnotPoint, zₖ₊₁::AbstractKnotPoint)
    xₖ = state(zₖ)
    uₖ = control(zₖ)
    xₖ₊₁ = state(zₖ₊₁)
    uₖ₊₁ = control(zₖ₊₁)
    Δt = timestep(zₖ)
    hermite_simpson_compressed(con.mechanism, Δt, xₖ, uₖ, xₖ₊₁, uₖ₊₁)
end
