@kwdef struct OuterSettings{T<:AbstractFloat}
    max_iter::UInt32 = 10
    time_limit::Float64 = Inf
    verbose::Bool = true
    μ_init::T = 1.0               # initial augmented Lagrangian penalty
    merit_function::Symbol = :fletcher  # :fletcher or :standard               # initial augmented Lagrangian penalty
end

OuterSettings(args...) = OuterSettings{Float64}(args...)

struct SQPSolver{T}
    mechanism::Mechanism{T}
    f::AbstractVector{<:AdjacentKnotPointsFunction}
    g::AbstractVector{<:AdjacentKnotPointsFunction}
    h::AbstractVector{<:AdjacentKnotPointsFunction}
    x::DiscreteTrajectory{T,T}
    λ::AbstractVector{T}
    v::AbstractVector{T}
    inner_settings::Clarabel.Settings{T}
    outer_settings::OuterSettings{T}
    guts::Dict{Symbol,Any}

    function SQPSolver(
        mechanism::Mechanism{T},
        f::AbstractVector{<:AdjacentKnotPointsFunction},
        g::AbstractVector{<:AdjacentKnotPointsFunction},
        h::AbstractVector{<:AdjacentKnotPointsFunction},
        x::DiscreteTrajectory{T,T},
        λ::Union{AbstractVector{T},Nothing} = nothing,
        v::Union{AbstractVector{T},Nothing} = nothing,
        inner_settings::Union{Clarabel.Settings{T},Nothing} = nothing,
        outer_settings::Union{OuterSettings{T},Nothing} = nothing,
    ) where {T}
        if isnothing(λ)
            λ = zeros(T, num_lagrange_multipliers(g))
        end
        if isnothing(v)
            v = zeros(T, num_lagrange_multipliers(h))
        end
        if isnothing(inner_settings)
            inner_settings = Clarabel.Settings()
        end
        if isnothing(outer_settings)
            outer_settings = OuterSettings()
        end

        ng = num_lagrange_multipliers(g)
        @assert length(λ) == ng "inequality constraint lagrange multipliers vector must have length $(ng) but has $(length(λ))"
        nh = num_lagrange_multipliers(h)
        @assert length(v) == nh "equality constraint lagrange multipliers vector must have length $(nh) but has $(length(v))"

        new{T}(
            mechanism,
            f,
            g,
            h,
            x,
            λ,
            v,
            inner_settings,
            outer_settings,
            Dict{Symbol,Any}(),
        )
    end
end

objectives(solver::SQPSolver) = solver.f

inequality_constraints(solver::SQPSolver) = solver.g

equality_constraints(solver::SQPSolver) = solver.h

inequality_duals(solver::SQPSolver) = solver.λ

equality_duals(solver::SQPSolver) = solver.v

primal(solver::SQPSolver) = solver.x

get_inner_settings(solver::SQPSolver) = solver.inner_settings

get_outer_settings(solver::SQPSolver) = solver.outer_settings

function initialize_trajectory(
    mechanism::Mechanism{T},
    N::Int,
    tf::T,
    nu::Int,
    q₀::AbstractVector{T},
    q₁::AbstractVector{T},
    v₀::AbstractVector{T},
    v₁::AbstractVector{T},
) where {T}
    nq = num_positions(mechanism)
    nv = num_velocities(mechanism)

    ts, qs, vs = straight_line_trajectory(N, tf, q₀, q₁, v₀, v₁)

    N = length(ts)
    nx = nq + nv
    knotpointsize = nx + nu
    num_decision_variables = N * knotpointsize
    zero_control_vector = zeros(nu)

    timesteps = diff(ts)
    # timesteps needs to be the same length as timestamps
    push!(timesteps, last(timesteps))

    knotpoints = Vector{T}(undef, num_decision_variables)

    for i = 1:N
        idx₀ = (i - 1) * knotpointsize + 1
        idx₁ = idx₀ + knotpointsize - 1
        knotpoints[idx₀:idx₁] = [qs[i]; vs[i]; zero_control_vector]
    end

    DiscreteTrajectory(ts, timesteps, knotpoints, knotpointsize, nx)
end

function num_lagrange_multipliers(constraints::AbstractVector{<:AdjacentKnotPointsFunction})
    isempty(constraints) && return 0
    sum(outputdim(c) * length(indices(c)) for c ∈ constraints)
end

function evaluate_objective(
    objectives::AbstractVector{<:AdjacentKnotPointsFunction},
    Z::DiscreteTrajectory,
)
    sum(objective(Val(Sum), Z) for objective ∈ objectives)
end

function super_gradient(
    objectives::AbstractVector{<:AdjacentKnotPointsFunction},
    Z::DiscreteTrajectory,
)
    z = knotpoints(Z)
    # Rest assured, no copying happening here
    fwrapped(z) = evaluate_objective(
        objectives,
        DiscreteTrajectory(time(Z), timesteps(Z), z, knotpointsize(Z), nstates(Z)),
    )
    ForwardDiff.gradient(fwrapped, z)
end

function evaluate_constraints(
    constraints::AbstractVector{<:AdjacentKnotPointsFunction},
    Z::DiscreteTrajectory{Ts,Tk},
) where {Ts,Tk}
    # TODO: preallocate before here
    result = Vector{Tk}()
    for constraint ∈ constraints
        val = constraint(Val(Stack), Z)
        append!(result, val)
    end
    return result
end

function super_jacobian(
    constraints::AbstractVector{<:AdjacentKnotPointsFunction},
    Z::DiscreteTrajectory{Ts,Tk},
) where {Ts,Tk}
    z = knotpoints(Z)
    # Rest assured, no copying happening here
    fwrapped(z) = evaluate_constraints(
        constraints,
        DiscreteTrajectory(time(Z), timesteps(Z), z, knotpointsize(Z), nstates(Z)),
    )
    ForwardDiff.jacobian(fwrapped, z)
end

function super_hessian_objective(
    objectives::AbstractVector{<:AdjacentKnotPointsFunction},
    Z::DiscreteTrajectory,
)
    z = knotpoints(Z)
    # Rest assured, no copying happening here
    fwrapped(z) = evaluate_objective(
        objectives,
        DiscreteTrajectory(time(Z), timesteps(Z), z, knotpointsize(Z), nstates(Z)),
    )
    result = DiffResults.HessianResult(z)
    result = ForwardDiff.hessian!(result, fwrapped, z)
    DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
end

function super_hessian_constraints(
    constraints::AbstractVector{<:AdjacentKnotPointsFunction},
    Z::DiscreteTrajectory,
    λ::AbstractVector{T},
) where {T}
    z = knotpoints(Z)
    n = length(z)
    if isempty(constraints)
        return T[], zeros(T, 0, n), Symmetric(zeros(T, n, n))
    end
    m = sum(length(indices(con)) * outputdim(con) for con ∈ constraints)
    y = zeros(T, m * n)
    H = DiffResults.JacobianResult(y, z)

    fwrapped(z) = evaluate_constraints(
        constraints,
        DiscreteTrajectory(time(Z), timesteps(Z), z, knotpointsize(Z), nstates(Z)),
    )

    # TODO: can't use ForwardDiff.jacobian! for innner jacobian
    # https://github.com/JuliaDiff/ForwardDiff.jl/issues/393
    H = ForwardDiff.jacobian!(H, z -> ForwardDiff.jacobian(fwrapped, z), z)

    # Outer Jacobian as 3-tensor: (i,j,k) = (output, ∂/∂z_j, ∂/∂z_k)
    G3 = reshape(DiffResults.jacobian(H), m, n, n)
    H3 = PermutedDimsArray(G3, (2, 3, 1))  # (n, n, m), one Hessian per output

    @assert length(λ) == size(H3, 3) "length(λ)=$(length(λ)) ≠ Hessian stack depth $(size(H3, 3))"

    ∑H = zeros(T, n, n)
    # λ-weighted sum of Hessians (no mutation of H3)
    @inbounds @views for k = 1:length(λ)
        ∑H .+= λ[k] .* H3[:, :, k]
    end
    # numeric symmetrization before wrapping to handle autodiff noise, maybe not
    # necessary?
    #    ∑H .= (∑H .+ ∑H') .* T(0.5)

    cval = evaluate_constraints(constraints, Z)
    Jmn = reshape(DiffResults.value(H), m, n)

    cval, Jmn, Symmetric(∑H)
end

negate!(x::AbstractArray) = x .*= -1

"""
Solve QP using Clarabel

minimize   1⁄2𝒙ᵀ𝑷𝒙 + 𝒒ᵀ𝒙
subject to  𝑨𝒙 + 𝒔 = 𝒃
                 𝒔 ∈ 𝑲
with decision variables 𝒙 ∈ ℝⁿ, 𝒔 ∈ 𝑲 and data matrices 𝑷 = 𝑷ᵀ ≥ 0,
𝒒 ∈ ℝⁿ, 𝑨 ∈ ℝᵐˣⁿ, and b ∈ ℝᵐ. The convext set 𝑲 is a composition of convex cones.
"""
function solve_qp(
    g::AbstractVector{T},
    Jg::AbstractMatrix{T},
    h::AbstractVector{T},
    Jh::AbstractMatrix{T},
    ▽L::AbstractVector{T},
    ▽²L::AbstractMatrix{T},
    inner_settings::Clarabel.Settings{T},
) where {T}
    P = sparse(▽²L)
    q = ▽L
    A = sparse([
        Jg;
        Jh;
    ])
    b = [
        g;
        h
    ]
    K = [Clarabel.NonnegativeConeT(length(g)), Clarabel.ZeroConeT(length(h))]

    if inner_settings.verbose
        println("P $(size(P)): ", P)
        println("q $(size(q)): ", q)
        println("A $(size(A)): ", A)
        println("b $(size(b)): ", b)
        println("K $(size(K)): ", K)
    end

    solver = Clarabel.Solver(P, q, A, b, K, inner_settings)
    solution = Clarabel.solve!(solver)
    # solution.x → primal solution
    # solution.z → dual solution
    # solution.s → slacks
    (solution.x, solution.z)
end

# ──── Null-space decomposition ────
#
# references:
# - "General Reduction Strategies For Linear Constraints" subsection of section 15.3 "Elimination of Variables" of Numerical Optimization, Nocedal & Wright (Second Edition)
# - Section 3.4. "Reduced Hessian SQP Methods" of Sequential Quadratic Programming, Paul T. Boggs (1996)
#
# Equality-constrained SQP step (ignoring inequalities):
#
#   min_p   ∇fᵀ p + ½ pᵀ ∇²L p
#   s.t.    Jh p + h = 0          (Jh is m×n, h ∈ ℝᵐ, n ≥ m)
#
# Factor the search direction in an orthonormal splitting of ℝⁿ tied to Jh:
#
#   range(Jhᵀ)  ⊂ ℝⁿ   — normal to the linearized constraint manifold  (Y ∈ ℝⁿˣᵐ)
#   null(Jh)    ⊂ ℝⁿ   — tangent to the linearized constraint manifold  (Z ∈ ℝⁿˣ⁽ⁿ⁻ᵐ⁾)
#
# so any p ∈ ℝⁿ is written uniquely as
#
#   p = Y p_y + Z p_z.
#
# Substitute into the linearized constraint:
#
#   Jh p = Jh Y p_y + Jh Z p_z = −h.
#
# By construction Jh Z = 0, so the range-space coefficient is fixed by
#
#   (Jh Y) p_y = −h    ⇒    p_y = −(Jh Y) \ h,
#
# and the free variable p_z is chosen by a reduced QP on Z.
# The same JhY ≔ Jh Y is reused for equality-multiplier recovery from QP
# stationarity (L = f + vᵀ h + …):
#
#   ∇f + ∇²L p + Jhᵀ v = 0.                         (KKT / ∇_p ℒ = 0)
#
# Split the residual in the orthonormal basis [Y Z]. The null-space block is
# already satisfied by construction of the reduced step (Zᵀ(∇f+∇²L p) = 0 when
# there are no inequalities, or is handled by the reduced QP duals), so drop it
# and keep only the range-space block — left-multiply by Yᵀ:
#
#   Yᵀ(∇f + ∇²L p) + Yᵀ Jhᵀ v = 0.
#
# With Jh Y = JhY and Yᵀ Y = I we have Yᵀ Jhᵀ = JhYᵀ, hence
#
#   JhYᵀ v = −Yᵀ(∇f + ∇²L p).
#
# Both QR and SVD of Jhᵀ (n×m) produce orthonormal [Y Z] with the properties
# above; they differ only in how JhY is obtained from the factorization.

function compute_null_range_bases(Jhᵀ::Matrix{T}; svd::Bool = false) where {T}
    n, m = size(Jhᵀ)
    if svd
        # Full SVD (σ sorted descending):
        #   Jhᵀ = U Σ Vᵀ,   U ∈ ℝⁿˣⁿ,  Σ = diag(σ) ∈ ℝᵐˣᵐ,  V ∈ ℝᵐˣᵐ
        #          = [Y Z] [Σ; 0] Vᵀ
        # with Y = U[:, 1:m], Z = U[:, m+1:n].
        #
        # Economy form used below: Jhᵀ = Y Σ Vᵀ.
        # Transpose both sides:
        #   Jh = V Σ Yᵀ.
        # Multiply on the right by Y (Yᵀ Y = I_m, Yᵀ Z = 0):
        #   Jh Y = V Σ Yᵀ Y = V Σ,
        #   Jh Z = V Σ Yᵀ Z = 0.
        # So JhY = V Σ is dense (not triangular) but nonsingular when rank(Jh)=m.
        F = svd(Jhᵀ; full = true)
        σ = F.S
        println("SVD σ_min=$(last(σ)), σ_max=$(first(σ))")
        Y = F.U[:, 1:m]
        Z = F.U[:, m+1:n]
        JhY = F.V * Diagonal(σ)
        return Y, Z, JhY
    else
        # Thin QR of the constraint transpose:
        #   Jhᵀ = Q₁ R = Y R,
        # where Y = Q₁ ∈ ℝⁿˣᵐ has orthonormal columns (range basis) and
        # R ∈ ℝᵐˣᵐ is upper triangular. Full Q = [Y Z] extends to an
        # orthonormal basis of ℝⁿ, so Z = Q₂ spans null(Jh).
        #
        # From Jhᵀ = Y R, transpose both sides:
        #   Jh = Rᵀ Yᵀ.
        # Multiply on the right by Y / Z (Yᵀ Y = I_m, Yᵀ Z = 0):
        #   Jh Y = Rᵀ Yᵀ Y = Rᵀ,
        #   Jh Z = Rᵀ Yᵀ Z = 0.
        # So JhY = Rᵀ is triangular — cheap to factor for p_y and v.
        F = qr(Jhᵀ)
        Y = Matrix(F.Q)              # thin Q = Q₁ (n×m)
        Q_full = F.Q * I             # full Q = [Y Z] (n×n)
        Z = Q_full[:, m+1:end]       # Q₂
        JhY = F.R'
        return Y, Z, JhY
    end
end

function standard_auglag_merit(f::T, h::AbstractVector{T}, v::AbstractVector{T},
                                μ::T) where {T}
    return f + v' * h + (μ / 2) * (h' * h)
end

function least_squares_multipliers(Jh::AbstractMatrix{T}, ∇f::AbstractVector{T}) where {T}
    δ = T(1e-8)
    return (Jh * Jh' + δ * I) \ (Jh * ∇f)
end

function fletcher_merit(f::T, h::AbstractVector{T}, ∇f::AbstractVector{T},
                        Jh::AbstractMatrix{T}, μ::T) where {T}
    λ = least_squares_multipliers(Jh, ∇f)
    return f - λ' * h + (μ / 2) * (h' * h)
end

function solve!(
    solver::SQPSolver{T},
    custom_gradients::Bool = false,
    expose_guts::Bool = true,
) where {T}
    inner_settings = get_inner_settings(solver)
    outer_settings = get_outer_settings(solver)
    μ = outer_settings.μ_init
    for k = 1:outer_settings.max_iter
        x = primal(solver)
        λ = inequality_duals(solver)
        v = equality_duals(solver)

        if custom_gradients
            f = evaluate_objective(objectives(solver), x)
            ▽f = gradient(Val(Sum), objectives(solver), x)
            ▽²f = hessian(objectives(solver), x)

            g = evaluate_constraints(inequality_constraints(solver), x)
            Jg = jacobian(inequality_constraints(solver), x)
            ▽²g = vector_hessian(inequality_constraints(solver), x, λ)

            h = evaluate_constraints(equality_constraints(solver), x)
            Jh = jacobian(equality_constraints(solver), x)
            ▽²h = vector_hessian(equality_constraints(solver), x, v)
        else
            f, ▽f, ▽²f = super_hessian_objective(objectives(solver), x)
            g, Jg, ▽²g =
                super_hessian_constraints(inequality_constraints(solver), x, λ)
            h, Jh, ▽²h =
                super_hessian_constraints(equality_constraints(solver), x, v)
        end

        L = f + λ' * g + v' * h
        ▽L = ▽f + Jg' * λ + Jh' * v
        ▽²L = ▽²f + ▽²g + ▽²h

        # ── Null-space decomposition ──
        # p = Y p_y + Z p_z  with  range(Y)=range(Jhᵀ),  range(Z)=null(Jh)
        # (see compute_null_range_bases). Linearized equalities fix p_y only:
        #   Jh p = Jh Y p_y = −h  (since Jh Z = 0).
        ∇h = Matrix(Jh')
        Y, Z, JhY = compute_null_range_bases(∇h)

        # Range-space step: (Jh Y) p_y = −h
        p_y = -JhY \ h

        # Reduced QP in p_z (null-space / tangential step).
        # With p = Y p_y + Z p_z and p_y fixed, the quadratic model becomes
        #   min_{p_z}  ½ p_zᵀ (Zᵀ ∇²L Z) p_z + (Zᵀ(∇L + ∇²L Y p_y))ᵀ p_z
        # and linearized inequalities g + Jg p ≤ 0 become
        #   g + Jg Y p_y + Jg Z p_z ≤ 0.
        R_red = Z' * ▽²L * Z
        λ_min_R = minimum(eigvals(Symmetric(R_red)))
        if λ_min_R ≤ 0
            R_red .+= (abs(λ_min_R) + 1e-8) * I(size(R_red, 1))
        end

        g_red = Z' * (▽L + ▽²L * (Y * p_y))
        g_ineq_red = g + Jg * (Y * p_y)
        Jg_red = Jg * Z

        # solve_qp encodes Ap ≤ b via NonnegativeCone with A = Jg_red, b = −g_ineq_red
        p_z, l_red = solve_qp(
            -g_ineq_red, Jg_red,
            T[], zeros(T, 0, size(R_red, 1)),
            g_red, R_red, inner_settings,
        )

        # Full step in the original coordinates
        pₖ = Y * p_y + Z * p_z

        # Equality multipliers from KKT stationarity (see null-space header):
        #   ∇f + ∇²L p + Jhᵀ v = 0
        # drop null-space block (Zᵀ · …), keep range-space block (Yᵀ · …):
        #   JhYᵀ v = −Yᵀ(∇f + ∇²L p)
        v_qp = -(JhY' \ (Y' * (▽f + ▽²L * pₖ)))
        λ_qp = l_red

        # Deltas for step-form update
        dv = v_qp - v
        dλ = λ_qp - λ

        if expose_guts
            push!(
                get!(solver.guts, :primal, Vector{DiscreteTrajectory{T,T}}()),
                deepcopy(x),
            )
        end

        # ── Step length: backtracking Armijo ──
        # Update μ by (18.36): μ ≥ (∇fᵀp + (σ/2)pᵀ∇²Lp) / ((1−ρ)||h||₁)
        # σ = 1 if pᵀ∇²L p > 0, else 0 (equation 18.37)
        pBp = pₖ' * ▽²L * pₖ
        σ = pBp > 0 ? T(1) : T(0)
        denom = (1 - T(0.1)) * sum(abs, h; init=zero(T))
        if denom > 0
            μ_min = (▽f' * pₖ + (σ / 2) * pBp) / denom
            # TODO: This max() implements exactly what "Numerical Optimization
            # Nocedal & Wright" says: "If the value of µ from the previous
            # iteration of the SQP method satisfies (18.36), it is left
            # unchanged" might need to add a decay mechanism to handle badly
            # scaled early steps.
            μ = max(μ, μ_min)
        end

        if outer_settings.merit_function == :fletcher
            # φ_F(x) = f(x) − λ(x)ᵀh(x) + (μ/2)||h(x)||²
            # λ(x) = (Jh·Jhᵀ)⁻¹Jh·∇f
            # ∇φ_F = ∇f − Jhᵀλ + μ·Jhᵀh − ∇λ·h
            λ_ls = least_squares_multipliers(Jh, ▽f)
            φ_curr = f - λ_ls' * h + (μ / 2) * (h' * h)
            # the ∇λ·h term requires ∂²c and ∂²f (third-order tensor), which is
            # expensive to compute. Dropped for now; vanishes as h→0.
            ∇φᵀpₖ = (▽f - Jh' * λ_ls + μ * Jh' * h)' * pₖ
        else  # :standard
            # φ_S(x) = f(x) + vᵀh(x) + (μ/2)||h(x)||²
            # ∇φ_S = ∇f + Jhᵀv + μ·Jhᵀh
            φ_curr = standard_auglag_merit(f, h, v, μ)
            ∇φᵀpₖ = (▽f + Jh' * (v + μ * h))' * pₖ
        end

        α = one(T)
        for _ in 1:20
            x_trial = deepcopy(x)
            x_trial.knotpoints .+= α * pₖ
            f_trial = evaluate_objective(solver.f, x_trial)
            h_trial = evaluate_constraints(solver.h, x_trial)
            if outer_settings.merit_function == :fletcher
                Jh_trial = jacobian(solver.h, x_trial)
                ▽f_trial = super_gradient(solver.f, x_trial)
                φ_trial = fletcher_merit(f_trial, h_trial, ▽f_trial, Jh_trial, μ)
            else
                φ_trial = standard_auglag_merit(f_trial, h_trial, v, μ)
            end
            # Armijo sufficient decrease: φ(x+αp) ≤ φ(x) + c·α·∇φᵀp  (c = 1e-4)
            if φ_trial ≤ φ_curr + T(1e-4) * α * ∇φᵀpₖ
                break
            end
            α *= T(0.5)
        end

        # solution step
        kp = knotpoints(x)
        @. kp += α * pₖ

        # Update multipliers in step form: v_{k+1} = v_k + α·dv
        if !isempty(dλ)
            @. λ += α * dλ
        end
        if !isempty(dv)
            @. v += α * dv
        end

        if expose_guts && k == outer_settings.max_iter
            push!(get!(solver.guts, :primal, Vector{DiscreteTrajectory{T,T}}()), x)
            push!(get!(solver.guts, :inequality_duals, Vector{Vector{T}}()), λ)
            push!(get!(solver.guts, :equality_duals, Vector{Vector{T}}()), v)
            push!(get!(solver.guts, :objective, Vector{T}()), f)
            push!(get!(solver.guts, :lagrangian, Vector{T}()), L)
        end

        if outer_settings.verbose
            println("primal x $(length(knotpoints(x))): ", x)
            println("dual λ $(length(λ)): ", λ)
            println("dual v $(length(v)): ", v)

            println("f $(length(f)): ", f)
            println("▽f $(size(▽f)): ", ▽f)
            println("▽²f $(size(▽²f)): ", ▽²f)

            println("g $(size(g)): ", g)
            println("Jg $(size(Jg)): ", Jg)
            println("▽²g $(size(▽²g)): ", ▽²g)

            println("h $(size(h)): ", h)
            println("Jh $(size(Jh)): ", Jh)
            println("▽²h $(size(▽²h)): ", ▽²h)

            println("L $(size(L)): ", L)
            println("▽L $(size(▽L)): ", ▽L)
            println("▽²L $(size(▽²L)): ", ▽²L)

            println("null-space decomp:")
            println("  Y  (range)   = $(size(Y))")
            println("  Z  (null)    = $(size(Z))")
            println("  JhY = Rᵀ    = $(size(JhY))")
            println("  p_y          = $(length(p_y))")
            println("  R_red        = $(size(R_red))")
            println("  g_red        = $(length(g_red))")
            println("  g_ineq_red   = $(length(g_ineq_red))")
            println("  Jg_red       = $(size(Jg_red))")
            println("  p_z          = $(length(p_z))")

            println("step pₖ $(length(pₖ)): ", pₖ)
        end
    end
end
