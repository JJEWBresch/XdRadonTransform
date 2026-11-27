# Fast implementation of `R(P)` and `Rθ(P)` (RadonTransform.jl) and the adjoint `backproject`.
#
# Same model and conventions as the original code in RadonTransform.jl, which stays available via
# `R(P; legacy=true)`. Differences: stable closed-form kernels (below), allocation-free type-stable
# loops threaded over directions (or voxels), Float32 or Float64 by the element type of the phantom.
# Implementation notes and timings: benchmark/README.md.

# ================================================================================================
# Kernels
# ================================================================================================

# Radon transform of one voxel (the paper's A_θ^a, Thm. 1) and its slab average (Cor. 6),
# evaluated in a numerically stable way.
#
# With b_i = a_i |θ_i| (a = voxel half-sizes), the paper's Lemma 1 / Thm. 1 say
#
#     A_θ^a(t) = vol · (u_1 * … * u_ℓ)(t),      vol = ∏ 2a_i = voxel volume,
#
# where u_i = 1/(2b_i) 1_{[-b_i,b_i]} are normalized boxes (the components with θ_i = 0 drop out),
# i.e. vol times the density of a sum of independent uniform variables. The slab average
# (2ε)⁻¹ V_θ^a(t-ε, t+ε) is the same with one more box of half-width ε = w/2.
#
# The alternating sum Σ_k P(k)(t + ⟨k,b⟩)_+^{ℓ-1} of Thm. 1 cancels catastrophically if some b_i is
# much smaller than the others (directions close to an axis); in Float32 this already happens for
# b_i/b_max ≈ 1e-3. Here, the boxes are sorted, b_1 ≤ … ≤ b_N, and the density is built from the
# centred antiderivatives of the convolution of the smaller boxes, each differenced only over the
# largest box so far:
#
#     f_N(t) = (G_{N-1}(t + b_N) - G_{N-1}(t - b_N)) / (2b_N),     G_{N-1}(s) = ∫_0^s f_{N-1},
#
# with closed forms for G_1, G_2 (piecewise linear/quadratic), G_3 via the antiderivative K_2 of G_2
# and G_4 via the antiderivatives K_2, L_2 of G_2. Every division is by the largest half-width of the
# boxes involved, so the relative error stays O(eps). N ≤ 5 (the plane and the slab in 2D, 3D and
# 4D) is written out; larger N uses the same closed forms generated per N (`GeneratedCubeKernel`).
# `generic = true` uses the recursion `generic_density` (the same idea down to single boxes, with
# branches) for all N.

"""
    CubeKernel{N,T}

Radon transform `A(t)` of a voxel for one direction: `scale` times the density of the sum of `N`
independent uniform random variables on `[-b_i, b_i]`, `0 < b_1 ≤ … ≤ b_N` (the convolution of
`N` normalized boxes). Evaluate with `K(t)`.

Built by [`cube_kernel`](@ref). For `N ≥ 6` see [`GeneratedCubeKernel`](@ref), for `generic = true` `generic_density`.
"""
struct CubeKernel{N,T<:AbstractFloat}
    b::NTuple{N,T}      # sorted half-widths, all > 0
    scale::T            # voxel volume
    support::T          # Σ b_i: K(t) = 0 for |t| > support
    c::NTuple{10,T}     # precomputed constants, depending on N
end

"""
    cube_kernel(halfsize, θ; width=0, threshold=1e-15, generic=false) -> CubeKernel

Kernel `t ↦ A_θ^a(t)` (paper, Thm. 1) of the voxel `(-a, a]`, `a = halfsize`, for the direction `θ`,
or for `width > 0` its average over the slab `[t - width/2, t + width/2]` (Cor. 6 divided by the
width). Components with `|a_i θ_i| ≤ threshold` are treated as zero (as `reduce` in the original
code), and also those below `eps(T)` times the largest one. `generic = true`: always the recursion
([`GenericCubeKernel`](@ref)), also for ≤ 5 boxes (for comparisons; slower, same values to rounding).
"""
function cube_kernel(a::NTuple{D,T}, θ::NTuple{D,T}; width::T = zero(T), threshold::T = T(1e-15),
                     generic::Bool = false) where {D,T<:AbstractFloat}
    scale = prod(2 .* a)
    bs = map((ai, θi) -> abs(ai * θi), a, θ)
    bmax = max(maximum(bs), width / 2)
    tol = max(threshold, eps(T) * bmax)
    # keep the nonzero half-widths (and the slab), sorted
    v = T[x for x in bs if x > tol]
    width > 0 && push!(v, width / 2)
    sort!(v)
    N = length(v)
    N == 0 && throw(ArgumentError("direction θ = $θ is zero"))
    b = Tuple(v)::NTuple{N,T}
    return generic ? GenericCubeKernel(b, scale) : _make_kernel(Val(N), b, scale)
end
cube_kernel(a::NTuple{D,<:Real}, θ::NTuple{D,<:Real}; kwargs...) where {D} =
    cube_kernel(float.(a), float.(θ); kwargs...)

_pad10(x::Tuple, ::Type{T}) where {T} = ntuple(i -> i <= length(x) ? T(x[i]) : zero(T), 10)

function _make_kernel(::Val{N}, b::NTuple{N,T}, scale::T) where {N,T}
    S = sum(b)
    half = T(1) / 2
    if N == 1
        b1, = b
        c = (scale / (2b1),)
    elseif N == 2
        b1, b2 = b
        # trapezoid: scale/(4b1b2) · clamp(b1 + b2 - |t|, 0, 2b1)
        c = (scale / (4 * b1 * b2), b1 + b2, 2b1)
    elseif N == 3
        b1, b2, b3 = b
        # G2 constants and the outer difference over b3
        c = (b2 - b1, b1 + b2, 1 / (2b2), 1 / (8 * b1 * b2), b3, scale / (2b3), half)
    elseif N == 4
        b1, b2, b3, b4 = b
        # K2 constants, G3 = (K2(s+b3) - K2(s-b3))/(2b3), outer difference over b4
        c = (b2 - b1, b1 + b2, 1 / (4b2), b2 / 4 + b1^2 / (12b2), 1 / (24 * b1 * b2),
             b3, 1 / (2b3), b1 + b2 + b3, b4, scale / (2b4))
    elseif N == 5
        b1, b2, b3, b4, b5 = b
        # L2 constants, K3 = (L2(s+b3) - L2(s-b3))/(2b3), G4 = (K3(s+b4) - K3(s-b4))/(2b4), outer
        # difference over b5 (b3, b4 from K.b, S3 = b1 + b2 + b3 computed)
        c = (b2 - b1, b1 + b2, 1 / (12b2), b2 / 4 + b1^2 / (12b2), (b1^2 + b2^2) / 12, 1 / (96 * b1 * b2),
             1 / (2b3), 1 / (2b4), b1 + b2 + b3 + b4, scale / (2b5))
    else
        return GeneratedCubeKernel(b, scale)
    end
    return CubeKernel{N,T}(b, scale, S, _pad10(c, T))
end

"""
    GenericCubeKernel{N,T}

As [`CubeKernel`](@ref), evaluated by the recursion of `generic_density` with precomputed partial
sums and moments, for any `N`; used with `generic = true` (see [`cube_kernel`](@ref)).
"""
struct GenericCubeKernel{N,T<:AbstractFloat}      # isbits (also usable on the GPU)
    b::NTuple{N,T}
    scale::T
    support::T
    S::NTuple{N,T}          # partial sums b_1 + … + b_k
    μ::NTuple{N,NTuple{N,T}} # μ[k][j+1] = E[(U_1 + … + U_k)^j], j < N
end
function GenericCubeKernel(b::NTuple{N,T}, scale::T) where {N,T}
    S = Tuple(cumsum(collect(b)))
    return GenericCubeKernel{N,T}(b, scale, S[N], S, _box_sum_moments(b))
end

support(K::CubeKernel) = K.support
support(K::GenericCubeKernel) = K.support

# --- closed forms -------------------------------------------------------------------------------

# N = 1: box, with the value 1/2 at the jumps (χ(0) = 1/2 as the original code)
@inline function (K::CubeKernel{1,T})(t::T) where {T}
    u = abs(t)
    b1 = K.b[1]
    c = K.c[1]
    return ifelse(u < b1, c, ifelse(u == b1, c / 2, zero(T)))
end

# min/max/clamp without NaN propagation: one instruction (Base.min/max are NaN-propagating, which
# costs several instructions on x86); no NaN can occur here
@inline _fmin(a, b) = ifelse(a < b, a, b)
@inline _fmax(a, b) = ifelse(a > b, a, b)
@inline _fclamp(x, lo, hi) = _fmin(_fmax(x, lo), hi)

# N = 2: trapezoid
@inline function (K::CubeKernel{2,T})(t::T) where {T}
    inv, s, m = K.c[1], K.c[2], K.c[3]
    return inv * _fmin(m, _fmax(zero(T), s - abs(t)))
end

# G2(s) = ∫_0^s (u_1 * u_2), odd; for x = |s|:
#   x ≤ b2 - b1:          x / (2b2)
#   b2 - b1 < x < b1 + b2: 1/2 - (b1 + b2 - x)² / (8 b1 b2)
#   x ≥ b1 + b2:          1/2
@inline function _G2(s::T, d21::T, s12::T, i2b2::T, i8::T) where {T}
    x = abs(s)
    r = s12 - x
    g = ifelse(x <= d21, x * i2b2, ifelse(x < s12, T(0.5) - r * r * i8, T(0.5)))
    return copysign(g, s)
end

# N = 3: (G2(t + b3) - G2(t - b3)) / (2b3)
@inline function (K::CubeKernel{3,T})(t::T) where {T}
    d21, s12, i2b2, i8, b3, outer = K.c[1], K.c[2], K.c[3], K.c[4], K.c[5], K.c[6]
    u = abs(t)
    v = outer * (_G2(u + b3, d21, s12, i2b2, i8) - _G2(u - b3, d21, s12, i2b2, i8))
    return ifelse(u < K.support, v, zero(T))
end

# K2(s) = ∫_0^s G2, even; for x = |s|, r = b1 + b2 - x, c0 = b2/4 + b1²/(12 b2):
#   x ≤ b2 - b1:          x² / (4 b2)
#   b2 - b1 < x < b1 + b2: x/2 - c0 + r³ / (24 b1 b2)
#   x ≥ b1 + b2:          x/2 - c0
@inline function _K2(s::T, d21::T, s12::T, i4b2::T, c0::T, i24::T) where {T}
    x = abs(s)
    r = s12 - x
    lin = x / 2 - c0
    return ifelse(x <= d21, x * x * i4b2, ifelse(x < s12, lin + r * r * r * i24, lin))
end

# G3(s) = ∫_0^s (u_1 * u_2 * u_3) = (K2(s + b3) - K2(s - b3)) / (2b3), = ±1/2 outside the support
@inline function _G3(s::T, d21, s12, i4b2, c0, i24, b3, i2b3, S3) where {T}
    v = (_K2(s + b3, d21, s12, i4b2, c0, i24) - _K2(s - b3, d21, s12, i4b2, c0, i24)) * i2b3
    return ifelse(abs(s) < S3, v, copysign(T(0.5), s))
end

# N = 4: (G3(t + b4) - G3(t - b4)) / (2b4)
@inline function (K::CubeKernel{4,T})(t::T) where {T}
    d21, s12, i4b2, c0, i24, b3, i2b3, S3, b4, outer = K.c
    u = abs(t)
    v = outer * (_G3(u + b4, d21, s12, i4b2, c0, i24, b3, i2b3, S3) -
                 _G3(u - b4, d21, s12, i4b2, c0, i24, b3, i2b3, S3))
    return ifelse(u < K.support, v, zero(T))
end

# L2(s) = ∫_0^s K2, odd; for x = |s|, r = b1 + b2 - x, c0 as in K2, c1 = (b1² + b2²)/12:
#   x ≤ b2 - b1:          x³ / (12 b2)
#   b2 - b1 < x < b1 + b2: x²/4 - c0 x + c1 - r⁴ / (96 b1 b2)
#   x ≥ b1 + b2:          x²/4 - c0 x + c1
@inline function _L2(s::T, d21::T, s12::T, i12b2::T, c0::T, c1::T, i96::T) where {T}
    x = abs(s)
    r = s12 - x
    q = x * (x / 4 - c0) + c1
    r2 = r * r
    g = ifelse(x <= d21, x * x * x * i12b2, ifelse(x < s12, q - r2 * r2 * i96, q))
    return copysign(g, s)
end

# K3(s) = (L2(s + b3) - L2(s - b3)) / (2b3), even, an antiderivative of G3; = |s|/2 - c0 outside the
# support (exact there; the closed form avoids the cancellation for |s| ≫ b3)
@inline function _K3(s::T, d21, s12, i12b2, c0, c1, i96, b3, i2b3, S3) where {T}
    x = abs(s)
    v = (_L2(x + b3, d21, s12, i12b2, c0, c1, i96) - _L2(x - b3, d21, s12, i12b2, c0, c1, i96)) * i2b3
    return ifelse(x < S3, v, x / 2 - c0)
end

# G4(s) = ∫_0^s (u_1 * … * u_4) = (K3(s + b4) - K3(s - b4)) / (2b4), = ±1/2 outside the support
@inline function _G4(s::T, d21, s12, i12b2, c0, c1, i96, b3, i2b3, S3, b4, i2b4, S4) where {T}
    v = (_K3(s + b4, d21, s12, i12b2, c0, c1, i96, b3, i2b3, S3) -
         _K3(s - b4, d21, s12, i12b2, c0, c1, i96, b3, i2b3, S3)) * i2b4
    return ifelse(abs(s) < S4, v, copysign(T(0.5), s))
end

# N = 5 (slab in 4D): (G4(t + b5) - G4(t - b5)) / (2b5)
@inline function (K::CubeKernel{5,T})(t::T) where {T}
    d21, s12, i12b2, c0, c1, i96, i2b3, i2b4, S4, outer = K.c
    b3, b4, b5 = K.b[3], K.b[4], K.b[5]
    S3 = s12 + b3
    u = abs(t)
    v = outer * (_G4(u + b5, d21, s12, i12b2, c0, c1, i96, b3, i2b3, S3, b4, i2b4, S4) -
                 _G4(u - b5, d21, s12, i12b2, c0, c1, i96, b3, i2b3, S3, b4, i2b4, S4))
    return ifelse(u < K.support, v, zero(T))
end

# --- any number of boxes ------------------------------------------------------------------------

(K::GenericCubeKernel{N,T})(t::T) where {N,T} = K.scale * _J(K.b, K.S, K.μ, t)

"""
    generic_density(b, t)

Density at `t` of the sum of independent uniform variables on `[-b_i, b_i]` (`b` sorted ascending,
all `> 0`), i.e. the normalized convolution of boxes, for any number of boxes. Uses the recursion

    J_k^{(m)}(s) = (J_{k-1}^{(m+1)}(s + b_k) - J_{k-1}^{(m+1)}(s - b_k)) / (2b_k),

for the `m`-th centred antiderivative `J_k^{(m)} = K_m * f_k` (`K_m(s) = sign(s)^m |s|^{m-1}/(2(m-1)!)`)
of the density `f_k` of the first `k` boxes, and its closed form (from the moments of `f_k`) outside
the support `|s| > b_1 + … + b_k`. Each difference is taken over the largest box so far, which keeps
the evaluation stable (unlike the alternating sum of Thm. 1). Cost `O(2^N)`.
"""
generic_density(b::NTuple{N,T}, t::T) where {N,T} =
    _J(b, Tuple(cumsum(collect(b))), _box_sum_moments(b), t)

# moments μ[k][j+1] = E[(U_1 + … + U_k)^j], j = 0..N-1 (the recursion needs j ≤ N-2), of the sum of
# the first k uniforms
function _box_sum_moments(b::NTuple{N,T}) where {N,T}
    μ = zeros(T, N, N)
    prev = [j == 0 ? one(T) : zero(T) for j in 0:N-1]
    for k in 1:N
        ν = [iseven(i) ? b[k]^i / (i + 1) : zero(T) for i in 0:N-1]
        μ[:, k] = prev = [sum(binomial(j, i) * prev[j-i+1] * ν[i+1] for i in 0:j) for j in 0:N-1]
    end
    return ntuple(k -> ntuple(j -> μ[j, k], Val(N)), Val(N))
end

# The recursion for k = N boxes and m = 0, written out as one expression tree (2^N leaves) when the
# function is compiled for N (no recursion at run time or in type inference).
@generated function _J(b::NTuple{N,T}, S, μ, s::T) where {N,T}
    function build(k, m, arg)
        a = gensym(:s)
        if k == 0          # K_m(a), m ≥ 1
            val = m == 1 ? :(sign($a) / 2) : :(sign($a)^$m * abs($a)^$(m - 1) / $(T(2 * factorial(m - 1))))
            return :(let $a = $arg; $val; end)
        end
        if m == 0
            tail = :(zero($T))
        else                # closed form outside the support: sign^m Σ_j C(m-1,j) |a|^(m-1-j) μ_j / (2(m-1)!)
            x = gensym(:x)
            terms = [:($(T(binomial(m - 1, j) / (2 * factorial(m - 1)))) * $x^$(m - 1 - j) * μ[$k][$(j + 1)]) for j in 0:2:m-1]
            v = Expr(:call, :+, terms..., :(zero($T)))
            tail = isodd(m) ? :(let $x = abs($a); copysign($v, $a); end) : :(let $x = abs($a); $v; end)
        end
        plus, minus = build(k - 1, m + 1, :($a + b[$k])), build(k - 1, m + 1, :($a - b[$k]))
        return :(let $a = $arg
                     abs($a) > S[$k] ? $tail : ($plus - $minus) / (2 * b[$k])
                 end)
    end
    return build(N, 0, :s)
end

# --- closed forms for any number of boxes -------------------------------------------------------

"""
    GeneratedCubeKernel{N,T}

As [`CubeKernel`](@ref) for `N ≥ 6` boxes (dimension ≥ 6, or 5 with a slab): the closed forms of
`CubeKernel{3..5}` continued to any `N`, generated per `N`. Base: the `m`-th antiderivative
`B_m(s) = ∫_0^s B_{m-1}`, `m = N - 2`, of the trapezoid of the two smallest boxes (`G2`, `K2`, `L2`
for `m = 1, 2, 3`), with parity `(-1)^m` and for `x = |s|`, `r = b1 + b2 - x`:

    x ≤ b2 - b1:          x^m / (m! 2b2)
    b2 - b1 < x < b1 + b2: P_m(x) + (-1)^m r^(m+1) / ((m+1)! 4b1b2)
    x ≥ b1 + b2:          P_m(x)

with the tail polynomial `P_m(x) = Σ_{i<m} C(m-1,i) (-1)^i E|U1 + U2|^i / 2 · x^(m-1-i) / (m-1)!`,

and then one difference `(F(s + b_k) - F(s - b_k)) / (2b_k)` per box `k = 3..N`, each with its exact
polynomial outside the support `|s| ≥ b1 + … + b_k` (the difference of the tail polynomial of the
level below, expanded: no cancellation for `|s| ≫ b_k`). Branch free (`ifelse`), `2^(N-2)` base
evaluations, isbits (GPU).
"""
struct GeneratedCubeKernel{N,T<:AbstractFloat}
    b::NTuple{N,T}
    scale::T
    support::T
    outer::T                    # scale / (2b_N)
    S::NTuple{N,T}              # partial sums b_1 + … + b_k
    i2b::NTuple{N,T}            # 1 / (2b_k)
    base::NTuple{4,T}           # b2 - b1, b1 + b2, 1/(m! 2b2), (-1)^m/((m+1)! 4b1b2)
    P::NTuple{N,T}              # tail polynomial of the base (ascending coefficients), degree m - 1
    Q::NTuple{N,NTuple{N,T}}    # Q[k]: tail polynomial at level k = 3..N-1, degree N - k - 1
end
support(K::GeneratedCubeKernel) = K.support

function GeneratedCubeKernel(b::NTuple{N,T}, scale::T) where {N,T}
    N >= 2 || throw(ArgumentError("GeneratedCubeKernel: at least 2 boxes"))
    W = promote_type(T, Float64)                     # constants computed in (at least) Float64
    bw = map(W, b)
    b1, b2 = bw[1], bw[2]
    d, s12 = b2 - b1, b1 + b2
    m = N - 2
    fact(k) = W(factorial(k))
    # absolute moments h_i = E|U1 + U2|^i (density 1/(2b2) on [0, d], linear on [d, s12]), as sums of
    # positive terms: ∫_d^s12 y^i (s12 - y) dy with y = d + (2b1 - u)
    h = [2 * (d^(i + 1) / ((i + 1) * 2b2) +
              sum(binomial(i, j) * d^(i - j) * (2b1)^(j + 2) / ((j + 1) * (j + 2)) for j in 0:i) / (4b1 * b2))
         for i in 0:max(m - 1, 0)]
    # P_m(x) = Σ_{i<m} C(m-1,i) (-1)^i h_i/2 · x^(m-1-i) / (m-1)!   (∫_0^x of the density, iterated)
    P = zeros(W, N)
    for i in 0:m-1
        P[m-i] = binomial(m - 1, i) * (-1)^i * h[i+1] / 2 / fact(m - 1)
    end
    # tails of the levels: Q_k(x) = (Q_{k-1}(x + b_k) - Q_{k-1}(x - b_k)) / (2b_k), Q_2 = P
    Q = [zeros(W, N) for _ in 1:N]
    q = P
    for k in 3:N-1
        qn = zeros(W, N)
        for i in 0:N-1, j in 1:2:i
            qn[i-j+1] += q[i+1] * binomial(i, j) * bw[k]^(j - 1)
        end
        Q[k] = q = qn
    end
    S = Tuple(cumsum(collect(bw)))
    tup(v) = ntuple(i -> T(v[i]), Val(N))
    return GeneratedCubeKernel{N,T}(b, scale, T(S[N]), T(N == 2 ? scale : scale / (2bw[N])),
        tup(S), tup([1 / (2x) for x in bw]),
        (T(d), T(s12), T(1 / (fact(m) * 2b2)), T((-1)^m / (fact(m + 1) * 4b1 * b2))),
        tup(P), ntuple(k -> tup(Q[k]), Val(N)))
end

@generated function (K::GeneratedCubeKernel{N,T})(t::T) where {N,T}
    m0 = N - 2
    function horner(c, x, deg)          # c[1] + c[2] x + … + c[deg+1] x^deg
        ex = :($c[$(deg + 1)])
        for i in deg:-1:1
            ex = :($c[$i] + $x * $ex)
        end
        return ex
    end
    # x^k as products (Base.:^ with an exponent > 3 calls the slow exact power function)
    pw(x, k) = k == 0 ? :(one($T)) : k == 1 ? x : Expr(:call, :*, fill(x, k)...)
    function base(arg)                  # B_{m0}
        a, x, r, p = gensym(:a), gensym(:x), gensym(:r), gensym(:p)
        pex = m0 == 0 ? :(zero($T)) : horner(:Pc, x, m0 - 1)
        g = :(ifelse($x <= d21, $(pw(x, m0)) * cin, ifelse($x < s12, $p + $(pw(r, m0 + 1)) * cmid, $p)))
        val = isodd(m0) ? :(copysign($g, $a)) : g
        return :(let $a = $arg, $x = abs($a), $r = s12 - $x, $p = $pex
                     $val
                 end)
    end
    child(k, s) = k == 2 ? base(s) : level(k, s)
    function level(k, arg)              # 3 ≤ k ≤ N-1, antiderivative of order m = N - k ≥ 1
        m = N - k
        a, x = gensym(:a), gensym(:x)
        tail = horner(:(Q[$k]), x, m - 1)
        if iseven(m)                    # even: evaluated at |s|
            v = :(($(child(k - 1, :($x + b[$k]))) - $(child(k - 1, :($x - b[$k])))) * i2b[$k])
            return :(let $x = abs($arg)
                         ifelse($x < S[$k], $v, $tail)
                     end)
        else
            v = :(($(child(k - 1, :($a + b[$k]))) - $(child(k - 1, :($a - b[$k])))) * i2b[$k])
            return :(let $a = $arg, $x = abs($a)
                         ifelse($x < S[$k], $v, copysign($tail, $a))
                     end)
        end
    end
    top = N == 2 ? :(K.outer * $(base(:u))) :
          :(K.outer * ($(child(N - 1, :(u + b[$N]))) - $(child(N - 1, :(u - b[$N])))))
    return quote
        $(Expr(:meta, :inline))
        b, S, i2b, Q, Pc = K.b, K.S, K.i2b, K.Q, K.P
        d21, s12, cin, cmid = K.base
        u = abs(t)
        v = $top
        ifelse(u < K.support, v, zero($T))
    end
end

# ================================================================================================
# Forward transform and adjoint
# ================================================================================================

# Forward transform and its adjoint (backprojection), voxel driven on the CPU.
#
# Per direction θ, the projections ⟨c_j, θ⟩ = Σ_i c_{j_i} θ_i of the voxel centres are separable,
# so they are tabulated per axis (`proj[i][j_i]`). Each nonzero voxel adds v · A_θ(⟨c_j,θ⟩ - t_k) to
# the radii t_k in its support |⟨c_j,θ⟩ - t_k| ≤ Σ b_i (closed interval: the box kernel of an axis
# direction has the value 1/2 at its jumps). For equispaced radii, the index range is computed
# directly (with a margin of one radius, the kernel vanishes there), otherwise by binary search.
# Directions are distributed over the threads; with fewer directions than threads, the voxels are.

using Base.Cartesian: @nloops, @nref

# --- radii lookup -------------------------------------------------------------------------------

# Radii t_k = r1 + (k-1)Δ + e_k with |e_k| ≤ τΔ, τ ≤ 1e-3 (rounding, or small perturbations) count as
# equispaced: the index range of a voxel is then computed from the grid, widened by τ; the kernel is
# always evaluated at the actual radii. Other (sorted) radii: binary search.
struct EquispacedRadii{T}
    r1::T
    invΔ::T
    τ::T          # max. deviation from the grid, in units of Δ
    n::Int
end
struct SortedRadii
    n::Int
end

const EQUISPACED_TOL = 1e-3

function radii_index(radii::AbstractVector{T}) where {T}
    n = length(radii)
    if n >= 2
        Δ = (radii[end] - radii[1]) / (n - 1)
        if Δ > 0
            τ = maximum(k -> abs(radii[k] - (radii[1] + (k - 1) * Δ)), 1:n) / Δ
            τ <= EQUISPACED_TOL && return EquispacedRadii{T}(radii[1], 1 / Δ, T(τ), n)
        end
    end
    return SortedRadii(n)
end

# indices k with lo ≤ radii[k] ≤ hi (possibly one more on each side for equispaced radii, which
# also covers the deviation τ ≤ 1e-3 from the grid)
@inline function krange(ri::EquispacedRadii{T}, radii, lo::T, hi::T) where {T}
    n = ri.n
    xlo = _fclamp((lo - ri.r1) * ri.invΔ, T(-2), T(n + 2))
    xhi = _fclamp((hi - ri.r1) * ri.invΔ, T(-2), T(n + 2))
    kmin = max(1, unsafe_trunc(Int, floor(xlo)))          # ≈ ⌈xlo⌉ + 1 - margin
    kmax = min(n, unsafe_trunc(Int, floor(xhi)) + 2)      # ≈ ⌊xhi⌋ + 1 + margin
    return kmin, kmax
end
@inline krange(::SortedRadii, radii, lo, hi) = (searchsortedfirst(radii, lo), searchsortedlast(radii, hi))

# --- per-direction setup ------------------------------------------------------------------------

# projections of the voxel centres per axis, c = (j - (n+1)/2) h, as the original code
function axis_projections!(proj::NTuple{D,Vector{T}}, n::NTuple{D,Int}, h::NTuple{D,T}, θ::NTuple{D,T}) where {D,T}
    for i in 1:D
        pr = proj[i]
        @inbounds for j in 1:n[i]
            pr[j] = ((T(j) - (T(n[i]) + 1) / 2) * h[i]) * θ[i]
        end
    end
    return proj
end
axis_projections(n::NTuple{D,Int}, h::NTuple{D,T}, θ::NTuple{D,T}) where {D,T} =
    axis_projections!(ntuple(i -> Vector{T}(undef, n[i]), D), n, h, θ)

_direction(angles::AbstractMatrix{T}, ℓ, ::Val{D}) where {T,D} = ntuple(i -> angles[ℓ, i], D)

# --- inner loops (one direction, all voxels with the last index in `lastrange`) ---------------------

# p = ((proj[1][j_1] + proj[2][j_2]) + …): the summation order of a dot product
function _psum(D)
    ex = :(proj[1][j_1])
    for d in 2:D
        ex = :($ex + proj[$d][$(Symbol(:j_, d))])
    end
    return ex
end

@generated function _forward_angle!(col::AbstractVector{T}, f::AbstractArray{T,D}, proj::NTuple{D,Vector{T}},
                                    radii::AbstractVector{T}, ri, K, lastrange::AbstractUnitRange) where {T,D}
    psum = _psum(D)
    quote
        S = support(K)
        @inbounds @nloops $D j d -> (d == $D ? lastrange : axes(f, d)) begin
            v = @nref $D f j
            if !iszero(v)
                p = $psum
                kmin, kmax = krange(ri, radii, p - S, p + S)
                for k in kmin:kmax
                    col[k] = muladd(v, K(p - radii[k]), col[k])
                end
            end
        end
        return col
    end
end

# Equispaced radii: each voxel updates a window of W·nb consecutive radii covering its support,
# branch free, so the kernel is evaluated in SIMD (the kernel vanishes on the extra radii). The box
# kernel (axis directions) gets a margin of m = 1 radius on both sides, since it has the value 1/2
# at its jumps; the other kernels are continuous and vanish at the end of the support. The column is padded by `off` radii on both sides (radii extrapolated linearly), so that the
# window needs no bounds checks; voxels outside the radii range only write into the padding.
# Consecutive voxels along the first axis write into U different copies of the column (`colp[:, u]`),
# which keeps the store → load dependencies of overlapping windows apart.
@generated function _forward_angle_win!(colp::AbstractMatrix{T}, f::AbstractArray{T,D}, proj::NTuple{D,Vector{T}},
                                        radp::Vector{T}, ri::EquispacedRadii{T}, K, lastrange::AbstractUnitRange,
                                        ::Val{W}, nb::Int, off::Int, m::Int) where {T,D,W}
    psum = _psum(D)
    quote
        S = support(K)
        U = size(colp, 2)
        n = ri.n
        lo, hi = T(-off + 1), T(n + off - W * nb)
        @inbounds @nloops $D j d -> (d == $D ? lastrange : axes(f, d)) begin
            v = @nref $D f j
            if !iszero(v)
                p = $psum
                x = _fclamp((p - S - ri.r1) * ri.invΔ - ri.τ, lo, hi)   # first radius ≈ x + 1
                base = unsafe_trunc(Int, floor(x)) + off + 1 - m
                u = (j_1 & (U - 1)) + 1
                for blk in 0:nb-1
                    _window_add!(colp, u, base + blk * W, radp, v, p, K, Val(W))
                end
            end
        end
        return colp
    end
end

# the W lanes of a window written out (no loop, no closure): the W kernel evaluations are
# vectorized by the SLP vectorizer
@generated function _window_add!(colp, u, b0, radp, v, p, K, ::Val{W}) where {W}
    quote
        Base.@_inline_meta
        @inbounds begin
            Base.Cartesian.@nexprs $W i -> (a_i = K(p - radp[b0+i-1]))
            Base.Cartesian.@nexprs $W i -> (colp[b0+i-1, u] = muladd(v, a_i, colp[b0+i-1, u]))
        end
        return nothing
    end
end

@generated function _window_dot(gp, b0, radp, p, K, ::Val{W}) where {W}
    quote
        Base.@_inline_meta
        @inbounds begin
            Base.Cartesian.@nexprs $W i -> (a_i = gp[b0+i-1] * K(p - radp[b0+i-1]))
            return Base.Cartesian.@ncall $W (+) a
        end
    end
end

# window length W ∈ (4, 8, 16), number of blocks and margin for the kernel K and the radii: the radii
# in [p - S, p + S] are among ⌈2S/Δ + 2τ⌉ + 1 consecutive ones from ⌊(p - S - r₁)/Δ - τ⌋ + 2
function _window(K, ri::EquispacedRadii)
    m = length(K.b) == 1 ? 1 : 0      # box kernel (axis direction): value 1/2 at the jumps
    need = ceil(Int, 2support(K) * ri.invΔ + 2ri.τ) + 1 + 2m
    W = need <= 4 ? 4 : need <= 8 ? 8 : 16
    return W, cld(need, W), m
end

function _padded_radii(radii::AbstractVector{T}, ri::EquispacedRadii{T}, off::Int) where {T}
    Δ = 1 / ri.invΔ
    radp = [ri.r1 + (k - 1) * Δ for k in 1-off:ri.n+off]
    radp[off+1:off+ri.n] .= radii
    return radp
end

const NBUF = 4     # column copies (power of 2)

# one direction, voxels with the last index in `lastrange`, added to `col`; `buf`: the padded column
# copies of the task (equispaced radii), `off`: the padding of the call
# W ∈ (4, 8, 16) as a compile-time constant without dynamic dispatch
@inline _withW(fn, W) = W == 4 ? fn(Val(4)) : W == 8 ? fn(Val(8)) : fn(Val(16))

function _forward_dir!(col, buf, f, proj, radii, radp, ri::EquispacedRadii, K, lastrange, off)
    fill!(buf, 0)
    W, nb, m = _window(K, ri)
    _withW(W) do w
        _forward_angle_win!(buf, f, proj, radp, ri, K, lastrange, w, nb, off, m)
    end
    @inbounds for k in eachindex(col)
        acc = col[k]
        for u in 1:NBUF
            acc += buf[k+off, u]
        end
        col[k] = acc
    end
    return col
end
_forward_dir!(col, buf, f, proj, radii, radp, ri::SortedRadii, K, lastrange, off) =
    _forward_angle!(col, f, proj, radii, ri, K, lastrange)

_buffer(::EquispacedRadii, ::Type{T}, radp) where {T} = zeros(T, length(radp), NBUF)
_buffer(::SortedRadii, ::Type{T}, radp) where {T} = nothing

@generated function _adjoint_angle!(out::AbstractArray{T,D}, g::AbstractVector{T}, proj::NTuple{D,Vector{T}},
                                    radii::AbstractVector{T}, ri, K, lastrange::AbstractUnitRange) where {T,D}
    psum = _psum(D)
    quote
        S = support(K)
        @inbounds @nloops $D j d -> (d == $D ? lastrange : axes(out, d)) begin
            p = $psum
            kmin, kmax = krange(ri, radii, p - S, p + S)
            acc = zero(T)
            for k in kmin:kmax
                acc = muladd(g[k], K(p - radii[k]), acc)
            end
            (@nref $D out j) += acc
        end
        return out
    end
end

# adjoint with the window of `_forward_angle_win!`: `gp` is the data of one direction, zero padded
@generated function _adjoint_angle_win!(out::AbstractArray{T,D}, gp::AbstractVector{T}, proj::NTuple{D,Vector{T}},
                                        radp::Vector{T}, ri::EquispacedRadii{T}, K, lastrange::AbstractUnitRange,
                                        ::Val{W}, nb::Int, off::Int, m::Int) where {T,D,W}
    psum = _psum(D)
    quote
        S = support(K)
        n = ri.n
        lo, hi = T(-off + 1), T(n + off - W * nb)
        @inbounds @nloops $D j d -> (d == $D ? lastrange : axes(out, d)) begin
            p = $psum
            x = _fclamp((p - S - ri.r1) * ri.invΔ - ri.τ, lo, hi)
            base = unsafe_trunc(Int, floor(x)) + off + 1 - m
            acc = zero(T)
            for blk in 0:nb-1
                acc += _window_dot(gp, base + blk * W, radp, p, K, Val(W))
            end
            (@nref $D out j) += acc
        end
        return out
    end
end

# one direction of the adjoint; `gcol`: the data of the direction (zero padded by `off` for
# equispaced radii)
function _adjoint_dir!(out, gcol, proj, radii, radp, ri::EquispacedRadii, K, lastrange, off)
    W, nb, m = _window(K, ri)
    return _withW(W) do w
        _adjoint_angle_win!(out, gcol, proj, radp, ri, K, lastrange, w, nb, off, m)
    end
end

# directions grouped by the type of their kernel: (kernels of one type, their directions)
function _groups(kernels)
    idx = Dict{Any,Vector{Int}}()
    for (ℓ, K) in enumerate(kernels)
        push!(get!(idx, typeof(K), Int[]), ℓ)
    end
    return [([kernels[ℓ] for ℓ in v], v) for v in values(idx)]   # concrete element types
end

# the adjoint for one group of directions (type-stable loop over the directions)
function _adjoint_group!(out, gp, proj, n::NTuple{D,Int}, h, angles, radii, radp, ri, ks, idx, part, off) where {D}
    for (K, ℓ) in zip(ks, idx)
        axis_projections!(proj, n, h, _direction(angles, ℓ, Val(D)))
        _adjoint_dir!(out, view(gp, :, ℓ), proj, radii, radp, ri, K, part, off)
    end
    return out
end
_adjoint_dir!(out, gcol, proj, radii, radp, ri::SortedRadii, K, lastrange, off) =
    _adjoint_angle!(out, gcol, proj, radii, ri, K, lastrange)

# --- drivers --------------------------------------------------------------------------------------

# The radii may be in any order: the computation runs on the sorted radii, the rows of the result
# are in the order of R.radii. `perm` is empty if the radii are sorted (computed per call, the plan
# stores only the geometry).
function _radii_perm(R::RadonTransform)
    any(isnan, R.radii) && throw(ArgumentError("NaN in the radii"))
    return issorted(R.radii) ? Int[] : sortperm(R.radii)
end

# sorted radii, angles, width and threshold in the precision T
function _geometry(R::RadonTransform, ::Type{T}, D) where {T}
    size(R.angles, 2) == D || throw(DimensionMismatch("the angles have $(size(R.angles, 2)) components, the phantom $D dimensions"))
    R.width >= 0 || throw(ArgumentError("negative width"))
    perm = _radii_perm(R)
    radii = convert(Vector{T}, isempty(perm) ? R.radii : R.radii[perm])
    return radii, convert(Matrix{T}, R.angles), T(R.width), T(R.reduction_threshold)
end

# voxel size as a tuple
function _voxel_size(P::Phantom)
    D = ndims(P.data)
    length(P.hypervoxel_size) == D || throw(DimensionMismatch("need one voxel size per dimension"))
    return ntuple(i -> P.hypervoxel_size[i], Val(D))
end

# the transform of a RestrictedRadonTransform: one direction
_plan(Rθ::RestrictedRadonTransform) =
    RadonTransform(Rθ.radii, reshape(Rθ.angle, 1, :); width = Rθ.width, reduction_threshold = Rθ.reduction_threshold)

# Per call: the kernels of all directions, the index structure of the radii, one padding `off` for
# all directions (the largest window) and the padded radii. Nothing of this is stored in the plan.
function _setup(R::RadonTransform, ::Type{T}, h::NTuple{D,T}; generic::Bool = false) where {T,D}
    radii, angles, width, thr = _geometry(R, T, D)
    a = map(x -> x / 2, h)
    kernels = [cube_kernel(a, _direction(angles, ℓ, Val(D)); width, threshold = thr, generic) for ℓ in axes(angles, 1)]
    ri = radii_index(radii)
    off, radp = _padding(ri, radii, kernels)
    return radii, angles, kernels, ri, off, radp
end
_padding(::SortedRadii, radii, kernels) = (0, radii)
function _padding(ri::EquispacedRadii, radii, kernels)
    off = maximum(K -> prod(_window(K, ri)[1:2]), kernels; init = 0) + 2
    return off, _padded_radii(radii, ri, off)
end

# rows of data on the sorted radii ↦ rows in the order of R.radii, and back
function _unsort(R::RadonTransform, S)
    perm = _radii_perm(R)
    return isempty(perm) ? S : S[invperm(perm), :]
end
function _sort(R::RadonTransform, g)
    perm = _radii_perm(R)
    return isempty(perm) ? g : g[perm, :]
end

# n contiguous parts of r
function _split(r::AbstractUnitRange, n::Int)
    n = max(1, min(n, length(r)))
    q, m = divrem(length(r), n)
    starts = cumsum([first(r); [q + (i <= m) for i in 1:n]])
    return [starts[i]:starts[i+1]-1 for i in 1:n]
end

"""
    radon(P::Phantom, R::RadonTransform; generic=false) -> Matrix (radii × angles)
    radon(P::Phantom, Rθ::RestrictedRadonTransform; generic=false) -> Vector (radii)

The data of `R(P)` (`Rθ(P)`), computed with the fast implementation in the precision of `P`
(`Float32` or `Float64`). The radii may be in any order. `generic = true`: the voxel kernels are
always evaluated by the general recursion instead of the closed forms for ≤ 5 boxes (see
[`cube_kernel`](@ref)).
"""
radon(P::Phantom, R::RadonTransform; generic::Bool = false) = _radon(P.data, _voxel_size(P), R; generic)
radon(P::Phantom, Rθ::RestrictedRadonTransform; generic::Bool = false) = vec(radon(P, _plan(Rθ); generic))

function _radon(f::AbstractArray{T,D}, h::NTuple{D,T}, R::RadonTransform; generic::Bool = false) where {T,D}
    n = size(f)
    radii, angles, kernels, ri, off, radp = _setup(R, T, h; generic)
    M, nr = size(angles, 1), length(radii)
    S = zeros(T, nr, M)
    nt = Threads.nthreads()
    if M >= nt
        # chunks of directions; each task allocates its buffers once
        @sync for chunk in _split(1:M, min(M, 4nt))
            Threads.@spawn begin
                proj = ntuple(i -> Vector{T}(undef, n[i]), D)
                buf = _buffer(ri, T, radp)
                for ℓ in chunk
                    axis_projections!(proj, n, h, _direction(angles, ℓ, Val(D)))
                    _forward_dir!(view(S, :, ℓ), buf, f, proj, radii, radp, ri, kernels[ℓ], axes(f, D), off)
                end
            end
        end
    else
        # few directions: split the voxels (last index), partial sums per part
        parts = _split(axes(f, D), nt)
        partial = zeros(T, nr, M, length(parts))
        @sync for (q, part) in enumerate(parts)
            Threads.@spawn begin
                proj = ntuple(i -> Vector{T}(undef, n[i]), D)
                buf = _buffer(ri, T, radp)
                for ℓ in 1:M
                    axis_projections!(proj, n, h, _direction(angles, ℓ, Val(D)))
                    _forward_dir!(view(partial, :, ℓ, q), buf, f, proj, radii, radp, ri, kernels[ℓ], part, off)
                end
            end
        end
        sum!(S, partial)
    end
    return _unsort(R, S)
end

"""
    backproject(R::RadonTransform, g, n, hypervoxel_size; generic=false) -> Array of size n
    backproject(R::RadonTransform, g, P::Phantom; generic=false)

Adjoint of `f ↦ radon(Phantom(f, hypervoxel_size), R)` applied to the data `g` (radii × angles),
computed in the precision of `g`: `out[j] = Σ_{θ,t} g[t,θ] A_θ(t - ⟨c_j,θ⟩)`. `generic` as in
[`radon`](@ref).
"""
function backproject(R::RadonTransform, g::AbstractMatrix{T}, n::NTuple{D,Int}, hypervoxel_size;
                     generic::Bool = false) where {T<:AbstractFloat,D}
    h = ntuple(i -> T(hypervoxel_size[i]), D)
    radii, angles, kernels, ri, off, radp = _setup(R, T, h; generic)
    size(g) == (length(radii), size(angles, 1)) || throw(DimensionMismatch("data must be radii × angles"))
    gp = _padded_data(ri, _sort(R, g), off)
    out = zeros(T, n)
    groups = _groups(kernels)
    # parts of the voxels (last index), each task loops over all directions
    @sync for part in _split(1:n[D], 4 * Threads.nthreads())
        Threads.@spawn begin
            proj = ntuple(i -> Vector{T}(undef, n[i]), D)
            for (ks, idx) in groups
                _adjoint_group!(out, gp, proj, n, h, angles, radii, radp, ri, ks, idx, part, off)
            end
        end
    end
    return out
end

# data with `off` zero rows above and below (equispaced radii), as is otherwise
function _padded_data(::EquispacedRadii, g::AbstractMatrix{T}, off) where {T}
    gp = zeros(T, size(g, 1) + 2off, size(g, 2))
    gp[off+1:off+size(g, 1), :] .= g
    return gp
end
_padded_data(::SortedRadii, g, off) = g

backproject(R::RadonTransform, g::AbstractMatrix, P::Phantom; kwargs...) =
    backproject(R, g, size(P.data), P.hypervoxel_size; kwargs...)
backproject(R::RadonTransform, S::Sinogram, args...; kwargs...) = backproject(R, S.data, args...; kwargs...)
