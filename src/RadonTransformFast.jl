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
# with closed forms for G_1, G_2 (piecewise linear/quadratic) and G_3 via the antiderivative K_2 of
# G_2. Every division is by the largest half-width of the boxes involved, so the relative error stays
# O(eps). N ≤ 4 covers the plane and the slab in 2D and 3D; larger N uses the recursion
# `generic_density` (the same idea with closed-form tails, any dimension).

"""
    CubeKernel{N,T}

Radon transform `A(t)` of a voxel for one direction: `scale` times the density of the sum of `N`
independent uniform random variables on `[-b_i, b_i]`, `0 < b_1 ≤ … ≤ b_N` (the convolution of
`N` normalized boxes). Evaluate with `K(t)`.

Built by [`cube_kernel`](@ref). `N ≤ 4` uses closed forms; for `N ≥ 5` see `generic_density`.
"""
struct CubeKernel{N,T<:AbstractFloat}
    b::NTuple{N,T}      # sorted half-widths, all > 0
    scale::T            # voxel volume
    support::T          # Σ b_i: K(t) = 0 for |t| > support
    c::NTuple{10,T}     # precomputed constants, depending on N
end

"""
    cube_kernel(halfsize, θ; width=0, threshold=1e-15) -> CubeKernel

Kernel `t ↦ A_θ^a(t)` (paper, Thm. 1) of the voxel `(-a, a]`, `a = halfsize`, for the direction `θ`,
or for `width > 0` its average over the slab `[t - width/2, t + width/2]` (Cor. 6 divided by the
width). Components with `|a_i θ_i| ≤ threshold` are treated as zero (as `reduce` in the original
code), and also those below `eps(T)` times the largest one.
"""
function cube_kernel(a::NTuple{D,T}, θ::NTuple{D,T}; width::T = zero(T), threshold::T = T(1e-15)) where {D,T<:AbstractFloat}
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
    return _make_kernel(Val(N), Tuple(v)::NTuple{N,T}, scale)
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
    else
        return GenericCubeKernel(b, scale)
    end
    return CubeKernel{N,T}(b, scale, S, _pad10(c, T))
end

"""
    GenericCubeKernel{N,T}

As [`CubeKernel`](@ref) for `N ≥ 5` boxes (dimension ≥ 5, or 4 with a slab): evaluated by the
recursion of `generic_density` with precomputed partial sums and moments.
"""
struct GenericCubeKernel{N,T<:AbstractFloat}
    b::NTuple{N,T}
    scale::T
    support::T
    S::NTuple{N,T}          # partial sums b_1 + … + b_k
    μ::Matrix{T}            # μ[j+1, k] = E[(U_1 + … + U_k)^j]
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

# moments μ[j+1, k] = E[(U_1 + … + U_k)^j], j = 0..N, of the sum of the first k uniforms
function _box_sum_moments(b::NTuple{N,T}) where {N,T}
    μ = zeros(T, N + 1, N)
    prev = [j == 0 ? one(T) : zero(T) for j in 0:N]
    for k in 1:N
        ν = [iseven(i) ? b[k]^i / (i + 1) : zero(T) for i in 0:N]
        μ[:, k] = prev = [sum(binomial(j, i) * prev[j-i+1] * ν[i+1] for i in 0:j) for j in 0:N]
    end
    return μ
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
            terms = [:($(T(binomial(m - 1, j) / (2 * factorial(m - 1)))) * $x^$(m - 1 - j) * μ[$(j + 1), $k]) for j in 0:2:m-1]
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
    m = K isa CubeKernel{1} ? 1 : 0
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
function _setup(R::RadonTransform, ::Type{T}, h::NTuple{D,T}) where {T,D}
    radii, angles, width, thr = _geometry(R, T, D)
    a = map(x -> x / 2, h)
    kernels = [cube_kernel(a, _direction(angles, ℓ, Val(D)); width, threshold = thr) for ℓ in axes(angles, 1)]
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
    radon(P::Phantom, R::RadonTransform) -> Matrix (radii × angles)
    radon(P::Phantom, Rθ::RestrictedRadonTransform) -> Vector (radii)

The data of `R(P)` (`Rθ(P)`), computed with the fast implementation in the precision of `P`
(`Float32` or `Float64`). The radii may be in any order.
"""
radon(P::Phantom, R::RadonTransform) = _radon(P.data, _voxel_size(P), R)
radon(P::Phantom, Rθ::RestrictedRadonTransform) = vec(radon(P, _plan(Rθ)))

function _radon(f::AbstractArray{T,D}, h::NTuple{D,T}, R::RadonTransform) where {T,D}
    n = size(f)
    radii, angles, kernels, ri, off, radp = _setup(R, T, h)
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
    backproject(R::RadonTransform, g, n, hypervoxel_size) -> Array of size n
    backproject(R::RadonTransform, g, P::Phantom)

Adjoint of `f ↦ radon(Phantom(f, hypervoxel_size), R)` applied to the data `g` (radii × angles),
computed in the precision of `g`: `out[j] = Σ_{θ,t} g[t,θ] A_θ(t - ⟨c_j,θ⟩)`.
"""
function backproject(R::RadonTransform, g::AbstractMatrix{T}, n::NTuple{D,Int}, hypervoxel_size) where {T<:AbstractFloat,D}
    h = ntuple(i -> T(hypervoxel_size[i]), D)
    radii, angles, kernels, ri, off, radp = _setup(R, T, h)
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

backproject(R::RadonTransform, g::AbstractMatrix, P::Phantom) = backproject(R, g, size(P.data), P.hypervoxel_size)
backproject(R::RadonTransform, S::Sinogram, args...) = backproject(R, S.data, args...)
