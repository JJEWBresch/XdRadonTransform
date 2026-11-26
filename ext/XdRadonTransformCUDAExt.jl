# CUDA version of `radon` (and so `R(P)`, `Rθ(P)`) and `backproject` for 2D and 3D phantoms whose
# data is a `CuArray` (prototype), loaded only after `using CUDA`. Same model and kernels as the CPU
# code (`CubeKernel` is isbits and its evaluation is branch free), computed in the precision of the
# data.
#
# Forward: one thread per (radius, direction), gather, no atomics. The thread loops over the voxel
# rows along the dominant axis c of the direction (largest |h_c θ_c|), in which only the
# 1 + 2S/|h_c θ_c| ≤ d + 1 (+ width/|h_c θ_c|) voxels around the plane contribute.
# Adjoint: one thread per voxel, looping over the directions (equispaced radii).
# Directions are grouped by the number of boxes of their kernel, one launch per group.
module XdRadonTransformCUDAExt

using XdRadonTransform
using XdRadonTransform: Phantom, RadonTransform, CubeKernel, cube_kernel, support, _geometry, _direction,
    radii_index, EquispacedRadii, _radii_perm
using CUDA

struct DirInfo{N,D,T}
    K::CubeKernel{N,T}
    θ::NTuple{D,T}
    ℓ::Int32
    c::Int32          # dominant axis
end

function _groups(angles, h::NTuple{D,T}, width, thr) where {D,T}
    a = map(x -> x / 2, h)
    groups = Dict{Int,Vector{Any}}()
    for ℓ in axes(angles, 1)
        θ = _direction(angles, ℓ, Val(D))
        K = cube_kernel(a, θ; width, threshold = thr)
        K isa CubeKernel || throw(ArgumentError("GPU: only kernels with ≤ 4 boxes"))
        c = argmax(map((hi, θi) -> abs(hi * θi), h, θ))
        push!(get!(groups, length(K.b), Any[]), DirInfo(K, θ, Int32(ℓ), Int32(c)))
    end
    return [CuArray([x for x in g]) for g in values(groups)]   # concrete element types
end

# projection of the voxel centre coordinate j along axis i, as `axis_projections`
@inline _pr(j, n, h::T, θ::T) where {T} = ((T(j) - (T(n) + 1) / 2) * h) * θ

# voxel indices j along the axis c with |pc(j) + q - r| ≤ S (with a margin, the kernel vanishes there)
@inline function _jrange(r::T, q::T, S::T, n::Int, h::T, θ::T) where {T}
    δ = h * θ
    u = (r - q) / δ + (T(n) + 1) / 2
    w = S / abs(δ)
    lo = clamp(floor(u - w), T(0), T(n + 1))
    hi = clamp(ceil(u + w), T(0), T(n + 1))
    return max(1, unsafe_trunc(Int, lo)), min(n, unsafe_trunc(Int, hi))
end

function _radon2d!(S, f, radii, infos, h::NTuple{2,T}) where {T}
    k = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    g = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    (k <= length(radii) && g <= length(infos)) || return nothing
    @inbounds begin
        info = infos[g]
        K, θ = info.K, info.θ
        r = radii[k]
        Sup = support(K)
        n1, n2 = size(f)
        acc = zero(T)
        if info.c == 1
            for j2 in 1:n2
                q = _pr(j2, n2, h[2], θ[2])
                jlo, jhi = _jrange(r, q, Sup, n1, h[1], θ[1])
                for j1 in jlo:jhi
                    acc += f[j1, j2] * K(_pr(j1, n1, h[1], θ[1]) + q - r)
                end
            end
        else
            for j1 in 1:n1
                q = _pr(j1, n1, h[1], θ[1])
                jlo, jhi = _jrange(r, q, Sup, n2, h[2], θ[2])
                for j2 in jlo:jhi
                    acc += f[j1, j2] * K(q + _pr(j2, n2, h[2], θ[2]) - r)
                end
            end
        end
        S[k, info.ℓ] = acc
    end
    return nothing
end

function _radon3d!(S, f, radii, infos, h::NTuple{3,T}) where {T}
    k = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    g = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    (k <= length(radii) && g <= length(infos)) || return nothing
    @inbounds begin
        info = infos[g]
        K, θ = info.K, info.θ
        r = radii[k]
        Sup = support(K)
        n1, n2, n3 = size(f)
        acc = zero(T)
        if info.c == 1
            for j3 in 1:n3, j2 in 1:n2
                p23 = _pr(j2, n2, h[2], θ[2])
                p3 = _pr(j3, n3, h[3], θ[3])
                jlo, jhi = _jrange(r, p23 + p3, Sup, n1, h[1], θ[1])
                for j1 in jlo:jhi
                    acc += f[j1, j2, j3] * K((_pr(j1, n1, h[1], θ[1]) + p23) + p3 - r)
                end
            end
        elseif info.c == 2
            for j3 in 1:n3, j1 in 1:n1
                p1 = _pr(j1, n1, h[1], θ[1])
                p3 = _pr(j3, n3, h[3], θ[3])
                jlo, jhi = _jrange(r, p1 + p3, Sup, n2, h[2], θ[2])
                for j2 in jlo:jhi
                    acc += f[j1, j2, j3] * K((p1 + _pr(j2, n2, h[2], θ[2])) + p3 - r)
                end
            end
        else
            for j2 in 1:n2, j1 in 1:n1
                p12 = _pr(j1, n1, h[1], θ[1]) + _pr(j2, n2, h[2], θ[2])
                jlo, jhi = _jrange(r, p12, Sup, n3, h[3], θ[3])
                for j3 in jlo:jhi
                    acc += f[j1, j2, j3] * K(p12 + _pr(j3, n3, h[3], θ[3]) - r)
                end
            end
        end
        S[k, info.ℓ] = acc
    end
    return nothing
end

# `radon(P, R)` (and `R(P)`, `Rθ(P)`) calls `_radon(P.data, voxel size, R)`
function XdRadonTransform._radon(f::CuArray{T,D}, h::NTuple{D,T}, R::RadonTransform) where {T,D}
    D in (2, 3) || throw(ArgumentError("GPU: 2D and 3D only"))
    radii, angles, width, thr = _geometry(R, T, D)
    S = CUDA.zeros(T, length(radii), size(angles, 1))
    rd = CuArray(radii)
    for infos in _groups(angles, h, width, thr)
        threads = (64, 4)
        blocks = (cld(length(radii), threads[1]), cld(length(infos), threads[2]))
        if D == 2
            @cuda threads = threads blocks = blocks _radon2d!(S, f, rd, infos, h)
        else
            @cuda threads = threads blocks = blocks _radon3d!(S, f, rd, infos, h)
        end
    end
    perm = _radii_perm(R)
    return isempty(perm) ? S : S[CuArray(invperm(perm)), :]
end

# adjoint: one thread per voxel, sum over the directions of one group (equispaced radii)
function _backproject!(out, g, radii, infos, h::NTuple{D,T}, r1::T, invΔ::T) where {D,T}
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i <= length(out) || return nothing
    @inbounds begin
        I = CartesianIndices(out)[i]
        n = size(out)
        nr = length(radii)
        acc = zero(T)
        for gi in eachindex(infos)
            info = infos[gi]
            K, θ = info.K, info.θ
            p = _pr(I[1], n[1], h[1], θ[1])
            for d in 2:D
                p += _pr(I[d], n[d], h[d], θ[d])
            end
            Sup = support(K)
            x = (p - Sup - r1) * invΔ
            xh = (p + Sup - r1) * invΔ
            kmin = max(1, unsafe_trunc(Int, floor(clamp(x, T(-2), T(nr + 2)))))
            kmax = min(nr, unsafe_trunc(Int, floor(clamp(xh, T(-2), T(nr + 2)))) + 2)
            for k in kmin:kmax
                acc += g[k, info.ℓ] * K(p - radii[k])
            end
        end
        out[i] += acc
    end
    return nothing
end

function XdRadonTransform.backproject(R::RadonTransform, g::CuArray{T,2}, n::NTuple{D,Int}, hypervoxel_size) where {T<:AbstractFloat,D}
    D in (2, 3) || throw(ArgumentError("GPU: 2D and 3D only"))
    radii, angles, width, thr = _geometry(R, T, D)
    ri = radii_index(radii)
    ri isa EquispacedRadii || throw(ArgumentError("GPU adjoint: equispaced radii only"))
    perm = _radii_perm(R)
    isempty(perm) || (g = g[CuArray(perm), :])
    h = ntuple(i -> T(hypervoxel_size[i]), D)
    out = CUDA.zeros(T, n)
    rd = CuArray(radii)
    for infos in _groups(angles, h, width, thr)
        @cuda threads = 256 blocks = cld(length(out), 256) _backproject!(out, g, rd, infos, h, ri.r1, ri.invΔ)
    end
    return out
end

end # module
