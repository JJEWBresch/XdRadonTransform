using Base.Threads
using LinearAlgebra

struct RadonTransform
    radii::Vector{Float64}
    num_radii::Int64
    angles::Matrix{Float64}
    num_angles::Int64
    dim::Int64
end

function RadonTransform(radii::Vector{Float64}, angles::Matrix{Float64})
    num_radii = length(radii)
    (num_angles, dim) = size(angles)
    return RadonTransform(radii, num_radii, angles, num_angles, dim)
end

struct RestrictedRadonTransform
    radii::Vector{Float64}
    num_radii::Int64
    angle::Vector{Float64}
    dim::Int64
end

function RestrictedRadonTransform(radii::Vector{Float64}, angle::Vector{Float64})
    num_radii = length(radii)
    dim = length(angle)
    return RestrictedRadonTransform(radii, num_radii, angle, dim)
end


struct Phantom
    data::Array{Float64}
    hypervoxel_size::Vector{Float64}
    # TODO: check dimensions
end

#function Phantom(image::AbstractMatrix)
#    pixel_size = sqrt(2) / max(size(image)...)
#    return Phantom(Float64.(image), Float64(pixel_size), size(image, 2), size(image, 1))  # Phantom(Double64.(image), pixel_size, size(image, 2), size(image, 1))    
#end


struct Sinogram
    data::Matrix{Float64}
    radii::Vector{Float64}
    angles::Matrix{Float64}
end

struct Hypervoxel
    center::Vector{Float64}
    value::Float64
    size::Vector{Float64}
end

function Hypervoxel(P::Phantom, j::CartesianIndex)
    c = (Float64.(j.I) .- (Float64.(size(P.data)) .+ 1) ./ 2) .* P.hypervoxel_size
    return Hypervoxel(c, P.data[j], P.hypervoxel_size)
end

struct ProjectedHypervoxel
    center::Float64
    radii_indices::UnitRange{Int64}
end

struct RadonKernel
    scaled_reduced_angle::Vector{Float64}
    reduced_dim::Int64
    prefactor::Float64
end

function RadonKernel(Rθ::RestrictedRadonTransform, hypervoxel_size::Vector{Float64})
    a = hypervoxel_size ./ 2.0
    aθ = reduce(a .* Rθ.angle)
    ℓ = length(aθ)
    d = Rθ.dim
    prefactor = 2^(d - ℓ) * prod(a) / prod(aθ) / factorial(ℓ - 1)
    return RadonKernel(aθ, ℓ, prefactor)
end

function reduce(v::Vector{Float64})
    return v[abs.(v) .> 1e-15]
    # TODO: constant as parameter
end

function (A::RadonKernel)(t::Float64)
    ℓ = A.reduced_dim
    aθ = A.scaled_reduced_angle
    Aa = 0.0
    for n in 0:2^ℓ - 1 
        k = hypercube_vertex(ℓ, n)
        τ = t + (k ⋅ aθ)
        Aa += prod(k) * χ(τ) * τ^(ℓ - 1)
    end
    return A.prefactor * Aa
end

function χ(t::Float64)
    if t < 0.0
        return 0.0
    elseif t > 0.0
        return 1.0
    else
        return 0.5
    end
end

function hypercube_vertex(dim::Int64, index::Int64)
    k = zeros(Float64, dim)
    for j in 1:dim
        k[j] = (index ÷ 2^(j-1)) % 2 == 0 ? -1.0 : 1.0 
    end
    return k
end


function (R::RadonTransform)(P::Phantom)
    S = zeros(Float64,  R.num_radii, R.num_angles)
    @threads for ℓ in 1:R.num_angles
        Rθ = RestrictedRadonTransform(R.radii, R.angles[ℓ, :])
        S[:, ℓ] = Rθ(P)
    end
    return Sinogram(S, R.radii, R.angles)
end

function (Rθ::RestrictedRadonTransform)(P::Phantom)
    S = zeros(Float64, Rθ.num_radii)
    A = RadonKernel(Rθ, P.hypervoxel_size)
    for j in CartesianIndices(P.data)
        if P.data[j] ≠ 0.0
            v = Hypervoxel(P, j) 
            pv = project(Rθ, v)
            for k in pv.radii_indices
                S[k] += v.value * A(pv.center - Rθ.radii[k])
            end
        end
    end
    return S
end

function project(Rθ::RestrictedRadonTransform, voxel::Hypervoxel)
    center = voxel.center ⋅ Rθ.angle
    diameter = norm(voxel.size) / 2.0
    min_index = search_min_radius(Rθ.radii, center - diameter)
    max_index = search_max_radius(Rθ.radii, center + diameter)
    return ProjectedHypervoxel(center, min_index:max_index)
end

function search_min_radius(radii::Vector{Float64}, value::Float64)
    if value < radii[1]
        return 1
    else
        return searchsortedlast(radii, value) + 1
    end    
end

function search_max_radius(radii::Vector{Float64}, value::Float64)
    if value > radii[end]
        return length(radii)
    else
        return searchsortedfirst(radii, value) - 1
    end    
end
