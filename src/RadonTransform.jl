using Base.Threads
using LinearAlgebra

struct RadonTransform
    radii::Vector{Float64}
    num_radii::Int64
    angles::Matrix{Float64}
    num_angles::Int64
    dim::Int64
    width::Float64
    reduction_threshold::Float64
end

function RadonTransform(radii::Vector{Float64}, angles::Matrix{Float64}; width::Float64=0.0, reduction_threshold::Float64=1e-15)
    num_radii = length(radii)
    (num_angles, dim) = size(angles)
    return RadonTransform(radii, num_radii, angles, num_angles, dim, width, reduction_threshold)
end

struct RestrictedRadonTransform
    radii::Vector{Float64}
    num_radii::Int64
    angle::Vector{Float64}
    dim::Int64
    width::Float64
    reduction_threshold::Float64
end

function RestrictedRadonTransform(radii::Vector{Float64}, angle::Vector{Float64}; width::Float64=0.0, reduction_threshold::Float64=1e-15)
    num_radii = length(radii)
    dim = length(angle)
    return RestrictedRadonTransform(radii, num_radii, angle, dim, width, reduction_threshold)
end


struct Phantom
    data::Array{Float64}
    hypervoxel_size::Vector{Float64}
end

function Phantom(image::AbstractArray)
    return Phantom(Float64.(image), ones(ndims(image)))
end

function scale(P::Phantom)
    s = norm(P.hypervoxel_size .* size(P.data))
    return Phantom(P.data, 2.0 ./ s .* P.hypervoxel_size)
end

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

function reduce(v::Vector{Float64}; ϵ::Float64=1e-15)
    return v[abs.(v) .> ϵ]
end

struct PlaneRadonKernel
    scaled_reduced_angle::Vector{Float64}
    reduced_dim::Int64
    prefactor::Float64
end

function PlaneRadonKernel(Rθ::RestrictedRadonTransform, hypervoxel_size::Vector{Float64})
    a = hypervoxel_size ./ 2.0
    aθ = reduce(a .* Rθ.angle, ϵ=Rθ.reduction_threshold)
    ℓ = length(aθ)
    d = Rθ.dim
    prefactor = 2^(d - ℓ) * prod(a) / prod(aθ) / factorial(ℓ - 1)
    return PlaneRadonKernel(aθ, ℓ, prefactor)
end

function (A::PlaneRadonKernel)(t::Float64)
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

struct SlabRadonKernel
    scaled_reduced_angle::Vector{Float64}
    reduced_dim::Int64
    prefactor::Float64
    width::Float64
end

function SlabRadonKernel(Rθ::RestrictedRadonTransform, hypervoxel_size::Vector{Float64})
    a = hypervoxel_size ./ 2.0
    aθ = reduce(a .* Rθ.angle, ϵ=Rθ.reduction_threshold)
    ℓ = length(aθ)
    d = Rθ.dim
    prefactor = 2^(d - ℓ) * prod(a) / prod(aθ) / factorial(ℓ) / Rθ.width
    return SlabRadonKernel(aθ, ℓ, prefactor, Rθ.width)
end

function (A::SlabRadonKernel)(t::Float64)
    ℓ = A.reduced_dim
    aθ = A.scaled_reduced_angle
    w = A.width / 2.0
    Aa = 0.0
    for n in 0:2^ℓ - 1 
        k = hypercube_vertex(ℓ, n)
        τ = t + w + (k ⋅ aθ)
        Aa += prod(k) * χ(τ) * τ^ℓ
        τ = t - w + (k ⋅ aθ)
        Aa -= prod(k) * χ(τ) * τ^ℓ
    end
    return A.prefactor * Aa
end

function select_Radon_kernel(Rθ::RestrictedRadonTransform, hypervoxel_size::Vector{Float64})
    if Rθ.width == 0.0
        return PlaneRadonKernel(Rθ, hypervoxel_size)
    else
        return SlabRadonKernel(Rθ, hypervoxel_size)
    end
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
        Rθ = RestrictedRadonTransform(R.radii, R.angles[ℓ, :], width=R.width, reduction_threshold=R.reduction_threshold)
        S[:, ℓ] = Rθ(P)
    end
    return Sinogram(S, R.radii, R.angles)
end

function (Rθ::RestrictedRadonTransform)(P::Phantom)
    S = zeros(Float64, Rθ.num_radii)
    A = select_Radon_kernel(Rθ, P.hypervoxel_size)
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
    diameter = (norm(voxel.size) + Rθ.width) / 2.0
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

function uniform_radii(quantity::Int64, max_radius::Float64=1.0)
    return collect(LinRange(-max_radius, max_radius, quantity))
end

function uniform_circle(quantity::Int64)
    θ = zeros(quantity, 2)
    for k in 1:quantity
        θ[k, 1] = cos(2pi * (k - 1) / quantity)
        θ[k, 2] = sin(2pi * (k - 1) / quantity)
    end
    return θ
end

function fibonacci_sphere(quantity::Int64)
    φ = π * (3 - sqrt(5))
	θ = zeros(quantity, 3)
    for k in 1:quantity
		θ[k, 3] = 1 - 2 * (k - 0.5) / quantity
		θ[k, 2] = sqrt(1 - θ[k, 3]^2) * sin(φ * (k - 1))
    	θ[k, 1] = sqrt(1 - θ[k, 3]^2) * cos(φ * (k - 1))
	end
	return θ
end

function random_spherical_directions(dimension::Int64, quantity::Int64)
    θ = zeros(quantity, dimension)
    for k in 1:quantity
        v = randn(dimension)
        θ[k, :] = v ./ norm(v)
    end
    return θ
end