using Base.Threads
using LinearAlgebra

# struct RadonTransform
#     radii::Vector{Float64}
#     num_radii::Int64
#     angles::Matrix{Float64}
#     num_angles::Int64
#     dim::Int64
#     width::Float64
#     reduction_threshold::Float64
# end

# function RadonTransform(radii::Vector{Float64}, angles::Matrix{Float64}; width::Float64=0.0, reduction_threshold::Float64=1e-15)
#     num_radii = length(radii)
#     (num_angles, dim) = size(angles)
#     return RadonTransform(radii, num_radii, angles, num_angles, dim, width, reduction_threshold)
# end

struct RestrictedJohnTransform
    tangentspace::Array{Vector{Float64}}
    resolution::Tuple
    angle::Vector{Float64}
    dim::Int64
end

function RestrictedJohnTransform(resolution::Tuple, size::Tuple, angle::Vector{Float64})
    tangentspace = tangent_grid(resolution, size, angle)
    dim = length(angle)
    return RestrictedJohnTransform(tangentspace, resolution, angle, dim)
end

struct RestrictedJohnTransformPixel
    point::Vector{Float64}
    index::CartesianIndex
    tangentspace::Array{Vector{Float64}}
    angle::Vector{Float64}
    dim::Int64
end

function RestrictedJohnTransformPixel(Jθ::RestrictedJohnTransform, k::CartesianIndex)
    point = Jθ.tangentspace[k]
    return RestrictedJohnTransformPixel(point, k, Jθ.tangentspace, Jθ.angle, Jθ.dim)
end

struct PhantomBox
    data::Array{Float64}
    hyperbox_size::Tuple{Vector{Float64},Vector{Float64}}
end

function PhantomBox(image::AbstractArray)
    return PhantomBox(Float64.(image), (-ones(ndims(image)), ones(ndims(image))))
end

# function scale(P::PhantomBox)
#     s = norm(P.hypervoxel_size .* size(P.data))
#     return PhantomBox(P.data, 2.0 ./ s .* P.hypervoxel_size)
# end

struct RestrictedSinogram
    data::Array{Float64}
    tangentspace::Array{Vector{Float64}}
    angle::Vector{Float64}
end

struct Hyperbox
    center::Vector{Float64}
    value::Float64
    size::Tuple{Vector{Float64},Vector{Float64}}
end

function Hyperbox(P::PhantomBox, j::CartesianIndex)
    c = (Float64.(j.I) .- (Float64.(size(P.data)) .+ 1) ./ 2) .* (P.hyperbox_size[2] .- P.hyperbox_size[1])./2
    return Hyperbox(c, P.data[j], P.hyperbox_size)
end

struct ProjectedHyperbox
    center::Vector{Float64}
    diameter::Float64
    # support_indices::Vector{CartesianIndex}
end

function project_line(JPθ::RestrictedJohnTransformPixel, box::Hyperbox)
    d = JPθ.dim
    θ = JPθ.angle
    Pθ = Matrix{Float64}(I, d, d) - θ * θ'
    proj_center = Pθ * box.center
    diameter = maximum((box.size[2] .- box.size[1])./2)
    # norm_distance = [maximum(p .- proj_center) < sqrt(d-1)*diameter for p in JPθ.tangentspace]
    # support_index = findall(!iszero, norm_distance)
    return proj_center, diameter # ProjectedHyperbox(proj_center, support_index)
end

struct LineJohnKernel
    angle::Vector{Float64}
    dim::Int64
    hyperbox_a::Vector{Float64}
    hyperbox_b::Vector{Float64}
end

function LineJohnKernel(JPθ::RestrictedJohnTransformPixel, hyperbox_size::Tuple{Vector{Float64},Vector{Float64}})
    θ = JPθ.angle
    d = JPθ.dim
    a = hyperbox_size[1]
    b = hyperbox_size[2]
    return LineJohnKernel(θ, d, a, b)
end

function (L::LineJohnKernel)(x::Array{Float64})
    d = L.dim
    θ = L.angle
    a = L.hyperbox_a
    b = L.hyperbox_b
    t₊ = max.((a .- x)./θ, (b .- x)./θ)
    t₋ = min.((a .- x)./θ, (b .- x)./θ)
    θn = iszero.(θ)
    xt = a .<= x .< b
    xna = x .< a
    xnb = b .<= x
    inft = θn .& xt
    inffa = θn .& xna
    inffb = θn .& xnb
    t₊[inft] .= Inf
    t₊[inffa] .= -Inf
    t₊[inffb] .= -Inf
    t₋[inft] .= -Inf
    t₋[inffa] .= Inf
    t₋[inffb] .= Inf
    ℓ = minimum(t₊) - maximum(t₋)
    if ℓ > 0.0
        return ℓ
    else 
        return 0.0
    end
end

function (Jθ::RestrictedJohnTransform)(P::PhantomBox)
    S = zeros(Float64,  Jθ.resolution)
    index = CartesianIndices(S)
    @threads for k in index
        JPθ = RestrictedJohnTransformPixel(Jθ, k)
        S[k] = JPθ(P)
    end
    return RestrictedSinogram(S, Jθ.tangentspace, Jθ.angle)
end

function (JPθ::RestrictedJohnTransformPixel)(P::PhantomBox)
    x = JPθ.point
    L = LineJohnKernel(JPθ, P.hyperbox_size)
    ℓ = 0.0
    d = JPθ.dim
    for j in CartesianIndices(P.data)
        if P.data[j] ≠ 0.0
            v = Hyperbox(P, j) 
            v_center, diam = project_line(JPθ, v)
            # @info maximum(pv .- x) < diam
            # if JPθ.index in pv.support_indices
            #     ℓ += v.value * L(pv.center .- x)
            # end
            if maximum(x .- v_center) < sqrt(d-1)*diam 
                ℓ += v.value * L(v_center .- x)
            end
        end
    end
    return ℓ
end

##### help-function 

function rotate_e₀_to_theta(θ::Vector{Float64})
    d = length(θ)
    if norm(θ) ≠ 1.0
        θ ./= norm(θ)
    end
    e₀ = zeros(d)
    e₀[end] = 1.0
    c = dot(e₀, θ)
    if c > 1 - 1e-12
        return Matrix{Float64}(I, d, d)
    elseif c < -1 + 1e-12
        R = Matrix{Float64}(I, d, d)
        R[1,1] = -1
        R[end,end] = -1
        return R
    end
    u = θ - c * e₀
    u ./= norm(u)
    s = sqrt(1 - c^2)
    R = Matrix{Float64}(I, d, d)
    R += (c - 1) * (e₀ * e₀' + u * u')
    R += s * (u * e₀' - e₀ * u')
    return R
end

function tangent_grid(n::Tuple,size::Tuple,θ::Vector{Float64})
    d = length(θ)
    d_n = length(n)
    d_s = length(size)
    if d_n ≠ d-1 || d_s ≠ d_n
        @error "wrong dimension in resolution, size, and angle!"
    end
    axes = ntuple(i -> range(-Float64(size[i]), Float64(size[i]); length=n[i]), d-1)
    params = collect(Iterators.product(axes...))
    grid = [vcat(collect(p), 0.0) for p in params]
    R₀ = rotate_e₀_to_theta(θ)
    grid₀ = [R₀ * p for p in grid]
    return grid₀
end