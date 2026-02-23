### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 8cbe0300-edff-11ef-2fad-d3b8cca171a9
begin
	import Pkg
	Pkg.activate("..")  
	using Revise
	# using XdRadonTransform
	# using XdRadonTransform.DataTransformations
	# using XdRadonTransform.TestImages
	using Images
	using Plots
	using Interpolations
	using Zygote
	using Flux
	using GLMakie
	# plotlyjs()
end

# ╔═╡ 178b1a7f-ceeb-43ba-9006-e2c5fd7f6b05
using Interpolations: LinearInterpolation

# ╔═╡ 651b9009-3456-4d72-908c-dbdb70dabdf7
using LinearAlgebra, Optimisers, Statistics

# ╔═╡ 22a41f28-207b-4b3b-9d35-7ed4248570b4
md"""
# arXiv:xxxx.xxxxx -- Table x, Figure x
This pluto notebook reproduces the numerical visualizations
with tamplates for Figure x from

- Robert Beinert, Jonas Bresch, and Michael Quellmalz
  '*A Discrete Radon Transform Based on the Area of Cube-Plane Intersection*',
  arXiv:xxxx.xxxxx, 2026.
"""

# ╔═╡ 28bde1c0-98d7-4ea3-9bf0-ef6432eeb6ac
md"""
## Free-support implementation 
of the Radon transform to apply the package `Zygote` for autodifferentiation and backpropagation. 
"""

# ╔═╡ 3dc87ab5-d777-4cf0-bca6-59dd6a5dc69d
begin
	struct PhantomBoxes
	    center::Matrix{Float64}
	    width::Matrix{Float64}
		val::Vector{Float64}
	end

	function PhantomBoxes(C::Matrix{Float64})
		n, d = size(C) 
		val = 1/n
		return PhantomBoxes(C, ones(n, d), val)
	end

	function PhantomBoxes(C::Matrix{Float64}, W::Matrix{Float64})
		n, d = size(C) 
		v = prod(W, dims=2)[:,1]
		s = sum(v)
		val = v ./ s
		return PhantomBoxes(C, W, val)
	end
end

# ╔═╡ cc4d0c44-a1bc-40f9-93d4-014320195f94
begin
	# c = [[1.0,0.0], [0.0,1.0]]
	c = [[-1.0,-1.0], [1.0,1.0]]
	C = Matrix(Transpose(hcat(c...)))
	W = 0.25*ones(2,2)
	P = PhantomBoxes(C, W)
end

# ╔═╡ 44ac40ef-eee6-4795-84da-bfc66b236667
PhantomBoxes(C, W)

# ╔═╡ d1c726ad-e592-4035-b2d9-0567bd9caaf9
function w2_1d(proj1, proj2)
    # Ensure the projections are treated as distributions (normalized)
    p1 = sort(proj1)
    p2 = sort(proj2)
    return mean((p1 .- p2).^2)
end

# ╔═╡ 6a1627db-9773-4ed5-99e1-35c376f41ea3
begin
	function fibonacci_spherical_directions(quantity::Int64)
	    φ = π * (3 - sqrt(5))
		θ = zeros(quantity, 3)
	    for k in 1:quantity
			θ[k, 3] = 1 - 2 * (k - 0.5) / quantity
			θ[k, 2] = sqrt(1 - θ[k, 3]^2) * sin(φ * (k - 1))
	    	θ[k, 1] = sqrt(1 - θ[k, 3]^2) * cos(φ * (k - 1))
		end
		return θ
	end

	function uniform_circle(quantity::Int64)
	    θ = zeros(quantity, 2)
	    for k in 1:quantity
	        θ[k, 1] = cos(2pi * (k - 1) / quantity)
	        θ[k, 2] = sin(2pi * (k - 1) / quantity)
	    end
	    return θ
	end

	function uniform_radii(quantity::Int64, max_radius::Float64=1.0)
	    return collect(LinRange(-max_radius, max_radius, quantity))
	end
end

# ╔═╡ 7c02d625-c8ee-4c63-bcf2-26e13650922c
md"""
## Radon transform
Setup the Radon transform in `3d`
Discretization, given by 

- number of equispaced radii in [-√3⋅d/2,√3⋅d/2],
- Fibonacci sphere points (nearly equispaced on the S^2).
"""

# ╔═╡ d6f3e24d-4bb8-42a3-9ed6-d3b6f59e25c4
begin
	d = maximum(abs.(P.center)) + maximum(abs.(P.width)) + 1
	r = 101
	# radii₃ = collect(LinRange(-sqrt(3)/2*(d),sqrt(3)/2*(d),r))
	radii₂ = uniform_radii(r, d)
	# angles₃ = fibonacci_spherical_directions(12)
	angles₂= uniform_circle(128)
	R₂ = RadonTransform(radii₂,angles₂) # ; width=mean(diff(radii₂))/2)
end

# ╔═╡ 294cc025-99fc-47e3-a380-0e201ba74000
begin
	RP₂ = R₂(P)
	Plots.heatmap(RP₂)
	@info sum(RP₂) * mean(diff(R₂.radii)) / 128
end

# ╔═╡ 2e29a24e-62e5-4387-8449-f260061af59b
md"""
## Mesh-support implementation 
of the Radon transform for the target measure. 
"""

# ╔═╡ b55b38c8-e410-40c2-8743-4a71a8e892fb
begin
	struct PhantomMesh
		diracs::Matrix{Float64}
		val::Float64
	end

	function PhantomMesh(M::Matrix{Float64})
		n = size(M)[1]
		val = 1/n 
		return PhantomMesh(M, val)
	end 

	struct RadonTransformMesh
		radii::Vector{Float64}
		num_radii::Int64
		angles::Matrix{Float64} 
		num_angles::Int64
	    dim::Int64
	end

	function RadonTransformMesh(radii::Vector{Float64}, angles::Matrix{Float64})
		dim = size(angles)[2]
		num_radii = length(radii)
		num_angles = size(angles)[1]
		return RadonTransformMesh(radii, num_radii, angles, num_angles, dim)
	end
	
	function (RMesh::RadonTransformMesh)(M::PhantomMesh)
		S = zeros(Float64, RMesh.num_radii, RMesh.num_angles)
		θ = RMesh.angles
		diff_radii = mean(diff(RMesh.radii)) / 2.0
		for k in axes(S, 2)
			rad = M.diracs * θ[k,:]
			for j in axes(S, 1)
				rⱼ = [abs(RMesh.radii[j] - r) < diff_radii/2.0 for r in rad]
				S[j, k] = sum(rⱼ) * M.val
			end
		end
		return S ./ sum(S)
	end
end

# ╔═╡ 22f2e89b-dc9f-406d-8c9b-32a80017e9ab
begin
	M = zeros(40,2)
	M[1:20,:] = 0.5.*rand(20,2) .+ 0.5.*ones(20,2)
	M[21:40,:] = 0.5.*rand(20,2) .- 1.0.*ones(20,2)
	PMesh = PhantomMesh(M)
end

# ╔═╡ a75c83b6-5333-4d3e-8571-4b56ea350dbf
Plots.scatter(M[:,1], M[:,2])

# ╔═╡ 57fe5a40-2d4f-4c26-affd-25549e56fdb2
begin
	RMesh₂ = RadonTransformMesh(radii₂,angles₂)
end

# ╔═╡ 12aa7a1a-6372-4af7-af9b-2c278979e190
RM₂ = RMesh₂(PMesh)

# ╔═╡ f84039fc-fb75-4eec-9309-7cc637b87046
Plots.heatmap(RM₂)

# ╔═╡ 2fc6c3cc-c518-4694-bc51-6cbbb832586a
md"""
## Setup 
for the sliced Wasserstein calculation.
"""

# ╔═╡ b3b24a2d-2a17-46da-8769-54db5c5b72da
begin
	# Parameters
	epochs = 100
	lr = 0.05
	
	# Pre-calculate targets
	# ct = [[1.0,0.0], [0.0,1.0]]
	# ct = [[-0.75,-0.75], [-0.75,0.75], [0.75,-0.75], [0.75,0.75]]
	# ct = [[0.75,-0.75], [-0.75,0.75]]
	ct = [[-0.25,-0.25], [0.25,0.25]]
	CC = Matrix(Transpose(hcat(ct...)))
	WW = 0.1*ones(2,2)
	sino_data = R₂(PhantomBoxes(CC, WW))
	sino_target = RMesh₂(PMesh)
	
	# Initialize Barycenter
	opt = Optimisers.Adam(
	    lr,                # learning rate
	    (0.9, 0.99),     # less smoothing on variance
	    # 1e-12# stronger numerical stability
	)
	opt_state = Optimisers.setup(opt, (CC, WW))
end

# ╔═╡ d4c130c2-9aa2-4599-935d-8e0d161ad23a
p1 = Plots.heatmap(sino_data*mean(diff(R₂.radii)), fontfamily="Computer Modern")

# ╔═╡ b489e6b0-0021-45ab-aa26-b88cc9847d61
# ╠═╡ disabled = true
#=╠═╡
savefig(p1, "sino_0_2d.pdf")
  ╠═╡ =#

# ╔═╡ 4276d6ac-aa78-4755-831d-8cf3c5828dbe
p2 = Plots.heatmap(sino_target*100, fontfamily="Computer Modern")

# ╔═╡ 4b560692-c941-4e34-b40c-5ef0d1310be3
# ╠═╡ disabled = true
#=╠═╡
savefig(p2, "sino_target_2d.pdf")
  ╠═╡ =#

# ╔═╡ 9b335ef1-bc07-4b1e-bd0d-661eba2dc39c
# ╠═╡ disabled = true
#=╠═╡
# THE MAIN EXECUTION BLOCK
begin
	history_CC = []
	history_WW = []
	history_loss = []
	push!(history_CC, copy(CC))
	push!(history_WW, copy(WW))
	push!(history_loss, sw_loss_matrix(sino_data*mean(diff(R₂.radii)) / R₂.num_angles, sino_target))
	for i in 1:epochs
		loss, grads = Flux.withgradient(CC, WW) do cc, ww
			# Forward pass through your exact kernel logic
			current_sino = R₂(PhantomBoxes(cc, ww))
			return sw_loss_matrix(current_sino*mean(diff(R₂.radii)) / R₂.num_angles, sino_target) 
		end
		
		# Update and constrain
		opt_state, (CC, WW) = Optimisers.update(opt_state, (CC, WW), grads)

		push!(history_CC, copy(CC))
    	push!(history_WW, copy(WW))
		push!(history_loss, copy(loss))
        
        if mod(i, 10) == 0 || i == 1
			# @info grads
			@info CC, WW
            @info "Iteration $i | Loss: $(round(loss, digits=6))"
        end
	end
	CCC, WWW = CC, WW
end
  ╠═╡ =#

# ╔═╡ bc046232-7fe6-491f-942d-641f5eb2d414
#=╠═╡
ploss = Plots.plot(history_loss, 
	fontfamily="Computer Modern", 
	label=L"\textrm{SW}_2^2", 
	# xlabel="Epochs", 
	size=(400,100), 
	yticks=([0.07,0.1,0.13],["0.07", "0.10", "0.13"])
)
  ╠═╡ =#

# ╔═╡ e905e5e7-830e-4983-b2d4-eac21eb7173c
# ╠═╡ disabled = true
#=╠═╡
savefig(ploss, "sclied_wasserstein_loss_2d.pdf")
  ╠═╡ =#

# ╔═╡ 7a66c997-0410-454a-806a-27b3c7244cb6
begin
	function plot_box_2d(C::Vector{Float64}, W::Vector{Float64};c=:red,a=:1.0,style=:solid,l=:false)
		Plots.plot!([C[1] + W[1]/2, C[1] + W[1]/2], [C[2] + W[2]/2, C[2] - W[2]/2], color=c, linewidth=2, alpha=a,style=style,label=l)
		Plots.plot!([C[1] + W[1]/2, C[1] - W[1]/2], [C[2] + W[2]/2, C[2] + W[2]/2], color=c, linewidth=2, alpha=a,style=style,label=false)
		Plots.plot!([C[1] - W[1]/2, C[1] - W[1]/2], [C[2] - W[2]/2, C[2] + W[2]/2], color=c, linewidth=2, alpha=a,style=style,label=false)
		Plots.plot!([C[1] - W[1]/2, C[1] + W[1]/2], [C[2] - W[2]/2, C[2] - W[2]/2], color=c, linewidth=2, alpha=a,style=style,label=false)
	end
end

# ╔═╡ 4bb9f4fd-b9ea-4040-9025-329c1c46fe4e
begin
	# 1. Setup the Opacity Gradient
	# We resample a base map and manually inject the alpha channel
	base_cmap = :amp
	n_steps = 14
	alphas = range(0.5, 1.0, length=n_steps)
	alpha_colors = [RGBAf(c.r, c.g, c.b, a) for (c, a) in zip(Makie.resample_cmap(base_cmap, n_steps), alphas)]

	kkk = 0:5:55
	
	# 2. Create a Figure with ONLY the Colorbar
	fig = Figure(size = (150, 400), figure_padding = 10) # Slim figure for just the bar
	cb = Colorbar(fig[1, 1], 
	limits = (0, 55),             # Change these to your min/max read values
	colormap = alpha_colors, 
	label = L"\text{Epochs}",
	ticklabelsize = 14,
	ticks = (kkk, ["$k" for k in kkk]),
	flipaxis = true,             # Moves labels to the right
	ticklabelfont = "Latin Modern Roman", 
    labelfont = "Latin Modern Roman",
	)
end

# ╔═╡ c549ef85-7886-4869-afbc-fea630ad4e34
#=╠═╡
begin
	# ℓ = argmin(history_loss)
	# @info minimum(history_loss) 
	ℓ = 50
	ℓℓ = 1
	@info history_loss[ℓ]
	kk = [2,3,4,5,10,15,20,25,30,35,40,45,50]
	pbox = Plots.scatter(M[:,1], M[:,2], fontfamily="Computer Modern", label=L"\frac{1}{n}\sum_{i = 1}^n \delta_{x_i}")
	plot_box_2d(history_CC[ℓ][1,:], history_WW[ℓ][1,:], l=L"\sum_{j = 1}^k \mu_j", c=alpha_colors[length(kk)])
	plot_box_2d(history_CC[ℓ][2,:], history_WW[ℓ][2,:],c=alpha_colors[length(kk)])
	plot_box_2d(history_CC[1][1,:], history_WW[1][1,:],c=:gray)
	plot_box_2d(history_CC[1][2,:], history_WW[1][2,:],c=:gray)
	for k in 1:length(kk)
		plot_box_2d(history_CC[kk[k]][1,:], history_WW[kk[k]][1,:];
			style=:dash,
			c=alpha_colors[k],
			#a=2*k*0.01
			)
		plot_box_2d(history_CC[kk[k]][2,:], history_WW[kk[k]][2,:]; 
			style=:dash, 
			c=alpha_colors[k],
			#a=2*k*0.01
			)
		# ℓℓ = ℓℓ+1
	end
end
  ╠═╡ =#

# ╔═╡ d49ddfcf-0737-4c4d-8368-2845b8add1d5
#=╠═╡
pbox
  ╠═╡ =#

# ╔═╡ bad700a3-f6fe-46c0-ba2e-2a93e3926b72
# ╠═╡ disabled = true
#=╠═╡
savefig(pbox, "sampling_cloud_2d.pdf")
  ╠═╡ =#

# ╔═╡ a26a9023-3036-45ba-847b-a24a01728412
#=╠═╡
PO = PhantomBoxes(history_CC[ℓ], history_WW[ℓ])
  ╠═╡ =#

# ╔═╡ 76875428-432c-4075-81dc-20e10c58db4d
fig

# ╔═╡ 75d52061-1659-4b92-a4f4-864f9c0b3afa
#=╠═╡
po = Plots.heatmap(R₂(PO)*mean(diff(R₂.radii)), fontfamily="Computer Modern")
  ╠═╡ =#

# ╔═╡ 4db1528b-1509-47c9-a064-f95a16be6989
# ╠═╡ disabled = true
#=╠═╡
savefig(po, "optimal_cover_2d.pdf")
  ╠═╡ =#

# ╔═╡ da17fdd4-3605-407e-9f8f-17aba9e15ed3
# ╠═╡ disabled = true
#=╠═╡
function (Rθ::RestrictedRadonTransform)(P::PhantomBoxes)
	    # S_buf = zeros(Float64, Rθ.num_radii) #
		S_buf = Zygote.Buffer(zeros(Float64, Rθ.num_radii))
		n = size(P.center)[1]
		v_val = P.val
		for j in 1:n
			w = P.width[j,:]
			# v_center = P.center[j,:]
			A = PlaneRadonKernel(Rθ, w)
			diameter = (norm(w) + Rθ.width) / 2.0
			proj_center = P.center[j,:] ⋅ Rθ.angle

			min_idx, max_idx = Zygote.ignore() do
			    mi = search_min_radius(Rθ.radii, proj_center - diameter)
			    ma = search_max_radius(Rθ.radii, proj_center + diameter)
				return (mi, ma)
			end

			# min_idx = search_min_radius(Rθ.radii, proj_center - diameter)
			# max_idx = search_max_radius(Rθ.radii, proj_center + diameter)
			
			for k in max(1, min_idx):min(Rθ.num_radii, max_idx)
	            τ = proj_center - Rθ.radii[k]
	            
	            # This is the most important check for Buffer stability
	            val = A(τ)
	            if isfinite(val)
					S_buf[k] += v_val[j] * val
				end
	        end
		end
	    return copy(S_buf)
	end
  ╠═╡ =#

# ╔═╡ 868a6dfb-e2f6-41d6-b06a-ee2d5eda1585
#=╠═╡
begin
	#using LinearAlgebra, Zygote, Flux, Optimisers, Statistics
	const It = Base.Iterators
	
	# --- 1. STRUCT DEFINITIONS ---
	struct RadonTransform
	    radii::Vector{Float64}
		num_radii::Int64
		angles::Matrix{Float64}
		num_angles::Int64
	    dim::Int64
		width::Float64
		reduction_threshold::Float64
	end
	RadonTransform(radii, angles; width=0.0, reduction_threshold=1e-15) = 
	    RadonTransform(radii, length(radii), angles, size(angles, 1), size(angles, 2), width, reduction_threshold)
	
	struct RestrictedRadonTransform
	    radii::Vector{Float64} 
		num_radii::Int64
		angle::Vector{Float64}
	    dim::Int64
		width::Float64
		reduction_threshold::Float64
	end
	RestrictedRadonTransform(radii, angle; width=0.0, reduction_threshold=1e-15) = 
	    RestrictedRadonTransform(radii, length(radii), angle, length(angle), width, reduction_threshold)
	
	struct Phantom
	    data::Array{Float64}
		hypervoxel_size::Vector{Float64}
	end
	Phantom(image::AbstractArray) = Phantom(Float64.(image), ones(ndims(image)))
	
	struct Sinogram
	    data::Matrix{Float64} 
		radii::Vector{Float64}
		angles::Matrix{Float64}
	end
	
	# --- 2. KERNEL & GEOMETRY LOGIC ---
	function χ(t::Float64) # Step function
	    t < 0.0 ? 0.0 : (t > 0.0 ? 1.0 : 0.5)
	end
	
	function hypercube_vertex(dim::Int64, index::Int64)
	    # k = zeros(Float64, dim)
	    # for j in 1:dim
		k = map(1:dim) do j
	        (index ÷ 2^(j-1)) % 2 == 0 ? -1.0 : 1.0 
	    end
	    return k
	end

	function redu(v::Vector{Float64}; ϵ::Float64=1e-15)
	    return v[abs.(v) .> ϵ]
	end
	
	struct PlaneRadonKernel
	    scaled_reduced_angle::Vector{Float64}; 
		reduced_dim::Int64; 
		prefactor::Float64
	end
	function PlaneRadonKernel(Rθ::RestrictedRadonTransform, hypervoxel_size::Vector{Float64})
	    a = hypervoxel_size ./ 2.0
	    aθ = redu(a .* Rθ.angle, ϵ=Rθ.reduction_threshold) # 
		# aθ = filter(x -> abs(x) > Rθ.reduction_threshold, a .* Rθ.angle)
	    ℓ = length(aθ)
		f = factorial(max(0, ℓ - 1))
    	denominator = prod(aθ) * f
		if abs(denominator) < 1e-8
		#	@info "H"
			return PlaneRadonKernel(aθ, ℓ, 0.0)
		end
	    prefactor = 2^(Rθ.dim - ℓ) * prod(a) / denominator
    	return PlaneRadonKernel(aθ, ℓ, prefactor)
	end
	#function safe_pow(τ, p)
	#	# τ > 1e-6 ? (p == 0 ? 1.0 : τ^p) : 0.0
	#	p == 0 ? (τ > 0.0 ? 1.0 : (τ < 0.0 ? 0.0 : 0.5)) : (τ > 1e-6 ? τ^p : (τ == 0.0 ? 0.5 : 0.0))
	#end
	# Use a more robust safe_pow that avoids the derivative explosion at 0
	function safe_pow(τ, p)
	    if p == 0
	        # Differentiable approximation of step function or use a hard switch
	        return τ > 0.0 ? 1.0 : (τ < 0.0 ? 0.0 : 0.5)
	    end
	    # Ensure we don't pass negative values to power if p is not an integer
	    # and add a small epsilon to avoid derivative issues at zero.
	    return max(τ, 0.0)^p
	end
	
	function (A::PlaneRadonKernel)(t::Float64)
	    Aa = 0.0
	    # This loop is small (2^dim), so it's usually fine, 
	    # but we should avoid any branching inside.
	    for n in 0:(2^A.reduced_dim - 1)
	        k = hypercube_vertex(A.reduced_dim, n)
	        τ = t + (k ⋅ A.scaled_reduced_angle)
	        Aa += prod(k) * safe_pow(τ, A.reduced_dim - 1)
	    end
	    return A.prefactor * Aa
	end
	
	# --- 3. DIFFERENTIABLE TRANSFORM CALLS ---
	function search_min_radius(radii, value)
	    value < radii[1] ? 1 : searchsortedlast(radii, value) + 1
	end
	function search_max_radius(radii, value)
	    value > radii[end] ? length(radii) : searchsortedfirst(radii, value) - 1
	end

	function (Rθ::RestrictedRadonTransform)(P::PhantomBoxes)
	    n = size(P.center, 1)
	    
	    # Map over each box and return the contribution to the sinogram
	    contributions = map(1:n) do j
	        w = P.width[j, :]
	        v_val = P.val[j]
	        A = PlaneRadonKernel(Rθ, w)
	        proj_center = P.center[j, :] ⋅ Rθ.angle
	        
	        # Vectorized calculation over all radii for this specific box
	        # This replaces the k-loop and the search_min/max logic
	        # which is often cleaner for Zygote.
	        map(Rθ.radii) do r
	            τ = proj_center - r
				vol_j = prod(P.width[j, :])
	            v_val * A(τ) / vol_j
	        end
	    end
	    
	    # Sum the contributions from all boxes
	    return reduce(+, contributions)
	end
	
	function (R::RadonTransform)(P::PhantomBoxes)
	    # Using map instead of @threads for AD compatibility
	    S_cols = map(1:R.num_angles) do ℓ
	        Rθ = RestrictedRadonTransform(R.radii, R.angles[ℓ, :], width=R.width, reduction_threshold=R.reduction_threshold)
	        Rθ(P)
	    end
	    # return Sinogram(reduce(hcat, S_cols), R.radii, R.angles)
		S = reduce(hcat, S_cols)
		# s = sum(S)
		return S
	end
	
	# --- 4. BARYCENTER LOSS & LOOP ---
	function sw_loss(sino_a::Sinogram, sino_b::Sinogram)
	    dist = 0.0
	    for j in 1:size(sino_a.data, 2)
	        # Using sort (not sort!) for AD
	        dist += sum(abs2, sort(sino_a.data[:, j]) .- sort(sino_b.data[:, j]))
	    end
	    return dist
	end

	function sw_loss_m(A, B)
        dist = 0.0
        for j in 1:size(A, 2)
            # sort() is differentiable; sort!() is NOT
            dist += sum(abs2, sort(A[:, j]) .- sort(B[:, j]))
        end
        return dist
    end

	function sw_loss_matrix(A, B; p=1)
	    # A and B are (num_radii x num_angles)
	    # Ensure they are normalized per angle (sum to 1.0)
	    # If they already sum to 1.0, you can skip the division.
	    
	    dist = 0.0
	    num_angles = size(A, 2)
	    
	    for j in 1:num_angles
	        # 1. Compute the CDF for each angle
	        # cumsum is differentiable in Zygote!
	        cdf_A = cumsum(A[:, j])
	        cdf_B = cumsum(B[:, j])
	        
	        # 2. Compute the L1 distance between CDFs (this is W1)
	        # Multiply by dr if your radii are not unit-spaced
	        dist += sum(abs.(cdf_A .- cdf_B))
	    end
	    
	    return dist / num_angles
	end
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╠═22a41f28-207b-4b3b-9d35-7ed4248570b4
# ╠═178b1a7f-ceeb-43ba-9006-e2c5fd7f6b05
# ╠═651b9009-3456-4d72-908c-dbdb70dabdf7
# ╠═28bde1c0-98d7-4ea3-9bf0-ef6432eeb6ac
# ╠═3dc87ab5-d777-4cf0-bca6-59dd6a5dc69d
# ╠═cc4d0c44-a1bc-40f9-93d4-014320195f94
# ╠═44ac40ef-eee6-4795-84da-bfc66b236667
# ╠═d1c726ad-e592-4035-b2d9-0567bd9caaf9
# ╠═868a6dfb-e2f6-41d6-b06a-ee2d5eda1585
# ╠═da17fdd4-3605-407e-9f8f-17aba9e15ed3
# ╠═6a1627db-9773-4ed5-99e1-35c376f41ea3
# ╠═7c02d625-c8ee-4c63-bcf2-26e13650922c
# ╠═d6f3e24d-4bb8-42a3-9ed6-d3b6f59e25c4
# ╠═294cc025-99fc-47e3-a380-0e201ba74000
# ╠═2e29a24e-62e5-4387-8449-f260061af59b
# ╠═b55b38c8-e410-40c2-8743-4a71a8e892fb
# ╠═22f2e89b-dc9f-406d-8c9b-32a80017e9ab
# ╠═a75c83b6-5333-4d3e-8571-4b56ea350dbf
# ╠═57fe5a40-2d4f-4c26-affd-25549e56fdb2
# ╠═12aa7a1a-6372-4af7-af9b-2c278979e190
# ╠═f84039fc-fb75-4eec-9309-7cc637b87046
# ╠═2fc6c3cc-c518-4694-bc51-6cbbb832586a
# ╠═b3b24a2d-2a17-46da-8769-54db5c5b72da
# ╠═d4c130c2-9aa2-4599-935d-8e0d161ad23a
# ╠═b489e6b0-0021-45ab-aa26-b88cc9847d61
# ╠═4276d6ac-aa78-4755-831d-8cf3c5828dbe
# ╠═4b560692-c941-4e34-b40c-5ef0d1310be3
# ╠═9b335ef1-bc07-4b1e-bd0d-661eba2dc39c
# ╠═bc046232-7fe6-491f-942d-641f5eb2d414
# ╠═e905e5e7-830e-4983-b2d4-eac21eb7173c
# ╠═7a66c997-0410-454a-806a-27b3c7244cb6
# ╠═c549ef85-7886-4869-afbc-fea630ad4e34
# ╠═d49ddfcf-0737-4c4d-8368-2845b8add1d5
# ╠═bad700a3-f6fe-46c0-ba2e-2a93e3926b72
# ╠═a26a9023-3036-45ba-847b-a24a01728412
# ╠═4bb9f4fd-b9ea-4040-9025-329c1c46fe4e
# ╠═76875428-432c-4075-81dc-20e10c58db4d
# ╠═75d52061-1659-4b92-a4f4-864f9c0b3afa
# ╠═4db1528b-1509-47c9-a064-f95a16be6989
