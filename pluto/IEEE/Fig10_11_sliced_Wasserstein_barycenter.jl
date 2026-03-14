### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 8cbe0300-edff-11ef-2fad-d3b8cca171a9
begin
	import Pkg
	Pkg.activate("..")  
	using Revise
	using XdRadonTransform
	using XdRadonTransform.DataTransformations
	using XdRadonTransform.TestImages
	using Images
	using Plots
	using Interpolations
	using Zygote
	using Flux
	using GLMakie
	using Augmentor
	#plotlyjs()
end

# ╔═╡ 178b1a7f-ceeb-43ba-9006-e2c5fd7f6b05
using Interpolations: LinearInterpolation

# ╔═╡ 651b9009-3456-4d72-908c-dbdb70dabdf7
using LinearAlgebra, Optimisers, Statistics

# ╔═╡ e1a3e10c-6ae3-45c0-b65d-14e17778a0b9
md"""
# arXiv:xxxx.xxxxx -- Section IV.B and Figure 10 and 11
This pluto notebook reproduces the numerical visualizations
of the target measures from Figure 10 (a) and (b) from

- Robert Beinert, Jonas Bresch, and Michael Quellmalz
  '*A Discrete Radon Transform Based on the Area of Cube-Plane Intersection*',
  arXiv:xxxx.xxxxx, 2026.
"""

# ╔═╡ 46ed69b7-ac95-4eec-924e-cab660b817b8
md"""
## Template Dataset
Generate the template dataset
using the submodule `TestImages`.
"""

# ╔═╡ 60d81917-043b-463a-aa99-79126d7d9ed1
begin
	stroke_width = 1.5;
	blur = 1; #15
end

# ╔═╡ 7c9a3d58-d68a-486b-984e-58e12ee3ce75
begin
	I₁_sharp = render(TestImages.Circle(scale=(1.0, 1.0)), size=(32,32), width=stroke_width);
	I₁ = augment(I₁_sharp, GaussianBlur(blur));
	J₁ = extend_image(I₁, (64, 64))
end

# ╔═╡ 4d61631b-0f41-4363-bce1-ad3a84885718
begin
	I₂_sharp = render(TestImages.Square(scale=(1.2, 1.2)), size=(32,32), width=stroke_width);
	I₂ = augment(I₂_sharp, GaussianBlur(blur));
	J₂ = extend_image(I₂, (64, 64))
end

# ╔═╡ 35b3caf9-f52c-4dc0-a7f4-65226cf5ffd4
begin
	T = [generate_3d_rot(J₁, order=2.0), generate_3d_rot(J₂, order=20.0)];
	T = T./sum.(T)
	Q = [copy(T[1]), copy(T[1])]
	#Q[1][:,32:64,:] .= 0.0
	Q[2][:,:,32:64] .= 0.0
	Q =  Q./sum.(Q)
end

# ╔═╡ 7daac749-b337-4127-ac46-33eb207d6014
Plots.heatmap(Q[1][:,:,31])

# ╔═╡ 46a3f5cf-05e8-424e-89b2-1240aeb7bc3c
begin
	Y1 = []
	Y2 = []
	Y3 = []
	for k in CartesianIndices(Q[1])
		if Q[1][k] ≠ 0.0
			append!(Y1, k.I)
		end
		if Q[2][k] ≠ 0.0
			append!(Y2, k.I)
		end
		if min(Q[1][k], Q[2][k]) ≠ 0.0
			append!(Y3, k.I)
		end
	end
	Y = [transpose(reshape(y, (3,length(y)÷3))) for y in [Y1,Y2,Y3]]
end

# ╔═╡ 7878266d-3c97-4d4e-9e7b-5575eb38ad15
GLMakie.volume(Q[2])

# ╔═╡ e7556a2e-fea8-4350-b9a6-d9377f8b7501
md"""
## Free-support implementation 
of the Radon transform to apply the package `Zygote` for autodifferentiation and backpropagation. 
"""

# ╔═╡ 868a6dfb-e2f6-41d6-b06a-ee2d5eda1585
begin
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
	
	function safe_pow(τ, p)
	    if p == 0
	        # Differentiable approximation of step function or use a hard switch
	        return τ > 0.0 ? 1.0 : (τ < 0.0 ? 0.0 : 0.5)
	    end
	    # Ensure we don't pass negative values to power if p is not an integer
	    # and add a small epsilon to avoid derivative issues at zero.
	    return max(τ, 0.0)^p
	end
	
	# --- 4. BARYCENTER LOSS & LOOP ---
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

# ╔═╡ ef2be742-118a-4378-a933-062d54fd6921
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

# ╔═╡ 75662163-3667-4800-9aae-cff2cf7f0b0c
begin
	#using LinearAlgebra, Zygote, Flux, Optimisers, Statistics
	# const It = Base.Iterators
	
	# --- 1. STRUCT DEFINITIONS ---
	struct RadonTransformFree
	    radii::Vector{Float64}
		num_radii::Int64
		angles::Matrix{Float64}
		num_angles::Int64
	    dim::Int64
		width::Float64
		reduction_threshold::Float64
	end
	RadonTransformFree(radii, angles; width=0.0, reduction_threshold=1e-15) = RadonTransformFree(radii, length(radii), angles, size(angles, 1), size(angles, 2), width, reduction_threshold)
	
	struct RestrictedRadonTransformFree
	    radii::Vector{Float64} 
		num_radii::Int64
		angle::Vector{Float64}
	    dim::Int64
		width::Float64
		reduction_threshold::Float64
	end
	RestrictedRadonTransformFree(radii, angle; width=0.0, reduction_threshold=1e-15) = RestrictedRadonTransformFree(radii, length(radii), angle, length(angle), width, reduction_threshold)
	
	# --- 3. DIFFERENTIABLE TRANSFORM CALLS ---
	struct PlaneRadonKernelFree
	    scaled_reduced_angle::Vector{Float64}; 
		reduced_dim::Int64; 
		prefactor::Float64
	end
	function PlaneRadonKernelFree(Rθ::RestrictedRadonTransformFree, hypervoxel_size::Vector{Float64})
	    a = hypervoxel_size ./ 2.0
	    aθ = redu(a .* Rθ.angle, ϵ=Rθ.reduction_threshold) # 
		# aθ = filter(x -> abs(x) > Rθ.reduction_threshold, a .* Rθ.angle)
	    ℓ = length(aθ)
		f = factorial(max(0, ℓ - 1))
    	denominator = prod(aθ) * f
		if abs(denominator) < 1e-12
			return PlaneRadonKernelFree(aθ, ℓ, 0.0)
		end
	    prefactor = 2^(Rθ.dim - ℓ) * prod(a) / denominator
    	return PlaneRadonKernelFree(aθ, ℓ, prefactor)
	end
	function (A::PlaneRadonKernelFree)(t::Float64)
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
	function (Rθ::RestrictedRadonTransformFree)(P::PhantomBoxes)
	    n = size(P.center, 1)
	    
	    # Map over each box and return the contribution to the sinogram
	    contributions = map(1:n) do j
	        w = P.width[j, :]
	        v_val = P.val[j]
	        A = PlaneRadonKernelFree(Rθ, w)
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
	
	function (R::RadonTransformFree)(P::PhantomBoxes)
	    # Using map instead of @threads for AD compatibility
	    S_cols = map(1:R.num_angles) do ℓ
	        Rθ = RestrictedRadonTransformFree(R.radii, R.angles[ℓ, :], width=R.width, reduction_threshold=R.reduction_threshold)
	        Rθ(P)
	    end
	    # return Sinogram(reduce(hcat, S_cols), R.radii, R.angles)
		S = reduce(hcat, S_cols)
		# s = sum(S)
		return S
	end
end

# ╔═╡ 6a1627db-9773-4ed5-99e1-35c376f41ea3
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

# ╔═╡ 36366919-651e-4944-a57c-b44ea4952911
md"""
## Radon transform
Setup the Radon transform in `3d`
Discretization, given by 

- number of equispaced radii in [-√3⋅d/2,√3⋅d/2],
- Fibonacci sphere points (nearly equispaced on the S^2).
"""

# ╔═╡ d6f3e24d-4bb8-42a3-9ed6-d3b6f59e25c4
begin
	d = 32
	r = 31
	radii = collect(LinRange(-sqrt(3)/2*(d),sqrt(3)/2*(d),r))
	angles = fibonacci_spherical_directions(42)
	R = RadonTransform(radii,angles)
	radiiFree = collect(LinRange(-1,1,r))
	RFree = RadonTransformFree(radiiFree,angles)
end

# ╔═╡ 7d71f754-3d9f-4378-af42-515634c36354
begin
	p₁ = R(Phantom(Q[1]))
	Plots.heatmap(p₁.data)
	@info sum(p₁.data) * mean(diff(R.radii)) / R.num_angles
end

# ╔═╡ b83315c9-ab66-49be-afb6-8358159a14ab
begin
	p₂ = R(Phantom(Q[2]))
	Plots.heatmap(p₂.data)
	@info sum(p₂.data) * mean(diff(R.radii)) / R.num_angles
end

# ╔═╡ c6dae4a3-152c-49cc-8ed5-2234a94eb58c
md"""
## Setup 
for the sliced Wasserstein barycenter calculation.
"""

# ╔═╡ e4695b2d-117a-4546-b6a3-61f9dcf60e36
begin
	# Initialize Barycenter
	#C = 2*rand(100,3) .- ones(100,3)
	C = 1/32*(Y[3] .- 32*ones(size(Y[3])))
	sparse = Int.(ceil.(size(C)[1]*rand(200)))
	C_sparse = C[sparse,:]
	W = 0.02*ones(200,3)
	B_data = PhantomBoxes(C_sparse, W)
	sino12_target = RFree(B_data)
end

# ╔═╡ b3b24a2d-2a17-46da-8769-54db5c5b72da
 begin
	# Parameters
	λ = 0.5
	epochs = 100
	lr = 0.01
	
	# Pre-calculate targets
	sino1_target = R(Phantom(Q[1])).data
	sino2_target = R(Phantom(Q[2])).data

	# B_data = S[1]
	opt = Optimisers.Adam(
	    lr,                # learning rate
	    (0.1, 0.99),     # less smoothing on variance
	    # 1e-12# stronger numerical stability
	)
	opt_state = Optimisers.setup(opt, (C_sparse, W))
end

# ╔═╡ e5e48411-a887-4c12-8bdd-300039dc8a14
Plots.scatter(C_sparse[:,1], C_sparse[:,2], C_sparse[:,3])

# ╔═╡ f155a4a0-6db6-4529-bafe-8e7d2f4cf7b2
Plots.heatmap(sino12_target, fontfamily="Computer Modern")

# ╔═╡ e1fde9eb-5726-4b41-9a10-59a6dd75b7f9
sum(sino12_target)*mean(diff(RFree.radii)) / RFree.num_angles

# ╔═╡ edf94488-9040-49a2-bc4f-d6a8948ecb38
sw_loss_matrix(sino12_target*mean(diff(RFree.radii)) / RFree.num_angles, sino1_target*mean(diff(R.radii)) / R.num_angles) + λ * sw_loss_matrix(sino12_target*mean(diff(RFree.radii)) / RFree.num_angles, sino2_target*mean(diff(R.radii)) / R.num_angles)

# ╔═╡ 31b1424a-d6d8-4d26-b186-856215abd39c
md"""
##### sliced Wasserstein barycenter calculation.
"""

# ╔═╡ 9b335ef1-bc07-4b1e-bd0d-661eba2dc39c
# ╠═╡ disabled = true
#=╠═╡
# THE MAIN EXECUTION BLOCK
begin
	CCC = copy(C_sparse)
	WWW = copy(W)
	history_WW = []
	history_CC = []
	history_loss = []
	push!(history_CC, copy(CCC))
	push!(history_WW, copy(WWW))
	push!(history_loss, copy(sw_loss_matrix(sino12_target*mean(diff(RFree.radii)) / RFree.num_angles, sino1_target*mean(diff(R.radii)) / R.num_angles) + λ * sw_loss_matrix(sino12_target*mean(diff(RFree.radii)) / RFree.num_angles, sino2_target*mean(diff(R.radii)) / R.num_angles)))
	for i in 1:epochs
		loss, grads = Flux.withgradient(CCC, WWW) do cc, ww
			# Forward pass through your exact kernel logic
			current_sino = RFree(PhantomBoxes(cc, ww))
			return (1-λ) * sw_loss_matrix(current_sino*mean(diff(RFree.radii)) / RFree.num_angles, sino1_target*mean(diff(R.radii)) / R.num_angles) + λ * sw_loss_matrix(current_sino*mean(diff(RFree.radii)) / RFree.num_angles, sino2_target*mean(diff(R.radii)) / R.num_angles)
		end

		
		# Update and constrain
		opt_state, (CCC, WWW) = Optimisers.update(opt_state, (CCC, WWW), grads)
		
		if mod(i, 5) == 0 || i == 1
			@info CCC, WWW
            @info "Iteration $i | Loss: $(round(loss, digits=6))"
        end

		push!(history_CC, copy(CCC))
		push!(history_WW, copy(WWW))
		push!(history_loss, copy(loss))
	end
end
  ╠═╡ =#

# ╔═╡ 66a61302-86ef-416d-86a3-61900eb166f6
#=╠═╡
Plots.scatter(history_CC[1][:,1], history_CC[1][:,2], history_CC[1][:,3])
  ╠═╡ =#

# ╔═╡ d1ae55e0-792f-479f-804f-2cd9d48295d2
#=╠═╡
begin
	ℓ = 101
	C_opt = PhantomBoxes(history_CC[ℓ], history_WW[ℓ])
	sino_opt = RFree(C_opt)
	Plots.scatter(history_CC[ℓ][:,1], history_CC[ℓ][:,2], history_CC[ℓ][:,3])
end
  ╠═╡ =#

# ╔═╡ 3f74012e-a9c5-4e36-856b-2ac256b0b802
#=╠═╡
ploss = Plots.plot(history_loss, legend=:false, fontfamily="Computer Modern")
  ╠═╡ =#

# ╔═╡ 3be42c56-0e6c-4911-abde-ae00a6f198bf
# ╠═╡ disabled = true
#=╠═╡
savefig(ploss, "barycenter_3d_loss_cup.pdf")
  ╠═╡ =#

# ╔═╡ 221a4848-6f61-4583-9c7c-84323a5e0b82
#=╠═╡
p3 = Plots.heatmap(RFree(PhantomBoxes(history_CC[100], history_WW[100])), fontfamily="Computer Modern")
  ╠═╡ =#

# ╔═╡ 1bff0929-dc54-43c8-aed5-aca828d776fa
#=╠═╡
savefig(p3, "barycenter_3d_sino_opt_sphere.pdf")
  ╠═╡ =#

# ╔═╡ b698c54a-261e-4fe7-b096-0b71588a1843
p2 = Plots.heatmap(sino2_target, fontfamily="Computer Modern")

# ╔═╡ 14178036-1575-421d-bcf2-531bf629bfc8
p1 = Plots.heatmap(sino1_target, fontfamily="Computer Modern")

# ╔═╡ 815808ab-df23-4386-865b-98d716668329
# ╠═╡ disabled = true
#=╠═╡
savefig(p1, "barycenter_3d_sino_1_sphere.pdf")
  ╠═╡ =#

# ╔═╡ d8bfcdfd-45a2-48b7-a237-094438679013
begin
	function plot_box_3d(C::Vector{Float64}, W::Vector{Float64};c=:gray,a=:1.0,style=:solid,l=:false)
		Plots.plot3d!([C[1] + W[1]/2, C[1] + W[1]/2], [C[2] + W[2]/2, C[2] + W[2]/2], [C[3] + W[3]/2, C[3] - W[3]/2], color=c, linewidth=2, alpha=a,style=style,label=l)
		Plots.plot3d!([C[1] - W[1]/2, C[1] + W[1]/2], [C[2] + W[2]/2, C[2] + W[2]/2], [C[3] + W[3]/2, C[3] + W[3]/2], color=c, linewidth=2, alpha=a,style=style,label=false)
		Plots.plot3d!([C[1] + W[1]/2, C[1] + W[1]/2], [C[2] - W[2]/2, C[2] + W[2]/2], [C[3] + W[3]/2, C[3] + W[3]/2], color=c, linewidth=2, alpha=a,style=style,label=false)
		Plots.plot3d!([C[1] + W[1]/2, C[1] + W[1]/2], [C[2] + W[2]/2, C[2] - W[2]/2], [C[3] - W[3]/2, C[3] - W[3]/2], color=c, linewidth=2, alpha=a,style=style,label=false)
		Plots.plot3d!([C[1] - W[1]/2, C[1] - W[1]/2], [C[2] + W[2]/2, C[2] - W[2]/2], [C[3] - W[3]/2, C[3] - W[3]/2], color=c, linewidth=2, alpha=a,style=style,label=false)
		Plots.plot3d!([C[1] - W[1]/2, C[1] - W[1]/2], [C[2] - W[2]/2, C[2] + W[2]/2], [C[3] + W[3]/2, C[3] + W[3]/2], color=c, linewidth=2, alpha=a,style=style,label=false)
		Plots.plot3d!([C[1] - W[1]/2, C[1] + W[1]/2], [C[2] - W[2]/2, C[2] - W[2]/2], [C[3] - W[3]/2, C[3] - W[3]/2], color=c, linewidth=2, alpha=a,style=style,label=false)
		Plots.plot3d!([C[1] - W[1]/2, C[1] + W[1]/2], [C[2] + W[2]/2, C[2] + W[2]/2], [C[3] - W[3]/2, C[3] - W[3]/2], color=c, linewidth=2, alpha=a,style=style,label=false)
		Plots.plot3d!([C[1] - W[1]/2, C[1] - W[1]/2], [C[2] + W[2]/2, C[2] + W[2]/2], [C[3] + W[3]/2, C[3] - W[3]/2], color=c, linewidth=2, alpha=a,style=style,label=false)
		Plots.plot3d!([C[1] + W[1]/2, C[1] + W[1]/2], [C[2] - W[2]/2, C[2] - W[2]/2], [C[3] - W[3]/2, C[3] + W[3]/2], color=c, linewidth=2, alpha=a,style=style,label=false)
		Plots.plot3d!([C[1] - W[1]/2, C[1] - W[1]/2], [C[2] - W[2]/2, C[2] - W[2]/2], [C[3] - W[3]/2, C[3] + W[3]/2], color=c, linewidth=2, alpha=a,style=style,label=false)
		Plots.plot3d!([C[1] - W[1]/2, C[1] + W[1]/2], [C[2] - W[2]/2, C[2] - W[2]/2], [C[3] + W[3]/2, C[3] + W[3]/2], color=c, linewidth=2, alpha=a,style=style,label=false)
	end
end

# ╔═╡ e54d8251-5340-4277-9be4-59b0a0132367
 pbox

# ╔═╡ 16e95fb6-0ee6-437b-aaef-c9780416ac8c
# ╠═╡ disabled = true
#=╠═╡
savefig(pbox, "barycenter_cups.pdf")
  ╠═╡ =#

# ╔═╡ c76292bf-c77c-4b85-86c0-af4a537f9617
begin
	# Extrahierte Loss-Werte für barycenter_3d_loss_cup.pdf
	# Startet bei ca. 0.096, stabilisiert sich am Ende knapp unter 0.035
	loss_cup = [
	0.0962, 0.0648, 0.0805, 0.0651, 0.0689, 0.0558, 0.0529, 0.0535, 0.0494, 0.0471,
	0.0453, 0.0453, 0.0430, 0.0426, 0.0424, 0.0423, 0.0421, 0.0416, 0.0408, 0.0401,
	0.0394, 0.0403, 0.0392, 0.0401, 0.0405, 0.0398, 0.0395, 0.0396, 0.0392, 0.0390,
	0.0388, 0.0381, 0.0392, 0.0384, 0.0390, 0.0376, 0.0376, 0.0376, 0.0379, 0.0373,
	0.0375, 0.0371, 0.0374, 0.0365, 0.0371, 0.0368, 0.0363, 0.0367, 0.0366, 0.0369,
	0.0362, 0.0367, 0.0365, 0.0361, 0.0366, 0.0359, 0.0362, 0.0357, 0.0358, 0.0362,
	0.0355, 0.0354, 0.0356, 0.0357, 0.0354, 0.0352, 0.0353, 0.0355, 0.0350, 0.0352,
	0.0351, 0.0354, 0.0358, 0.0351, 0.0353, 0.0354, 0.0352, 0.0351, 0.0350, 0.0349,
	0.0348, 0.0349, 0.0347, 0.0351, 0.0345, 0.0348, 0.0349, 0.0344, 0.0349, 0.0351,
	0.0347, 0.0348, 0.0350, 0.0348, 0.0347, 0.0346, 0.0345, 0.0346, 0.0344, 0.0343, 0.0351
	]
	
	# Extrahierte Loss-Werte für barycenter_3d_loss_sphere.pdf
	# Startet bei ca. 0.091, fällt deutlich tiefer auf ca. 0.008
	loss_sphere = [
	0.0806146, 0.04991, 0.0733173, 0.0562752, 0.0474751, 0.0423834, 0.0396918, 0.0368678, 0.0333199, 0.0360692, 0.0323552, 0.0338655, 0.0312367, 0.0314563, 0.0308073, 0.030865, 0.029475, 0.0290299, 0.0295414, 0.0294143, 0.029974, 0.0286704, 0.0285208, 0.0297801, 0.0291053, 0.0281543, 0.0285651, 0.0276501, 0.0277505, 0.0288505, 0.0283665, 0.0274369, 0.0274704, 0.0268603, 0.0275049, 0.0272623, 0.0273045, 0.0273971, 0.0265757, 0.0276971, 0.0270467, 0.0278273, 0.0263136, 0.0274057, 0.0269626, 0.0260413, 0.0269219, 0.0273562, 0.0260661, 0.0266554, 0.0262396, 0.0266529, 0.0260082, 0.0261633, 0.0265784, 0.0259673, 0.0261873, 0.0256744, 0.0254453, 0.0255572, 0.0260333, 0.0254042, 0.0261541, 0.0254174, 0.0256879, 0.0254313, 0.0258094, 0.025667, 0.0255717, 0.0257188, 0.0257716, 0.0255479, 0.0252632, 0.0252618, 0.0254937, 0.0252181, 0.0255067, 0.0254771, 0.0258391, 0.0256907, 0.0253512, 0.02518, 0.0253003, 0.0252466, 0.0253856, 0.0251168, 0.0250904, 0.0254022, 0.0255924, 0.024883, 0.0257003, 0.0255791, 0.0254683, 0.0254937, 0.0252584, 0.0252102, 0.0253801, 0.0250989, 0.0256851, 0.0256169, 0.0246503
	]
end

# ╔═╡ ef29be2f-56f3-44ad-b09f-319b64b48276
begin
	Plots.plot(loss_cup, legend=:false, color=:darkred, yticks=[0.03, 0.06, 0.09])
	p10 = Plots.plot!(loss_sphere, color=:darkblue, fontfamily="Computer Modern", size=(400,100))
end

# ╔═╡ a605bdf1-0421-49ec-8c38-6ab90178d10c
# ╠═╡ disabled = true
#=╠═╡
savefig(p10, "loss_barycenter_cup_sphere.pdf")
  ╠═╡ =#

# ╔═╡ 9ec62156-bc0f-4747-adec-033440e3db76
# ╠═╡ disabled = true
#=╠═╡
begin
	ssparse = Int.(ceil.(size(Y[1])[1]*rand(100)))
	dddim = 1/maximum(sqrt.(sum(Y[1][ssparse,:].^2, dims=2)))
	Plots.scatter3d!(dddim*Y[1][ssparse,1], dddim*Y[1][ssparse,2], dddim*Y[1][ssparse,3], marker=:rect, color=:darkred)
	Plots.scatter3d!(dddim*Y[2][ssparse,1], dddim*Y[2][ssparse,2], dddim*Y[2][ssparse,3], marker=:rect, color=:orange)
end
  ╠═╡ =#

# ╔═╡ a6a2c717-3a16-444e-a3e8-0ee4b0effa48
#=╠═╡
begin
	# ℓ = argmin(history_loss)
	# @info minimum(history_loss) 
	ℓℓ = 100
	ddim = 1/maximum(sqrt.(sum(history_CC[1].^2, dims=2)))
	pbox = Plots.scatter()
	for k in 1:size(history_CC[ℓℓ])[1]
		plot_box_3d(ddim*history_CC[1][k,:], ddim*history_WW[1][k,:])
		plot_box_3d(ddim*history_CC[ℓℓ][k,:], ddim*history_WW[ℓℓ][k,:], c=:darkblue)
	end
	YS1 = Y[1] .- 32*ones(size(Y[1]))
	YS2 = Y[2] .- 32*ones(size(Y[2]))
	ssparse = Int.(ceil.(size(YS2)[1]*rand(200)))
	dddim = 1/maximum(sqrt.(sum(YS1[ssparse,:].^2, dims=2)))
	Plots.scatter3d!(dddim*YS1[2*ssparse,1], dddim*YS1[2*ssparse,2], dddim*YS1[2*ssparse,3], marker=:rect, color=:darkred, alpha=:0.5)
	Plots.scatter3d!(dddim*YS2[ssparse,1], dddim*YS2[ssparse,2], dddim*YS2[ssparse,3], marker=:rect, color=:orange, alpha=:0.5)
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╠═e1a3e10c-6ae3-45c0-b65d-14e17778a0b9
# ╠═178b1a7f-ceeb-43ba-9006-e2c5fd7f6b05
# ╠═651b9009-3456-4d72-908c-dbdb70dabdf7
# ╠═46ed69b7-ac95-4eec-924e-cab660b817b8
# ╠═60d81917-043b-463a-aa99-79126d7d9ed1
# ╠═7c9a3d58-d68a-486b-984e-58e12ee3ce75
# ╠═4d61631b-0f41-4363-bce1-ad3a84885718
# ╠═35b3caf9-f52c-4dc0-a7f4-65226cf5ffd4
# ╠═7daac749-b337-4127-ac46-33eb207d6014
# ╠═46a3f5cf-05e8-424e-89b2-1240aeb7bc3c
# ╠═7878266d-3c97-4d4e-9e7b-5575eb38ad15
# ╠═e7556a2e-fea8-4350-b9a6-d9377f8b7501
# ╠═868a6dfb-e2f6-41d6-b06a-ee2d5eda1585
# ╠═ef2be742-118a-4378-a933-062d54fd6921
# ╠═75662163-3667-4800-9aae-cff2cf7f0b0c
# ╠═6a1627db-9773-4ed5-99e1-35c376f41ea3
# ╠═36366919-651e-4944-a57c-b44ea4952911
# ╠═d6f3e24d-4bb8-42a3-9ed6-d3b6f59e25c4
# ╠═7d71f754-3d9f-4378-af42-515634c36354
# ╠═b83315c9-ab66-49be-afb6-8358159a14ab
# ╠═c6dae4a3-152c-49cc-8ed5-2234a94eb58c
# ╠═e4695b2d-117a-4546-b6a3-61f9dcf60e36
# ╠═b3b24a2d-2a17-46da-8769-54db5c5b72da
# ╠═e5e48411-a887-4c12-8bdd-300039dc8a14
# ╠═f155a4a0-6db6-4529-bafe-8e7d2f4cf7b2
# ╠═e1fde9eb-5726-4b41-9a10-59a6dd75b7f9
# ╠═edf94488-9040-49a2-bc4f-d6a8948ecb38
# ╠═31b1424a-d6d8-4d26-b186-856215abd39c
# ╠═9b335ef1-bc07-4b1e-bd0d-661eba2dc39c
# ╠═66a61302-86ef-416d-86a3-61900eb166f6
# ╠═d1ae55e0-792f-479f-804f-2cd9d48295d2
# ╠═3f74012e-a9c5-4e36-856b-2ac256b0b802
# ╠═3be42c56-0e6c-4911-abde-ae00a6f198bf
# ╠═221a4848-6f61-4583-9c7c-84323a5e0b82
# ╠═1bff0929-dc54-43c8-aed5-aca828d776fa
# ╠═b698c54a-261e-4fe7-b096-0b71588a1843
# ╠═14178036-1575-421d-bcf2-531bf629bfc8
# ╠═815808ab-df23-4386-865b-98d716668329
# ╠═d8bfcdfd-45a2-48b7-a237-094438679013
# ╠═a6a2c717-3a16-444e-a3e8-0ee4b0effa48
# ╠═e54d8251-5340-4277-9be4-59b0a0132367
# ╠═16e95fb6-0ee6-437b-aaef-c9780416ac8c
# ╠═9ec62156-bc0f-4747-adec-033440e3db76
# ╠═c76292bf-c77c-4b85-86c0-af4a537f9617
# ╠═ef29be2f-56f3-44ad-b09f-319b64b48276
# ╠═a605bdf1-0421-49ec-8c38-6ab90178d10c
