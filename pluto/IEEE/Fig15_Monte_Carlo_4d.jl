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
	using IterTools
	# plotlyjs()
end

# ╔═╡ c20ec051-9ae9-40f7-97cb-7abf2baf4fe4
using FileIO, MeshIO, GLMakie, Flux3D, PlotlyJS # CairoMakie, Flux3D

# ╔═╡ 1fa304a4-ecf5-4645-8f3a-90aedde4323e
using Markdown, InteractiveUtils

# ╔═╡ 178b1a7f-ceeb-43ba-9006-e2c5fd7f6b05
using Interpolations: LinearInterpolation

# ╔═╡ 62aa8854-c7b6-4da4-a904-22d56ed5364c
using Statistics, Random, StatsBase

# ╔═╡ 0ebabfea-d457-4730-9be4-d142f8278898
using LinearAlgebra

# ╔═╡ a1aab451-32cb-4346-84ee-e180058dda75
md"""
# arXiv:xxxx.xxxxx -- Table x, Figure x
This pluto notebook reproduces the numerical visualizations
with tamplates for Figure x from

- Robert Beinert, Jonas Bresch, and Michael Quellmalz
  '*A Discrete Radon Transform Based on the Area of Cube-Plane Intersection*',
  arXiv:xxxx.xxxxx, 2026.
"""

# ╔═╡ 626dc57d-e8f1-4e7a-ace8-4d18ebb76bbe
begin
	# Box = zeros(128,128,128,128);
	# Box[64:65,64:65,64:65,64:65] = ones(2,2,2,2);
	Box = ones(2,2,2,2)
end

# ╔═╡ 9a88aa43-f6b6-458a-939b-a91510bdf571
begin
	PBox = Phantom(Box)
end

# ╔═╡ c4a5b59b-5cc9-477a-9c9c-c195418c5c7d
md"""
## Radon transform
Setup the Radon transform in `4d`
Discretization, given by 

- number of equispaced radii in [-√4⋅d/2,√4⋅d/2],
- Sobol' sphere points (nearly equispaced on the S^3).
"""

# ╔═╡ 8d783205-ffa0-4ac6-957c-706b7505806b
begin
	d = 2
	r = 128
	radii = collect(LinRange(-sqrt(4)/2*(d),sqrt(4)/2*(d),r))
	angles = sobol_spherical_directions(4, 128)
	R = RadonTransform(radii,angles;width=0.01)
end

# ╔═╡ 335f4a75-a720-4433-942f-c18168c06208
begin
	RBox = R(PBox)
	Plots.heatmap(RBox.data)
	RBox.data
end

# ╔═╡ 44ca8090-3794-44b2-83e9-84225c5cde79
begin
	Rr = XdRadonTransform.RestrictedRadonTransform(radii,angles[1,:])
	# Rr = XdRadonTransform.RestrictedRadonTransform(radii,[1.0,0.0,0.0])
end

# ╔═╡ 26c8e445-9060-4001-b47f-74a4d8ea4e61
begin
	RrBox = Rr(PBox)
	Plots.plot(RrBox)
end

# ╔═╡ a739eb0d-4332-4d72-83d9-c08aefea3946
md"""
## Monte-Carlo approximation 
of the 4D Radon transform for the unit voxel [-1,1]^4.

Input: 
- ω : normal vector (will be normalized)
- svals : vector of s-values
- N : number of Monte-Carlo samples
- ε : slab half-width
"""

# ╔═╡ 84a05c40-42d3-4203-8b67-3c9b9e112ec1
function radon_MC(ω, svals; N=7, ε=1e-3)
	ω = ω / norm(ω)

	# draw samples
	X = 2*rand(2^N, 4) .- 1               # uniform in [-1,1]^4
	# X = rand(N, 4)               # uniform in [0,1]^4
	dots = X * ω                 # ⟨ω,x⟩ values

	R = zeros(length(svals))

	for (i, s) in enumerate(svals)
		hits = abs.(dots .- s) .<= ε
		R[i] = mean(hits) / (2ε)
	end

	return 16*R
end

# ╔═╡ 905e0059-98b9-43b2-9a24-ceee36478f8a
md"""
## Deterministic Monte-Carlo approximation 
of the 4D Radon transform for the unit voxel [-1,1]^4.

Input: 
- ω : normal vector (will be normalized)
- svals : vector of s-values
- N : number of Monte-Carlo samples
- ε : slab half-width
"""

# ╔═╡ 11adfc9f-73d4-4cad-86a5-b99a2fc16d34
begin
	function grid4_power(N)
	    @assert N % 4 == 0 "N must be divisible by 4"
	
	    m = 2^(N ÷ 4)                # points per dimension
	    xs = range(-1, 1; length=m)
	
	    X = Array{Float64}(undef, m^4, 4)
	    idx = 1
	    for x1 in xs, x2 in xs, x3 in xs, x4 in xs
	        X[idx, :] .= (x1, x2, x3, x4)
	        idx += 1
	    end
	    X
	end

	function radon_grid(ω, svals; N=4*7, ε=1e-3)
	    ω = ω / norm(ω)
	
	    # draw samples
	    X = grid4_power(N)
	    dots = X * ω                 # ⟨ω,x⟩ values
	
	    R = zeros(length(svals))
	
	    for (i, s) in enumerate(svals)
	        hits = abs.(dots .- s) .<= ε
	        R[i] = mean(hits) / (2ε)
	    end
	
	    return 16*R
	end
end

# ╔═╡ 05261121-56b9-43cd-901e-1fb5de310158
begin
	rrr = radon_grid(angles[1,:], radii; N=24, ε=1e-1)
	Plots.plot(rrr)
end

# ╔═╡ 797d6640-c5ea-4319-8264-6a03ccfd7e06
md"""
## Conditional Monte-Carlo approximation 
of the 4D Radon transform for the unit voxel [-1,1]^4.

Input: 
- ω : normal vector (will be normalized)
- svals : vector of s-values
- N : number of Monte-Carlo samples
"""

# ╔═╡ 9762efb6-93d7-478f-bab6-5040e25fd064
function radon_conditional(ω, svals; N=10)
    d = length(ω)
    # pick coordinate with largest abs weight to avoid small divisor
    j = argmax(abs.(ω))
    wj = ω[j]
    ω_minus = ω[[i for i in 1:d if i != j]]  # length d-1

    R = zeros(length(svals))
    # Pre-sample x_{-j} once and reuse for each s (variance reduction)
    Xminus = 2 .* rand(2^N, d-1) .- 1  # each row is a sample in [-1,1]^{d-1}

    for (k, s) in enumerate(svals)
        # compute t = (s - sum_{i!=j} ω_i x_i) / ω_j  for each row
        # dot each row with ω_minus (vectorized)
        dots = Xminus * ω_minus            # N-vector: sum_{i!=j} ω_i * x_i
        t = (s .- dots) ./ wj
        I = abs.(t) .<= 1.0
        R[k] = (2.0^(d-1) / abs(wj)) * mean(I)  # estimator
    end

    return R
end


# ╔═╡ 555352d0-a8cf-4da5-8078-fb2dc4fdeed4
Plots.heatmap(RBox.data)

# ╔═╡ 88dd0b48-2e3c-4024-9c4a-bc69caad8771
begin
	Rrmc₇ = radon_MC(angles[1,:], radii; N=24, ε=1e-3)
	Rrmc₆ = radon_MC(angles[1,:], radii; N=20, ε=1e-3)
	Rrmc₅ = radon_MC(angles[1,:], radii; N=16, ε=1e-3)
end

# ╔═╡ f4a73a2f-4e0a-4a9a-9fbf-0b8e83b12163
begin
	Rrde₇ = radon_grid(angles[1,:], radii; N=24, ε=1e-1)
	Rrde₆ = radon_grid(angles[1,:], radii; N=20, ε=1e-1)
	Rrde₅ = radon_grid(angles[1,:], radii; N=16, ε=1e-1)
end

# ╔═╡ 7d9bc5a2-16c5-4c21-9b14-a30338747796
begin
	Rrcmc₇ = radon_conditional(angles[1,:], radii; N=12)
	Rrcmc₆ = radon_conditional(angles[1,:], radii; N=10)
	Rrcmc₅ = radon_conditional(angles[1,:], radii; N=8)
end

# ╔═╡ 6c0cb238-5a24-46b6-b90c-4102f4d16a33
begin
	p₁ = Plots.plot(Rrmc₇,
	label = L"\textrm{MC} (N = 2^{24})", color=:darkblue)
	p₁ = Plots.plot!(Rrmc₆,
	label = L"\textrm{MC} (N = 2^{20})", color=:darkgreen)
	p₁ = Plots.plot!(Rrmc₅,
	label = L"\textrm{MC} (N = 2^{16})", color=:orange)
	Plots.plot!(RrBox,
	fontfamily="Computer Modern",
	xticks = ([1,32,64,96,128], round.(radii,digits=4)[[1,32,64,96,128]]),
	label = "Exact", color=:darkred, legend=:false)
end

# ╔═╡ 6752dbfb-cc60-41bc-b4b0-d2cc533ffc57
begin
	p₂ = Plots.plot(Rrde₇,
	label = L"N = 2^{24}", color=:darkblue)
	p₂ = Plots.plot!(Rrde₆,
	label = L"N = 2^{20}", color=:darkgreen)
	p₂ = Plots.plot!(Rrde₅,
	label = L"N = 2^{16}", color=:orange)
	p₂ = Plots.plot!(RrBox,
	fontfamily="Computer Modern",
	xticks = ([1,32,64,96,128], round.(radii,digits=4)[[1,32,64,96,128]]),
	label = "Exact", color=:darkred, legend=:false)
end

# ╔═╡ 0b8da562-d571-4756-92af-0f9bfcc25dba
begin
	p₃ = Plots.plot(Rrcmc₇,
	label = L"\textrm{CMC} (N = 2^{12})")
	p₃ = Plots.plot!(Rrcmc₆,
	label = L"\textrm{CMC} (N = 2^{10})")
	p₃ = Plots.plot!(Rrcmc₅,
	label = L"\textrm{CMC} (N = 2^{8})")
	p₃ = Plots.plot!(RrBox,
	fontfamily="Computer Modern",
	xticks = ([1,32,64,96,128], round.(radii,digits=4)[[1,32,64,96,128]]),
	label = "Exact")
end

# ╔═╡ 6a5c1200-da89-4559-8acc-863150ca0604
# ╠═╡ disabled = true
#=╠═╡
Plots.savefig(p₁, "exact_MC_comp_765_out.pdf")
  ╠═╡ =#

# ╔═╡ ffbd8ef7-eac0-497d-bf69-4e61117223d6
begin
	L₇ = zeros(128)
	L₆ = zeros(128)
	L₅ = zeros(128)
	for ℓ in 1:10
		Rrmc₇ = radon_MC(angles[1,:], radii; N=24, ε=1e-3)
		Rrmc₆ = radon_MC(angles[1,:], radii; N=20, ε=1e-3)
		Rrmc₅ = radon_MC(angles[1,:], radii; N=16, ε=1e-3)
		L₇ += abs.(RrBox .- Rrmc₇)
		L₆ += abs.(RrBox .- Rrmc₆)
		L₅ += abs.(RrBox .- Rrmc₅)
	end
end

# ╔═╡ ed15f017-7d0f-414b-a695-305169351bc8
begin
	Lc₇ = zeros(128)
	Lc₆ = zeros(128)
	Lc₅ = zeros(128)
	for ℓ in 1:10
		Rrcmc₇ = radon_conditional(angles[1,:], radii; N=12)
		Rrcmc₆ = radon_conditional(angles[1,:], radii; N=10)
		Rrcmc₅ = radon_conditional(angles[1,:], radii; N=8)
		Lc₇ += abs.(RrBox .- Rrcmc₇)
		Lc₆ += abs.(RrBox .- Rrcmc₆)
		Lc₅ += abs.(RrBox .- Rrcmc₅)
	end
end

# ╔═╡ 1461c922-665e-4fb0-a0ac-1d729daa0b61
begin
	Rrgmc₇ = radon_grid(angles[1,:], radii; N=24, ε=1e-1)
	Rrgmc₆ = radon_grid(angles[1,:], radii; N=20, ε=1e-1)
	Rrgmc₅ = radon_grid(angles[1,:], radii; N=16, ε=1e-1)
	Lg₇ = abs.(RrBox .- Rrgmc₇)
	Lg₆ = abs.(RrBox .- Rrgmc₆)
	Lg₅ = abs.(RrBox .- Rrgmc₅)
end

# ╔═╡ e624e680-ad8f-4b27-8840-d139d7efa0cc
begin
	Plots.plot(L₇./10,
	label = L"\mathbb{E}|\textrm{Exact} - \textrm{MC}(N = 2^{24})|",
	fontfamily="Computer Modern", legend=:false, color=:darkblue)
	Plots.plot!(L₆./10,
	label = L"\mathbb{E}|\textrm{Exact} - \textrm{MC}(N = 2^{20})|",
	fontfamily="Computer Modern", color=:darkgreen)
	p₄ = Plots.plot!(L₅./10,
	label = L"\mathbb{E}|\textrm{Exact} - \textrm{MC}(N = 2^{16})|",
	fontfamily="Computer Modern", color=:orange,
	xticks = ([1,32,64,96,128], round.(radii,digits=4)[[1,32,64,96,128]]))
end

# ╔═╡ a8a7167f-af5b-4bbc-95ff-3a012e2bd6e2
begin
	Plots.plot(Lc₇./10,
	label = L"\mathbb{E}|\textrm{Exact} - \textrm{CMC}(N = 2^{12})|",
	fontfamily="Computer Modern")
	Plots.plot!(Lc₆./10,
	label = L"\mathbb{E}|\textrm{Exact} - \textrm{CMC}(N = 2^{10})|",
	fontfamily="Computer Modern")
	p₅ = Plots.plot!(Lc₅./10,
	label = L"\mathbb{E}|\textrm{Exact} - \textrm{CMC}(N = 2^8)|",
	fontfamily="Computer Modern",
	xticks = ([1,32,64,96,128], round.(radii,digits=4)[[1,32,64,96,128]]))
end

# ╔═╡ bea49077-a96b-4682-9fcd-ae31abbe42b4
begin
	Plots.plot(Lg₇./10,
	label = L"\mathbb{E}|\textrm{Exact} - \textrm{GMC}(N = 2^{12})|",
	fontfamily="Computer Modern", legend=:false, color=:darkblue)
	Plots.plot!(Lg₆./10,
	label = L"\mathbb{E}|\textrm{Exact} - \textrm{GMC}(N = 2^{10})|",
	fontfamily="Computer Modern", color=:darkgreen)
	p₆ = Plots.plot!(Lg₅./10,
	label = L"\mathbb{E}|\textrm{Exact} - \textrm{GMC}(N = 2^8)|",
	fontfamily="Computer Modern", color=:orange,
	xticks = ([1,32,64,96,128], round.(radii,digits=4)[[1,32,64,96,128]]))
end

# ╔═╡ b6b957b3-a46a-45d3-a240-079e98cad0c6
# ╠═╡ disabled = true
#=╠═╡
Plots.savefig(p₄, "exact_MC_mean_diff_765.pdf")
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╠═a1aab451-32cb-4346-84ee-e180058dda75
# ╠═c20ec051-9ae9-40f7-97cb-7abf2baf4fe4
# ╠═1fa304a4-ecf5-4645-8f3a-90aedde4323e
# ╠═178b1a7f-ceeb-43ba-9006-e2c5fd7f6b05
# ╠═62aa8854-c7b6-4da4-a904-22d56ed5364c
# ╠═626dc57d-e8f1-4e7a-ace8-4d18ebb76bbe
# ╠═9a88aa43-f6b6-458a-939b-a91510bdf571
# ╠═c4a5b59b-5cc9-477a-9c9c-c195418c5c7d
# ╠═8d783205-ffa0-4ac6-957c-706b7505806b
# ╠═335f4a75-a720-4433-942f-c18168c06208
# ╠═44ca8090-3794-44b2-83e9-84225c5cde79
# ╠═26c8e445-9060-4001-b47f-74a4d8ea4e61
# ╠═0ebabfea-d457-4730-9be4-d142f8278898
# ╠═a739eb0d-4332-4d72-83d9-c08aefea3946
# ╠═84a05c40-42d3-4203-8b67-3c9b9e112ec1
# ╠═05261121-56b9-43cd-901e-1fb5de310158
# ╠═905e0059-98b9-43b2-9a24-ceee36478f8a
# ╠═11adfc9f-73d4-4cad-86a5-b99a2fc16d34
# ╠═797d6640-c5ea-4319-8264-6a03ccfd7e06
# ╠═9762efb6-93d7-478f-bab6-5040e25fd064
# ╠═555352d0-a8cf-4da5-8078-fb2dc4fdeed4
# ╠═88dd0b48-2e3c-4024-9c4a-bc69caad8771
# ╠═f4a73a2f-4e0a-4a9a-9fbf-0b8e83b12163
# ╠═7d9bc5a2-16c5-4c21-9b14-a30338747796
# ╠═6c0cb238-5a24-46b6-b90c-4102f4d16a33
# ╠═6752dbfb-cc60-41bc-b4b0-d2cc533ffc57
# ╠═0b8da562-d571-4756-92af-0f9bfcc25dba
# ╠═6a5c1200-da89-4559-8acc-863150ca0604
# ╠═ffbd8ef7-eac0-497d-bf69-4e61117223d6
# ╠═ed15f017-7d0f-414b-a695-305169351bc8
# ╠═1461c922-665e-4fb0-a0ac-1d729daa0b61
# ╠═e624e680-ad8f-4b27-8840-d139d7efa0cc
# ╠═a8a7167f-af5b-4bbc-95ff-3a012e2bd6e2
# ╠═bea49077-a96b-4682-9fcd-ae31abbe42b4
# ╠═b6b957b3-a46a-45d3-a240-079e98cad0c6
