### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 8cbe0300-edff-11ef-2fad-d3b8cca171a9
begin
	import Pkg
	Pkg.activate("../..")
	#Pkg.instantiate()    
	using Revise
	using XdRadonTransform
	using XdRadonTransform.TestImages
	using XdRadonTransform.DataTransformations
	#using XdRadontransform.Classify
	using Images
	using Plots
	#using MLDatasets
	#using Random
	#Random.seed!(42)
	using GLMakie
	using ImagePhantoms
	using Statistics
	using RadonKA
end

# ╔═╡ dde2b58d-f858-4201-8fa8-94165d256b66
I₁ = render(TestImages.Circle(scale=(0.35, 0.35)), size=(50,50), width=13.0);

# ╔═╡ 6c40e0bf-2c24-44f7-8a40-c7f92fe1a3d1
J₁ = extend_image(I₁, (70, 70))

# ╔═╡ ede97db3-9824-4460-a987-7215d5bb30d9
I₂ = render(TestImages.Square(scale=(0.4, 0.4)), size=(50,50), width=13.0);

# ╔═╡ 12ebe477-e5c4-42fa-b835-36914670d63e
J₂ = extend_image(I₂, (70, 70))

# ╔═╡ c77b98a3-57b2-4468-abc2-23ca53fd5882
T₁ = generate_3d_rot(J₁, order=2.0);

# ╔═╡ 1f0eb701-b0a4-47e3-bd67-b8dd97678563
T₂ = generate_3d_rot(J₂, order=20.0);

# ╔═╡ f3d5f267-2e2d-471c-af10-ffe49c07e789
volume(T₁, algorithm=:mip, colormap=:magma)

# ╔═╡ 608f1453-e2aa-4038-a176-d4efdd4ce501
volume(T₂, algorithm=:mip, colormap=:magma)

# ╔═╡ 49330671-af13-416c-8e48-815cf1849d45
R3d = RadonTransform3d(101,15,15)

# ╔═╡ e547b115-e732-48fa-bbe9-a5e6591df570
R₁ = R3d(T₁);

# ╔═╡ ea97c12a-0126-4c49-8f39-385d1e98a1c2
R₂ = R3d(T₂);

# ╔═╡ 2d044d9a-5a3b-4564-b3a8-53c267defffd
volume(R₂, algorithm=:mip, colormap=:magma)

# ╔═╡ beaba48a-f78c-4d99-8a9f-9d615b4889d1
size(R₂)

# ╔═╡ bf6924e8-612c-4117-90c2-47d6133de1a5
Plots.heatmap(R₂[51,:,:], clim=(0,1))

# ╔═╡ e71e0760-b18c-4b25-83ed-a15672b696e6
Plots.heatmap(R₁[:,7,:]) #, clim=(0,1))

# ╔═╡ 67ea086e-600e-4835-908c-b7087d46f011
pp = Plots.heatmap(R₂[:,1,:], clim=(0,1.0))

# ╔═╡ c6d3bbd3-0193-479c-a7bd-fbcb98b6804e
# ╠═╡ disabled = true
#=╠═╡
savefig(pp, "theta_101_15_15_cube_slice_1__n.pdf")
  ╠═╡ =#

# ╔═╡ e26996ab-8686-4f51-9dd2-ba91da4c4fc6
gg = Plots.heatmap(R₁[44,:,:], clim=(0,0.8))

# ╔═╡ c6e23755-ff52-4a39-99e7-9fe07165778c
# ╠═╡ disabled = true
#=╠═╡
savefig(gg, "t_101_15_15_cube_slice_44.pdf")
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╠═dde2b58d-f858-4201-8fa8-94165d256b66
# ╠═6c40e0bf-2c24-44f7-8a40-c7f92fe1a3d1
# ╠═ede97db3-9824-4460-a987-7215d5bb30d9
# ╠═12ebe477-e5c4-42fa-b835-36914670d63e
# ╠═c77b98a3-57b2-4468-abc2-23ca53fd5882
# ╠═1f0eb701-b0a4-47e3-bd67-b8dd97678563
# ╠═f3d5f267-2e2d-471c-af10-ffe49c07e789
# ╠═608f1453-e2aa-4038-a176-d4efdd4ce501
# ╠═49330671-af13-416c-8e48-815cf1849d45
# ╠═e547b115-e732-48fa-bbe9-a5e6591df570
# ╠═ea97c12a-0126-4c49-8f39-385d1e98a1c2
# ╠═2d044d9a-5a3b-4564-b3a8-53c267defffd
# ╠═beaba48a-f78c-4d99-8a9f-9d615b4889d1
# ╠═bf6924e8-612c-4117-90c2-47d6133de1a5
# ╠═e71e0760-b18c-4b25-83ed-a15672b696e6
# ╠═67ea086e-600e-4835-908c-b7087d46f011
# ╠═c6d3bbd3-0193-479c-a7bd-fbcb98b6804e
# ╠═e26996ab-8686-4f51-9dd2-ba91da4c4fc6
# ╠═c6e23755-ff52-4a39-99e7-9fe07165778c
