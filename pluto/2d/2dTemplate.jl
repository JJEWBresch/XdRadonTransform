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
	using Random
	#Random.seed!(42)
	#using GLMakie, Makie
end

# ╔═╡ 9bc6bbed-558b-471e-bfdb-62d18897a910
I₁ = render(TestImages.Circle(), width=4.0);

# ╔═╡ d272426c-53c1-46f4-8d39-ab2f635ad683
J₁ = extend_image(I₁, (128, 128))

# ╔═╡ aa47398a-b248-48fd-943e-b349b6e8f6a8
I₁₁ = render(TestImages.OrbAndCross(TestImages.Circle(),TestImages.Star(2)), width=4.0);

# ╔═╡ 29700517-53ff-493a-acec-e4eac297eee1
J₁₁ = extend_image(I₁₁, (128, 128))

# ╔═╡ 0f0ba691-f135-4647-8039-c6ba6ff09eea
I₁₂ = render(TestImages.OrbAndCross(TestImages.Square(),TestImages.Star(4)), width=4.0);

# ╔═╡ 324fb221-103b-463c-93ed-3e287c7f64f8
J₁₂ = extend_image(I₁₂, (128, 128))

# ╔═╡ 9ef98a53-81d3-444e-ba4e-e4a58c011fce
I₁₃ = render(TestImages.Shield(TestImages.Triangle()), width=4.0);

# ╔═╡ 911e62dc-58a9-48b7-a206-f60cc37c7548
J₁₃ = extend_image(I₁₃, (128, 128))

# ╔═╡ ea84ebd7-633c-497f-a4b6-7a34d3312ee3
I₁₄ = render(TestImages.Emblem(TestImages.Square()), width=4.0);

# ╔═╡ 93e65b37-ae92-40f1-90e5-39a9b9144b30
J₁₄ = extend_image(I₁₄, (128, 128))

# ╔═╡ 50ef63a9-f9f9-4edd-a7aa-d24d24f56043
J = [J₁, J₁₁, J₁₂, J₁₃, J₁₄]

# ╔═╡ fdbd201e-4bcf-4ef4-af73-d71712f60b39
nP = 256; P = TestImages.generate_phantom(nP);

# ╔═╡ 74ac3469-bee9-4856-bfd8-4d45654027f8
R = RadonTransform(851,256)

# ╔═╡ bd9637f0-6872-4f19-a66d-c6a3bd0ee592
Plots.heatmap(P[:,:,100], size=(550,500),c=colormap("Grays"))

# ╔═╡ cdc54d42-4469-430e-a1d9-3dc1bf4cc710
rP = R(J[3]);
# rP = R(P[:,:,100]);

# ╔═╡ 8a1cdfb4-6e20-4bf1-bb18-5d703c3131b4
Plots.heatmap(rP)

# ╔═╡ a588f894-0e37-4c9b-affb-b9493a7755b8
B = Backprojection(128,128,"linear","ramp",0.1)

# ╔═╡ 7e28417a-fb1a-410a-89c2-396ac3c113df
bP = B(rP); bP[bP.<0.0] .= 0.0;

# ╔═╡ bb92eeae-e59b-4eeb-968b-00aa83572e9a
Plots.heatmap(bP, size=(550,500) ,c=colormap("Grays")
)

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╠═9bc6bbed-558b-471e-bfdb-62d18897a910
# ╠═d272426c-53c1-46f4-8d39-ab2f635ad683
# ╠═aa47398a-b248-48fd-943e-b349b6e8f6a8
# ╠═29700517-53ff-493a-acec-e4eac297eee1
# ╠═0f0ba691-f135-4647-8039-c6ba6ff09eea
# ╠═324fb221-103b-463c-93ed-3e287c7f64f8
# ╠═9ef98a53-81d3-444e-ba4e-e4a58c011fce
# ╠═911e62dc-58a9-48b7-a206-f60cc37c7548
# ╠═ea84ebd7-633c-497f-a4b6-7a34d3312ee3
# ╠═93e65b37-ae92-40f1-90e5-39a9b9144b30
# ╠═50ef63a9-f9f9-4edd-a7aa-d24d24f56043
# ╠═fdbd201e-4bcf-4ef4-af73-d71712f60b39
# ╠═74ac3469-bee9-4856-bfd8-4d45654027f8
# ╠═bd9637f0-6872-4f19-a66d-c6a3bd0ee592
# ╠═cdc54d42-4469-430e-a1d9-3dc1bf4cc710
# ╠═8a1cdfb4-6e20-4bf1-bb18-5d703c3131b4
# ╠═a588f894-0e37-4c9b-affb-b9493a7755b8
# ╠═7e28417a-fb1a-410a-89c2-396ac3c113df
# ╠═bb92eeae-e59b-4eeb-968b-00aa83572e9a
