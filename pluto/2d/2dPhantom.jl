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

# ╔═╡ 7a209b02-782f-4b79-a4b4-d89029206c23
using ImagePhantoms, Statistics

# ╔═╡ 0791c621-08c5-4772-bbcb-906957a721be
using RadonKA

# ╔═╡ dde2b58d-f858-4201-8fa8-94165d256b66
# SheppLogan
# SheppLoganEmis
# SheppLoganBrainWeb
# SheppLoganToft

# ╔═╡ 6c40e0bf-2c24-44f7-8a40-c7f92fe1a3d1
d = 1024

# ╔═╡ cbca8d0c-a240-4c24-883b-0068f3f7a24f
image = ImagePhantoms.shepp_logan(d, ImagePhantoms.SheppLoganToft());

# ╔═╡ 8b759fa1-91c4-42f1-97aa-583b445417d6
shapelogan_d = Plots.heatmap(image, color=:grays)

# ╔═╡ 9df08053-c96e-4677-86f8-e9756a432902
image_large = zeros(2*d,2*d); image_large[Int(d/2):Int(3*d/2)-1,Int(d/2):Int(3*d/2)-1] = image;

# ╔═╡ 758a65df-be21-499d-9a46-25a0dfb79a06
# ╠═╡ disabled = true
#=╠═╡
savefig(shapelogan_d, "shapelogan_1024.pdf")
  ╠═╡ =#

# ╔═╡ c753f989-f12f-44c1-8766-43d790bbd635
p = ImagePhantoms.shepp_logan(SheppLoganToft());

# ╔═╡ 6046bce1-8495-4c39-8e54-c684e8d4ddea
r = 501

# ╔═╡ 030eccaa-119e-4201-a132-831dcb86d2e6
a = 360

# ╔═╡ 65ae09ac-8d45-4f96-a73c-5658a8200916
sinogram = ImagePhantoms.radon(range(-1/sqrt(2),1/sqrt(2),r), deg2rad.(range(-90,270,a+1)[1:end-1]), p);

# ╔═╡ 834e3571-c71f-4f2d-8470-f8dc1dcdb667
sinoexactP = Plots.heatmap(sinogram)

# ╔═╡ bc312a49-0150-406f-8de0-20410ba5de37
# ╠═╡ disabled = true
#=╠═╡
savefig(sinoexactP, "sinoexactP_201_361_1024.pdf")
  ╠═╡ =#

# ╔═╡ 74ac3469-bee9-4856-bfd8-4d45654027f8
R = RadonTransform(r,a)

# ╔═╡ cdc54d42-4469-430e-a1d9-3dc1bf4cc710
rP = R(image); rPr = reverse(rP, dims=2); size(rPr)

# ╔═╡ 8a1cdfb4-6e20-4bf1-bb18-5d703c3131b4
sinorPr = Plots.heatmap(rPr)

# ╔═╡ 4f8e1835-2919-47e9-b0d1-c7936af185d4
# ╠═╡ disabled = true
#=╠═╡
savefig(sinorPr, "sinorPr_201_361_1024.pdf")
  ╠═╡ =#

# ╔═╡ 5b47b7b7-887b-4f31-ab3f-9576a4837c03
diffrPr = Plots.heatmap(abs.(rPr.*1/sqrt(2) .- sinogram))

# ╔═╡ 0693b458-a817-4f27-a434-da011040088d
# ╠═╡ disabled = true
#=╠═╡
savefig(diffrPr, "diffrPr_201_361_1024.pdf")
  ╠═╡ =#

# ╔═╡ 4fe62680-20d5-42b1-8b99-20d7af42d1ec
maximum(abs.(rPr.*1/sqrt(2) .- sinogram)), mean(abs.(rPr.*1/sqrt(2) .- sinogram)), var(abs.(rPr.*1/sqrt(2) .- sinogram))

# ╔═╡ c757ac44-9bf4-482d-b473-5c8677bde6cc
geoKA = RadonKA.RadonParallelCircle(2*d, range(-d/2*sqrt(2),d/2*sqrt(2),r));

# ╔═╡ 26bab76f-0851-4cdd-967e-4a7acb9bc476
angKA=Array(range(0,2*π,a+1))[1:end-1];

# ╔═╡ a666dfcc-dfdd-4302-88f3-3f0d53f31a67
rkaP = RadonKA.radon(image_large, angKA; geometry=geoKA); rkaPr = reverse(rkaP, dims=2); size(rkaP)

# ╔═╡ bf631f71-6800-40e8-be00-bd2e7ee160a4
sinorkaPr = Plots.heatmap(rkaPr)

# ╔═╡ 83f8d6bc-1caf-4dcc-9345-c85ef3f0429d
# ╠═╡ disabled = true
#=╠═╡
savefig(sinorkaPr, "sinorkaPr_201_361_1024.pdf")
  ╠═╡ =#

# ╔═╡ c28e022d-a699-4fb2-8a7e-745906dbf9c0
diffrkaPr = Plots.heatmap(abs.(rkaPr/d .- sinogram))

# ╔═╡ 2459dd02-8208-4020-9338-5f64d332ba39
# ╠═╡ disabled = true
#=╠═╡
savefig(diffrkaPr, "diffrkaPr_201_361_1024.pdf")
  ╠═╡ =#

# ╔═╡ 3386f3bc-50a3-4dc3-8507-32eeea05d262
maximum(abs.(rkaPr/d .- sinogram)), mean(abs.(rkaPr/d .- sinogram)), var(abs.(rkaPr/d .- sinogram))

# ╔═╡ a588f894-0e37-4c9b-affb-b9493a7755b8
B = Backprojection(1445,1445,"linear","ramp",0.001)
#B = Backprojection(1448,1448,"linear","ramp",0.01)

# ╔═╡ 7e28417a-fb1a-410a-89c2-396ac3c113df
bP = B(rP); bP[bP.<0.0] .= 0.0;

# ╔═╡ ba700aa4-7b54-468a-a611-51d6e0df0019
bPred = rotl90(bP[213:1236,213:1236]./maximum(bP[213:1236,213:1236]));

# ╔═╡ bb92eeae-e59b-4eeb-968b-00aa83572e9a
image_re = Plots.heatmap(bPred[400:600,400:600],color=:grays)

# ╔═╡ 0c150993-0fa8-4a46-b43c-303723141f33
Plots.heatmap(bPred,color=:grays)

# ╔═╡ 2ad49b8c-7671-4274-99ab-2db6bf80deb3
# ╠═╡ disabled = true
#=╠═╡
savefig(image_re, "shapelogan_1024_re.pdf")
  ╠═╡ =#

# ╔═╡ 6ed4b710-17a1-4c86-b930-52a4276a38a7
rbP = rotl90(bP[213:1236,213:1236]); size(rbP)

# ╔═╡ 7dfeed07-15cb-4398-9f68-beb2709f3af2
image_diff = Plots.heatmap(abs.(rbP./maximum(rbP) .- image),color=:grays)

# ╔═╡ 0281c046-e41a-4d1a-91e5-9e2c3c5095d9
# ╠═╡ disabled = true
#=╠═╡
savefig(image_diff, "shapelogan_1024_diff.pdf")
  ╠═╡ =#

# ╔═╡ 811e5885-c1e2-4a61-b00e-11b5ce150b2f
maximum(abs.(rbP./maximum(rbP) .- image)), mean(abs.(rbP./maximum(rbP) .- image)),var(abs.(rbP./maximum(rbP) .- image))

# ╔═╡ 92d57f91-f727-43f9-86ca-f998e7a38c85
bkaP = RadonKA.backproject_filtered(rkaPr,angKA;geometry=geoKA); bkaP[bkaP.<0.0].=0.0;

# ╔═╡ c9173339-026c-4594-9f30-f77920f5572d
bkaPa = abs.(bkaP[512:512+1023,512:512+1023])./maximum(abs.(bkaP[512:512+1023,512:512+1023]));

# ╔═╡ 7d1f40ef-475c-41cc-8c85-4d1e7e769463
bkaPa[bkaPa.>0.5].=0.5

# ╔═╡ ddd862d6-6336-406c-9913-a4b9fcfaafc4
bkapared = abs.(bkaPa[end:-1:1,:]./maximum(bkaPa));

# ╔═╡ e115253d-1c87-4b8b-8675-6f6dc9583c1e
Plots.heatmap(bkapared[400:600,400:600],color=:grays)

# ╔═╡ 0545bb1e-1c8c-4bbc-9450-4b6881900832
Plots.heatmap(abs.(bkaPa[end:-1:1,:]./maximum(bkaPa).-image),color=:grays)

# ╔═╡ 3a340021-298e-4727-ac4b-b098df2eed53
maximum(abs.(bkaPa[end:-1:1,:]./maximum(bkaPa).-image)), mean(abs.(bkaPa[end:-1:1,:]./maximum(bkaPa).-image)),var(abs.(bkaPa[end:-1:1,:]./maximum(bkaPa).-image))

# ╔═╡ 47cf3dde-861c-4b97-a6e2-caacc4eb88e3
bkaPl = RadonKA.backproject_filtered(rkaPr,angKA); bkaPl[bkaPl.<0.0].=0.0;

# ╔═╡ 54aef50f-3ed0-446c-9530-87217ef9d1aa
bkaPli = imresize(bkaPl[250-175:250+175,250-175:250+175],(1024,1024));

# ╔═╡ 4f44d76e-3eed-4518-83b2-d1a2ea9e8fe1
bkaPlired = abs.(bkaPli[end:-1:1,:]./maximum(bkaPli));

# ╔═╡ b8a60cf9-d157-4c18-834e-30ba7933b411
Plots.heatmap(bkaPlired[400:600,400:600],color=:grays)

# ╔═╡ d3c1aa02-740e-495e-abb4-37f930a7d414
Plots.heatmap(abs.(bkaPli[end:-1:1,:]./maximum(bkaPli).-image),color=:grays)

# ╔═╡ cf4323bd-a61b-4274-b129-49ed107c647c
maximum(abs.(bkaPli[end:-1:1,:]./maximum(bkaPli) .- image)), mean(abs.(bkaPli[end:-1:1,:]./maximum(bkaPli) .- image)),var(abs.(bkaPli[end:-1:1,:]./maximum(bkaPli) .- image))

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╠═7a209b02-782f-4b79-a4b4-d89029206c23
# ╠═0791c621-08c5-4772-bbcb-906957a721be
# ╠═dde2b58d-f858-4201-8fa8-94165d256b66
# ╠═6c40e0bf-2c24-44f7-8a40-c7f92fe1a3d1
# ╠═cbca8d0c-a240-4c24-883b-0068f3f7a24f
# ╠═8b759fa1-91c4-42f1-97aa-583b445417d6
# ╠═9df08053-c96e-4677-86f8-e9756a432902
# ╠═758a65df-be21-499d-9a46-25a0dfb79a06
# ╠═c753f989-f12f-44c1-8766-43d790bbd635
# ╠═6046bce1-8495-4c39-8e54-c684e8d4ddea
# ╠═030eccaa-119e-4201-a132-831dcb86d2e6
# ╠═65ae09ac-8d45-4f96-a73c-5658a8200916
# ╠═834e3571-c71f-4f2d-8470-f8dc1dcdb667
# ╠═bc312a49-0150-406f-8de0-20410ba5de37
# ╠═74ac3469-bee9-4856-bfd8-4d45654027f8
# ╠═cdc54d42-4469-430e-a1d9-3dc1bf4cc710
# ╠═8a1cdfb4-6e20-4bf1-bb18-5d703c3131b4
# ╠═4f8e1835-2919-47e9-b0d1-c7936af185d4
# ╠═5b47b7b7-887b-4f31-ab3f-9576a4837c03
# ╠═0693b458-a817-4f27-a434-da011040088d
# ╠═4fe62680-20d5-42b1-8b99-20d7af42d1ec
# ╠═c757ac44-9bf4-482d-b473-5c8677bde6cc
# ╠═26bab76f-0851-4cdd-967e-4a7acb9bc476
# ╠═a666dfcc-dfdd-4302-88f3-3f0d53f31a67
# ╠═bf631f71-6800-40e8-be00-bd2e7ee160a4
# ╠═83f8d6bc-1caf-4dcc-9345-c85ef3f0429d
# ╠═c28e022d-a699-4fb2-8a7e-745906dbf9c0
# ╠═2459dd02-8208-4020-9338-5f64d332ba39
# ╠═3386f3bc-50a3-4dc3-8507-32eeea05d262
# ╠═a588f894-0e37-4c9b-affb-b9493a7755b8
# ╠═7e28417a-fb1a-410a-89c2-396ac3c113df
# ╠═ba700aa4-7b54-468a-a611-51d6e0df0019
# ╠═bb92eeae-e59b-4eeb-968b-00aa83572e9a
# ╠═0c150993-0fa8-4a46-b43c-303723141f33
# ╠═2ad49b8c-7671-4274-99ab-2db6bf80deb3
# ╠═6ed4b710-17a1-4c86-b930-52a4276a38a7
# ╠═7dfeed07-15cb-4398-9f68-beb2709f3af2
# ╠═0281c046-e41a-4d1a-91e5-9e2c3c5095d9
# ╠═811e5885-c1e2-4a61-b00e-11b5ce150b2f
# ╠═92d57f91-f727-43f9-86ca-f998e7a38c85
# ╠═c9173339-026c-4594-9f30-f77920f5572d
# ╠═7d1f40ef-475c-41cc-8c85-4d1e7e769463
# ╠═ddd862d6-6336-406c-9913-a4b9fcfaafc4
# ╠═e115253d-1c87-4b8b-8675-6f6dc9583c1e
# ╠═0545bb1e-1c8c-4bbc-9450-4b6881900832
# ╠═3a340021-298e-4727-ac4b-b098df2eed53
# ╠═47cf3dde-861c-4b97-a6e2-caacc4eb88e3
# ╠═54aef50f-3ed0-446c-9530-87217ef9d1aa
# ╠═4f44d76e-3eed-4518-83b2-d1a2ea9e8fe1
# ╠═b8a60cf9-d157-4c18-834e-30ba7933b411
# ╠═d3c1aa02-740e-495e-abb4-37f930a7d414
# ╠═cf4323bd-a61b-4274-b129-49ed107c647c
