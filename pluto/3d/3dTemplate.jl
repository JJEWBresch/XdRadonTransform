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
end

# ╔═╡ 9bc6bbed-558b-471e-bfdb-62d18897a910
I₁ = render(TestImages.Circle(), size=(36,36), width=2.0);

# ╔═╡ d272426c-53c1-46f4-8d39-ab2f635ad683
J₁ = extend_image(I₁, (64, 64))

# ╔═╡ aa47398a-b248-48fd-943e-b349b6e8f6a8
I₂ = render(TestImages.OrbAndCross(TestImages.Circle(),TestImages.Star(2)), size=(36,36), width=2.0);

# ╔═╡ 29700517-53ff-493a-acec-e4eac297eee1
J₂ = extend_image(I₂, (64, 64))

# ╔═╡ 0f0ba691-f135-4647-8039-c6ba6ff09eea
I₃ = render(TestImages.OrbAndCross(TestImages.Square(),TestImages.Star(4)), size=(36,36), width=2.0);

# ╔═╡ 324fb221-103b-463c-93ed-3e287c7f64f8
J₃ = extend_image(I₃, (64, 64))

# ╔═╡ 9ef98a53-81d3-444e-ba4e-e4a58c011fce
I₄ = render(TestImages.Shield(TestImages.Triangle()), size=(36,36), width=2.0);

# ╔═╡ 911e62dc-58a9-48b7-a206-f60cc37c7548
J₄ = extend_image(I₄, (64, 64))

# ╔═╡ 50ef63a9-f9f9-4edd-a7aa-d24d24f56043
J = [J₁, J₂, J₃, J₄]

# ╔═╡ 2662e8a0-a39b-40ba-96bd-f1544fd4e158
J3d = [generate_3d_rot(J₁, order=2.0),generate_3d_rot(J₂, order=2.0),generate_3d_rot(J₃, order=2.0),generate_3d_rot(J₄, order=2.0)];

# ╔═╡ e9911f32-ec10-4ffe-a4dd-6b1fb1d44b05
volume(J3d[1], algorithm=:mip, colormap=:magma)

# ╔═╡ ff0f0c7d-8541-4611-979b-70d2287205a1
volume(J3d[2], algorithm=:mip, colormap=:magma)

# ╔═╡ 0614c960-edcf-4e82-874e-f7ae30bcb587
volume(J3d[3], algorithm=:mip, colormap=:magma)

# ╔═╡ 1bcf7cba-8e52-4fa1-bc73-31a2d1359cec
volume(J3d[4], algorithm=:mip, colormap=:magma)

# ╔═╡ 7a9b6963-3ecc-4ffb-9adc-9e32baa48981
A3d = RandomAffineTransformation3d(
	#rotate_x = (-π,π),
	#rotate_y = (-π,π),
	#rotate_z = (-π,π),
	scale_x = (0.5, 1.25),
	scale_y = (0.5, 1.25),
	scale_z = (0.5, 1.25),
	#shear_xy = (-0.45, 0.45),
	#shear_yx = (-0.45, 0.45),
	#shear_xz = (-0.45, 0.45),
	#shear_zx = (-0.45, 0.45),
	#shear_yz = (-0.45, 0.45),
	#shear_zy = (-0.45, 0.45),
	shift_x = (-10,10),
	shift_y = (-10,10),
	shift_z = (-10,10)
)

# ╔═╡ 5049bc96-83b0-4260-a617-dd1e5d5d57e0
TJ3d = A3d.(J3d);

# ╔═╡ 0e032711-d004-440a-b529-90d966e2a766
for k in 1:length(TJ3d)
	TJ3d[k][isnan.(TJ3d[k])] .= 0.0;
	k += 1
end

# ╔═╡ 8a161f3b-8d6d-4b6e-b113-3d40230dc609
R3d = RadonTransform3d(51,11,11)

# ╔═╡ fae7e7a0-4e52-4e68-a78c-0d6b6624ee45
RJ3d = R3d.(J3d);

# ╔═╡ 3e16567e-53aa-49c3-9705-18dfc0f955fb
volume(RJ3d[1], algorithm=:mip, colormap=:magma)

# ╔═╡ 40b21d15-be47-4cdd-a4d0-57ffad24d1be
Plots.heatmap(RJ3d[1][:,:,5])

# ╔═╡ 06e290e5-e7cb-4535-9976-d3fd8886545b
volume(RJ3d[2], algorithm=:mip, colormap=:magma)

# ╔═╡ 033ba164-42f9-4def-8eda-374e5cc06bb9
volume(RJ3d[3], algorithm=:mip, colormap=:magma)

# ╔═╡ a78b05ef-a5e2-4428-82fb-e8c04ed6acf3
volume(RJ3d[4], algorithm=:mip, colormap=:magma)

# ╔═╡ cd3ce46e-e6ae-4729-b272-19ca6bd4bcae
RTJ3d = R3d.(TJ3d);

# ╔═╡ 83850e18-d752-4cc7-9dfd-8dd924356140
volume(RTJ3d[1], algorithm=:mip, colormap=:magma)

# ╔═╡ 252ea80d-356e-4606-b9a6-a70b0d838e0e
Plots.heatmap(RTJ3d[1][:,:,5])

# ╔═╡ a2ba1f8c-f590-4195-a9b5-a0cc588bf568
volume(RTJ3d[2], algorithm=:mip, colormap=:magma)

# ╔═╡ 9b3047ea-7a55-4cc4-a2e9-3683695f5d38
volume(RTJ3d[3], algorithm=:mip, colormap=:magma)

# ╔═╡ b3c63365-722e-477a-ac35-90417467ae64
volume(RTJ3d[4], algorithm=:mip, colormap=:magma)

# ╔═╡ fdbd201e-4bcf-4ef4-af73-d71712f60b39
nP = 128; P = TestImages.generate_phantom(nP);

# ╔═╡ 1134d325-07d0-4069-b0b1-51456e1add8d
pP = []; for k in 1:32
	push!(pP, Plots.heatmap(P[:,:,Int(nP/32)*k], colorbar=false, axes=false, xticks=false, yticks=false)) 
end

# ╔═╡ b4ceaf21-397f-4320-97b2-70603fa796fb
Plots.plot(pP[1], pP[2], pP[3], pP[4], pP[5], pP[6], pP[7], pP[8], pP[9], pP[10], pP[11], pP[12], pP[13], pP[14], pP[15], pP[16], pP[17], pP[18], pP[19], pP[20], pP[21], pP[22], pP[23], pP[24], pP[25], pP[26], pP[27], pP[28], pP[29], pP[30], pP[31], pP[32], layout=(4,8), size=(800,400), margin=0Plots.mm, left_margin=-2Plots.mm, right_margin=-2Plots.mm, top_margin=-2Plots.mm, bottom_margin=-2Plots.mm)

# ╔═╡ 2a41a388-ccc7-42dc-ba91-4ea9a11fc10f
rP = R3d(P);

# ╔═╡ d347dbc3-49c4-436d-8e7d-5ec86422361f
volume(rP, algorithm=:mip, colormap=:magma)

# ╔═╡ 69078640-e19c-44aa-bfde-f79e589a55b2
Plots.heatmap(rP[:,:,11])

# ╔═╡ c3187b37-f3c0-4c8b-be92-6a39446a1659
prP = []; for k in 1:11
	push!(prP, Plots.heatmap(rP[:,:,k], colorbar=false, axes=false, xticks=false, yticks=false)) 
end

# ╔═╡ 12017f2d-1c7b-4292-85b3-dd7340c6cf8f
plP = []; for k in 1:10
	push!(plP, Plots.heatmap(rP[:,k,:], colorbar=false, axes=false, xticks=false, yticks=false)) 
end

# ╔═╡ 4bdadfc7-4fc0-4288-9eb3-88951d728f46
Plots.plot(prP[1], prP[2], prP[3], prP[4], prP[5], prP[6], prP[7], prP[8], prP[9], prP[10], plP[1], plP[2], plP[3], plP[4], plP[5], plP[6], plP[7], plP[8], plP[9], plP[10], layout=(2,10), size=(1100,200), margin=0Plots.mm, left_margin=-2Plots.mm, right_margin=-2Plots.mm, top_margin=-2Plots.mm, bottom_margin=-2Plots.mm)

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
# ╠═50ef63a9-f9f9-4edd-a7aa-d24d24f56043
# ╠═2662e8a0-a39b-40ba-96bd-f1544fd4e158
# ╠═e9911f32-ec10-4ffe-a4dd-6b1fb1d44b05
# ╠═ff0f0c7d-8541-4611-979b-70d2287205a1
# ╠═0614c960-edcf-4e82-874e-f7ae30bcb587
# ╠═1bcf7cba-8e52-4fa1-bc73-31a2d1359cec
# ╠═7a9b6963-3ecc-4ffb-9adc-9e32baa48981
# ╠═5049bc96-83b0-4260-a617-dd1e5d5d57e0
# ╠═0e032711-d004-440a-b529-90d966e2a766
# ╠═8a161f3b-8d6d-4b6e-b113-3d40230dc609
# ╠═fae7e7a0-4e52-4e68-a78c-0d6b6624ee45
# ╠═3e16567e-53aa-49c3-9705-18dfc0f955fb
# ╠═40b21d15-be47-4cdd-a4d0-57ffad24d1be
# ╠═06e290e5-e7cb-4535-9976-d3fd8886545b
# ╠═033ba164-42f9-4def-8eda-374e5cc06bb9
# ╠═a78b05ef-a5e2-4428-82fb-e8c04ed6acf3
# ╠═cd3ce46e-e6ae-4729-b272-19ca6bd4bcae
# ╠═83850e18-d752-4cc7-9dfd-8dd924356140
# ╠═252ea80d-356e-4606-b9a6-a70b0d838e0e
# ╠═a2ba1f8c-f590-4195-a9b5-a0cc588bf568
# ╠═9b3047ea-7a55-4cc4-a2e9-3683695f5d38
# ╠═b3c63365-722e-477a-ac35-90417467ae64
# ╠═fdbd201e-4bcf-4ef4-af73-d71712f60b39
# ╠═1134d325-07d0-4069-b0b1-51456e1add8d
# ╠═b4ceaf21-397f-4320-97b2-70603fa796fb
# ╠═2a41a388-ccc7-42dc-ba91-4ea9a11fc10f
# ╠═d347dbc3-49c4-436d-8e7d-5ec86422361f
# ╠═69078640-e19c-44aa-bfde-f79e589a55b2
# ╠═c3187b37-f3c0-4c8b-be92-6a39446a1659
# ╠═12017f2d-1c7b-4292-85b3-dd7340c6cf8f
# ╠═4bdadfc7-4fc0-4288-9eb3-88951d728f46
