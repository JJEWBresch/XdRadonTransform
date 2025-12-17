### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 8cbe0300-edff-11ef-2fad-d3b8cca171a9
begin
	import Pkg
	Pkg.activate("..")    
	using Revise
	using XdRadonTransform
	using XdRadonTransform.TestImages
	using Images
	using Plots
	using LinearAlgebra
	plotlyjs()
end

# ╔═╡ 7a209b02-782f-4b79-a4b4-d89029206c23
using ImagePhantoms, Statistics

# ╔═╡ dde2b58d-f858-4201-8fa8-94165d256b66
# SheppLogan
# SheppLoganEmis
# SheppLoganBrainWeb
# SheppLoganToft

# ╔═╡ 6c40e0bf-2c24-44f7-8a40-c7f92fe1a3d1
d = 512

# ╔═╡ cbca8d0c-a240-4c24-883b-0068f3f7a24f
image = ImagePhantoms.shepp_logan(d, ImagePhantoms.SheppLoganToft());
#image = render(Circle(), size=(256,256), width=10.0)
#image = render(Square(), size=(256,256), width=10.0)

# ╔═╡ 7391c53c-d495-4388-8bc7-04bfac6a98b8
P = scale(Phantom(image))

# ╔═╡ 8202320e-fe6d-4761-89da-838346b8c258
begin
	num_radii = 513
	num_angles = 512
	R = RadonTransform(uniform_radii(num_radii), uniform_circle(num_angles))
end

# ╔═╡ 4c112924-ec35-4204-b53c-fefa86af2f86
S = R(P)

# ╔═╡ 9f59bc9a-f6f2-4b37-a62e-c0ef86e022ab
heatmap(S.data)

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╠═7a209b02-782f-4b79-a4b4-d89029206c23
# ╠═dde2b58d-f858-4201-8fa8-94165d256b66
# ╠═6c40e0bf-2c24-44f7-8a40-c7f92fe1a3d1
# ╠═cbca8d0c-a240-4c24-883b-0068f3f7a24f
# ╠═7391c53c-d495-4388-8bc7-04bfac6a98b8
# ╠═8202320e-fe6d-4761-89da-838346b8c258
# ╠═4c112924-ec35-4204-b53c-fefa86af2f86
# ╠═9f59bc9a-f6f2-4b37-a62e-c0ef86e022ab
