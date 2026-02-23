### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 8cbe0300-edff-11ef-2fad-d3b8cca171a9
begin
	import Pkg
	Pkg.activate("../..")  
	using Revise
	using XdRadonTransform
	using XdRadonTransform.DataTransformations
	using XdRadonTransform.TestImages
	using Images
	using Plots
	# plotlyjs()
end

# ╔═╡ c20ec051-9ae9-40f7-97cb-7abf2baf4fe4
using FileIO, MeshIO, GLMakie, Flux3D, PlotlyJS # CairoMakie, Flux3D

# ╔═╡ 1fa304a4-ecf5-4645-8f3a-90aedde4323e
using Markdown, InteractiveUtils

# ╔═╡ 178b1a7f-ceeb-43ba-9006-e2c5fd7f6b05
using Interpolations: LinearInterpolation

# ╔═╡ 62aa8854-c7b6-4da4-a904-22d56ed5364c
using Statistics, Random, StatsBase, LinearAlgebra

# ╔═╡ eb9a0617-8519-4ce4-a611-485bcbd723fb
md"""
# arXiv:xxxx.xxxxx -- Table x, Figure x
This pluto notebook reproduces the numerical visualizations
with tamplates for Figure x from

- Robert Beinert, Jonas Bresch, and Michael Quellmalz
  '*A Discrete Radon Transform Based on the Area of Cube-Plane Intersection*',
  arXiv:xxxx.xxxxx, 2026.
"""

# ╔═╡ 3208b1c3-d726-4027-aee9-5875b0f79965
md"""
## Dataset
Load the reduced `ModelNet10` dataset.
Further information about ModelNet10 can be found in

- Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang and J. Xiao, 
  '[3D ShapeNets: A deep representation for volumetric shapes]
  (https://doi.org/10.1109/CVPR.2015.7298801)',
  *IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)*,
  1912--920 (2015). 
"""

# ╔═╡ 4a94351b-d4d4-4a34-9505-7a40697c6704
begin
	voxel_bathtub = load("../../data/smallModelNet10/voxel_bathtub.jld2")["voxel_bathtub"]
	voxel_bed = load("../../data/smallModelNet10/voxel_bed.jld2")["voxel_bed"]
	voxel_chair = load("../../data/smallModelNet10/voxel_chair.jld2")["voxel_chair"]
	voxel_desk = load("../../data/smallModelNet10/voxel_desk.jld2")["voxel_desk"]
	voxel_dresser = load("../../data/smallModelNet10/voxel_dresser.jld2")["voxel_dresser"]
	voxel_monitor = load("../../data/smallModelNet10/voxel_monitor.jld2")["voxel_monitor"]
	voxel_night_stand = load("../../data/smallModelNet10/voxel_night_stand.jld2")["voxel_night_stand"]
	voxel_sofa = load("../../data/smallModelNet10/voxel_sofa.jld2")["voxel_sofa"]
	voxel_table = load("../../data/smallModelNet10/voxel_table.jld2")["voxel_table"]
	voxel_toilet = load("../../data/smallModelNet10/voxel_toilet.jld2")["voxel_toilet"]
end

# ╔═╡ 7de17916-6d95-44d7-959b-a48f504f8f07
begin
	smallModelNet10 = [voxel_bathtub[1],voxel_bed[1],voxel_chair[1],voxel_desk[1],voxel_dresser[1],voxel_monitor[1],voxel_night_stand[1],voxel_sofa[1],voxel_table[1],voxel_toilet[1]]
	Label = 1:10
end

# ╔═╡ 085e2183-97b3-4a7a-873a-560e9e305f36
begin
	function extend_3d(A::AbstractArray,d::Int64)
		dd = max(d,128)
		voxel_array = zeros(dd,dd,dd)
		voxel_array[dd÷4:dd÷4+63,dd÷4:dd÷4+63,dd÷4:dd÷4+63] = A
		return voxel_array
	end
	ModelNet10 = extend_3d.(smallModelNet10,128)
end

# ╔═╡ 5c07bf3d-b1bc-499f-8c6f-60821fe19cef
md"""
## Template Dataset
Generate the template dataset
using the submodule `TestImages`.
"""

# ╔═╡ d8c1637c-3d72-4124-8267-79b036051ca2
Class, Labels = generate_academic_classes_3d(ModelNet10, Label, class_size=10);

# ╔═╡ 9e927481-a86b-486d-967f-57786d5c36a1
A = RandomAffineTransformation3d(
	rotate_x = (-π,π),
	rotate_y = (-π,π),
	rotate_z = (-π,π),
	scale_x = (0.5, 1.25),
	scale_y = (0.5, 1.25),
	scale_z = (0.5, 1.25),
	shear_xy = (-0.45, 0.45),
	shear_yx = (-0.45, 0.45),
	shear_xz = (-0.45, 0.45),
	shear_zx = (-0.45, 0.45),
	shear_yz = (-0.45, 0.45),
	shear_zy = (-0.45, 0.45),
	shift_x = (-10,10),
	shift_y = (-10,10),
	shift_z = (-10,10)
)

# ╔═╡ 8a45a3b5-1005-407a-b51d-3a0369beff0f
AClass = A.(Class)

# ╔═╡ 9fa792fc-a766-4c0c-bf97-71034c790cd5
for k in 1:length(AClass)
	AClass[k][isnan.(AClass[k])] .= 0.0;
	k += 1
end

# ╔═╡ 7df0a0ed-762b-48cf-b7d8-1ab8d3f86076
GLMakie.volume(AClass[1], algorithm=:mip, colormap=:magma)

# ╔═╡ 3a2fbfdf-5aa6-4a5a-b58a-b6ac49d1d076
GLMakie.volume(AClass[11], algorithm=:mip, colormap=:magma)

# ╔═╡ 9a88aa43-f6b6-458a-939b-a91510bdf571
begin
	"""
	    radon3d(volume, radii, dirs; voxel_size=(1.0,1.0,1.0),
	            origin=nothing, linear=true, normalize_dirs=true, tol_uniform=1e-8)
	
	Compute an approximate 3D Radon transform of `volume` by projecting each voxel center
	onto lines defined by `dirs` (each a 3-vector) and summing voxel values into `radii` bins.
	
	# Arguments
	- `volume::AbstractArray{T,3}` : 3D array (nx,ny,nz) of voxel values (T numeric).
	- `radii::AbstractVector{R}` : 1D array of radii/positions along the line (must be sorted ascending).
	- `dirs` : a collection of direction vectors. Accepts:
	    - `Array{<:AbstractVector}` (vector of 3-vectors), or
	    - `Matrix` with size (ndirs, 3) or (3, ndirs).
	- Keyword args:
	- `voxel_size` : tuple (dx,dy,dz) physical size of a voxel (defaults to (1,1,1)).
	- `origin` : if provided, should be a tuple (ox,oy,oz) in world coords; otherwise voxel centers are centered so that middle voxel is at 0.
	- `linear` : if `true` use linear interpolation between neighbouring radii; otherwise nearest-bin.
	- `normalize_dirs` : if `true` the direction vectors are normalized.
	- `tol_uniform` : tolerance to detect uniform spacing in `radii`.
	
	# Returns
	- `R` : matrix of size (ndirs, length(radii)). Row `i` is the projection-sum for `dirs[i]`.
	
	# Notes
	- Complexity: O(ndirs * nvoxels). Try fewer directions or downsample volume for speed.
	- If `radii` is uniform, a fast direct map is used; otherwise binary search used per voxel.
	"""
	struct ApproxRadonTransform
	    radii::Vector{Float64}
	    angles::Matrix{Float64}
		linear::Bool
	end

	#function ApproxRadonTransform(radii::Vector{Float64}, angles::Matrix{Float64}, linear::Bool)
	#    return ApproxRadonTransform(radii, angles, linear)
	#end

	function (AR::ApproxRadonTransform)(volume::AbstractArray)
		dirs = [AR.angles[k,:] for k in 1:size(AR.angles)[1]]
		sino = radon3d(volume, AR.radii, dirs; AR.linear)
		return sino'
	end
	
	function radon3d(volume::AbstractArray, radii::AbstractVector, dirs;
	                 voxel_size=(1.0,1.0,1.0),
	                 origin=nothing,
	                 linear::Bool=true,
	                 normalize_dirs::Bool=true,
	                 tol_uniform::Float64=1e-8)
	
	    # sizes
	    nx, ny, nz = size(volume)
	    N = nx * ny * nz
	
	    # radii as Float64, require sorted ascending
	    radii = collect(Float64.(radii))
	    nr = length(radii)
	    if nr < 1
	        error("radii must be non-empty")
	    end
	    if !issorted(radii)
	        error("radii must be sorted ascending")
	    end
	
	    # voxel size
	    dx, dy, dz = Float64.(voxel_size)
	
	    # coordinate origin (world coords): default centers grid at 0
	    cx = (nx + 1) / 2
	    cy = (ny + 1) / 2
	    cz = (nz + 1) / 2
	    if origin === nothing
	        ox, oy, oz = 0.0, 0.0, 0.0
	        # mapping: x_world = (i - cx)*dx + ox
	    else
	        ox, oy, oz = Float64.(origin)
	    end
	
	    xs = ((1:nx) .- cx) .* dx .+ ox
	    ys = ((1:ny) .- cy) .* dy .+ oy
	    zs = ((1:nz) .- cz) .* dz .+ oz
	
	    # build flattened position matrix (N x 3)
	    positions = Array{Float64}(undef, N, 3)
	    idx = 1
	    for k in 1:nz
	        z = zs[k]
	        for j in 1:ny
	            y = ys[j]
	            for i in 1:nx
	                x = xs[i]
	                positions[idx,1] = x
	                positions[idx,2] = y
	                positions[idx,3] = z
	                idx += 1
	            end
	        end
	    end
	
	    # flattened intensities (matching same order as positions)
	    vals = Float64.(vec(volume))
	
	    # parse directions into an nd x 3 matrix
	    dirs_mat = nothing
	    if isa(dirs, AbstractMatrix)
	        # accept (nd,3) or (3,nd)
	        if size(dirs,2) == 3
	            dirs_mat = Array{Float64}(dirs)
	        elseif size(dirs,1) == 3
	            dirs_mat = Array{Float64}(permutedims(dirs))  # make nd x 3
	        else
	            error("dirs as matrix must be (nd,3) or (3,nd)")
	        end
	    else
	        # assume iterable of 3-vectors
	        dlist = collect(dirs)
	        nd = length(dlist)
	        dirs_mat = zeros(Float64, nd, 3)
	        for i in 1:nd
	            v = Float64.(dlist[i])
	            if length(v) != 3
	                error("each direction must be a 3-vector")
	            end
	            dirs_mat[i,:] .= v
	        end
	    end
	
	    ndirs = size(dirs_mat, 1)
	    # optionally normalize
	    if normalize_dirs
	        for i in 1:ndirs
	            nrm = norm(dirs_mat[i, :])
	            if nrm == 0
	                error("direction $i is zero vector")
	            end
	            dirs_mat[i, :] ./= nrm
	        end
	    end
	
	    # detect uniform radii spacing (for fast mapping)
	    is_uniform = false
	    if nr >= 2
	        diffs = diff(radii)
	        mean_d = mean(diffs)
	        if abs(mean_d) > 0 && maximum(abs.(diffs .- mean_d)) <= tol_uniform * max(1.0, abs(mean_d))
	            is_uniform = true
	            r0 = radii[1]
	            dr = mean_d
	        end
	    end
	
	    # prepare output: rows: dirs, cols: radii
	    sino = zeros(Float64, ndirs, nr)
	
	    # main loop over directions
	    for d_idx in 1:ndirs
	        u = view(dirs_mat, d_idx, :)
	        # projected distances t for all voxels
	        t = positions * u  # N-vector
	
	        if is_uniform
	            # map to fractional indices
	            idxf = (t .- r0) ./ dr .+ 1.0  # 1-based index (float)
	            # linear interpolation between floor and ceil indices
	            if linear
	                # accumulate using simple loop (safe and straightforward)
	                for i in 1:N
	                    fi = idxf[i]
	                    low = floor(Int, fi)
	                    w_high = fi - low
	                    v = vals[i]
	                    if 1 <= low <= nr
	                        sino[d_idx, low] += v * (1.0 - w_high)
	                    end
	                    high = low + 1
	                    if 1 <= high <= nr
	                        sino[d_idx, high] += v * w_high
	                    end
	                end
	            else
	                # nearest neighbor bin
	                idx_nn = round.(Int, idxf)
	                for i in 1:N
	                    k = idx_nn[i]
	                    if 1 <= k <= nr
	                        sino[d_idx, k] += vals[i]
	                    end
	                end
	            end
	        else
	            # non-uniform radii -> binary search per voxel
	            if linear
	                # linear interpolation between adjacent radii[k-1], radii[k]
	                for i in 1:N
	                    ti = t[i]
	                    # find first index >= ti
	                    k = searchsortedfirst(radii, ti)
	                    if k == 1
	                        # ti <= radii[1] -> put all weight to radii[1]
	                        sino[d_idx, 1] += vals[i]
	                    elseif k > nr
	                        # ti > radii[end] -> put all weight to end
	                        sino[d_idx, nr] += vals[i]
	                    else
	                        # radii[k-1] < ti <= radii[k]
	                        rlow = radii[k-1]
	                        rhigh = radii[k]
	                        if rhigh == rlow
	                            sino[d_idx, k] += vals[i]
	                        else
	                            w_high = (ti - rlow) / (rhigh - rlow)
	                            sino[d_idx, k-1] += vals[i] * (1.0 - w_high)
	                            sino[d_idx, k]   += vals[i] * w_high
	                        end
	                    end
	                end
	            else
	                # nearest bin (non-uniform): choose closer of radii[k-1], radii[k] where k = searchsortedfirst
	                for i in 1:N
	                    ti = t[i]
	                    k = searchsortedfirst(radii, ti)
	                    if k == 1
	                        sino[d_idx, 1] += vals[i]
	                    elseif k > nr
	                        sino[d_idx, nr] += vals[i]
	                    else
	                        # compare distances
	                        if abs(ti - radii[k-1]) <= abs(ti - radii[k])
	                            sino[d_idx, k-1] += vals[i]
	                        else
	                            sino[d_idx, k] += vals[i]
	                        end
	                    end
	                end
	            end
	        end
	    end
	
	    return sino
	end
end

# ╔═╡ a6fcea5b-3f86-42ac-b104-f4709ba2ee95
md"""
## Radon transform
Setup the Radon transform in `3d`
Discretization, given by 

- number of equispaced radii in [-√3⋅d/2,√3⋅d/2],
- Fibonacci sphere points (nearly equispaced on the S^2).
"""

# ╔═╡ 8d783205-ffa0-4ac6-957c-706b7505806b
begin
	d = 128
	r = 64
	radii = collect(LinRange(-sqrt(3)/2*(d-1),sqrt(3)/2*(d-1),r))
	angles = fibonacci_spherical_directions(64)
	AR = ApproxRadonTransform(radii,angles,false)
end

# ╔═╡ 0c46cd8e-57eb-46fe-8426-ef2aa69a2c56
AA = AR(ModelNet10[4])

# ╔═╡ 5db931a4-f5ef-4f0e-8729-3648c119147f
Plots.heatmap(AA)

# ╔═╡ 9359f6f1-ae2c-4313-b480-ffc433e98eb4
begin
	struct RadonCDT 
	    quantiles::Int64
	    Radon::ApproxRadonTransform
	    signal_raising::Float64
	    function RadonCDT(q, AR, ϵ)
	        if q <= 0
	            error("non-positive number of quantiles")
	        elseif ϵ <= 0
	            error("non-positive signal raising")
	        end
	        return new(q, AR, ϵ)
	    end
	end
	
	
	RadonCDT(q, AR) = RadonCDT(q, AR, 1e-10)
	
	
	function (RCDT::RadonCDT)(image::AbstractArray)
	    I = RCDT.Radon(image)
	    Q = zeros(RCDT.quantiles, size(RCDT.Radon.angles)[1])
	    for k in axes(Q, 2)
	        Q[:, k] = cdt(I[:, k], RCDT)
	    end
	    return Q
	end
	
	function cdt(s::Vector{Float64}, RCDT::RadonCDT)
	    p = signal_to_density(s, RCDT.signal_raising)
	    P = cumsum(p)
	    s_grid = collect(LinRange(-1, 1, length(s)))
	    q_grid = collect(LinRange(0, 1, RCDT.quantiles + 2))[2:end - 1]
	    q = LinearInterpolation(P, s_grid)
	    return q(q_grid)
	end
	
	function signal_to_density(s::Vector{Float64}, ϵ::Float64)
	    p = s .+ ϵ
	    return p ./= sum(p)
	end
	
	struct NormRadonCDT
	    RCDT::RadonCDT
	end
	
	function (NRCDT::NormRadonCDT)(image::AbstractArray)
	    rcdt = NRCDT.RCDT(image)
	    nrcdt = (rcdt .- mean(rcdt, dims=1)) ./ std(rcdt, dims=1)
	    return nrcdt
	end
	
	struct MaxNormRadonCDT
	    RCDT::RadonCDT
	end
	
	function (mNRCDT::MaxNormRadonCDT)(image::AbstractArray)
	    NRCDT = NormRadonCDT(mNRCDT.RCDT)
	    nrcdt = NRCDT(image)
	    mnrcdt = dropdims(maximum(dropdims(maximum(nrcdt, dims=2), dims=2), dims=2))
	    return mnrcdt
	end
	
	function normalization(rcdt::AbstractArray)
	    nrcdt = (rcdt .- mean(rcdt, dims=1)) ./ std(rcdt, dims=1)
	    return nrcdt
	end
	
	function max_normalization(rcdt::AbstractArray)
	    nrcdt = normalization(rcdt)
	    (nq, nt, np) = size(nrcdt)
	    mnrcdt = dropdims(maximum(reshape(nrcdt, (nq, nt*np)), dims=2), dims=2)
	    return mnrcdt
	end
	
	function mean_normalization(rcdt::AbstractArray)
	    nrcdt = normalization(rcdt)
	    (nq, nt, np) = size(nrcdt)
	    mnrcdt = dropdims(mean(reshape(nrcdt, (nq, nt*np)), dims=2), dims=2)
	    return mnrcdt
	end
end

# ╔═╡ aa720f14-298a-45df-a6f3-921fd5f77ac2
begin
	function maxnormalization(rcdt::AbstractArray)
	    nrcdt = normalization(rcdt)
	    mnrcdt = dropdims(maximum(nrcdt, dims=2), dims=2)
	    return mnrcdt
	end
	
	function meannormalization(rcdt::AbstractArray)
	    nrcdt = normalization(rcdt)
	    mnrcdt = dropdims(mean(nrcdt, dims=2), dims=2)
	    return mnrcdt
	end
end

# ╔═╡ 4beb1c43-5d2e-4580-a740-ac4edb625e17
begin
	RCDT = RadonCDT(64, AR);
	NRCDT = NormRadonCDT(RCDT);
end

# ╔═╡ 654dc837-3c6d-4b87-8ccb-6cbfadd2bb78
qTempNRCDT = NRCDT.RCDT.(ModelNet10);

# ╔═╡ fae811bb-55ee-4300-9e63-f52fef455b1f
qClassNRCDT = NRCDT.RCDT.(AClass);

# ╔═╡ 08c9d81b-206c-4d73-b0be-eb3361873010
pₜ = Plots.heatmap(qTempNRCDT[1])

# ╔═╡ da8a20b8-fa41-4b4e-88c6-426bba656d34
# ╠═╡ disabled = true
#=╠═╡
Plots.savefig(pₜ, "rcdt_c_128_128_t1.pdf")
  ╠═╡ =#

# ╔═╡ 2456fa7b-e1ab-41f3-abfc-16d11bc26636
pₜₜ = Plots.heatmap(qClassNRCDT[1])

# ╔═╡ bca4915d-9da6-4929-98b2-dadc17b170b0
# ╠═╡ disabled = true
#=╠═╡
Plots.savefig(pₜₜ, "rcdt_c_128_128_c1.pdf")
  ╠═╡ =#

# ╔═╡ 2d6b2930-f83b-4a60-9a59-8cb7308cf78b
nqTempNRCDT = normalization.(qTempNRCDT)

# ╔═╡ acc458e9-41d4-48be-87f7-199580d2ee10
nqClassNRCDT = normalization.(qClassNRCDT)

# ╔═╡ 4e2f1004-6fbe-4151-94ad-81f428250493
pₙ = Plots.heatmap(nqTempNRCDT[1])

# ╔═╡ b8b48b65-4cff-4e31-be11-3f67522b2322
# ╠═╡ disabled = true
#=╠═╡
Plots.savefig(pₙ, "nrcdt_c_128_128_t1.pdf")
  ╠═╡ =#

# ╔═╡ dd38aed4-157f-451e-90a0-811a9d680bbc
pₙₙ = Plots.heatmap(nqClassNRCDT[1])

# ╔═╡ ea35d682-b34f-463c-995b-011c2673a061
# ╠═╡ disabled = true
#=╠═╡
Plots.savefig(pₙₙ, "nrcdt_c_128_128_c1.pdf")
  ╠═╡ =#

# ╔═╡ 974bde48-c1de-4011-a3cc-c3a406c6904a
begin
	mnqTempNRCDT = maxnormalization.(qTempNRCDT)
	mnqClassNRCDT = maxnormalization.(qClassNRCDT)
	
	anqTempNRCDT = meannormalization.(qTempNRCDT)
	anqClassNRCDT = meannormalization.(qClassNRCDT)
end

# ╔═╡ 6cce609b-fc68-4eb9-a056-e9d75c2cc1e0
begin
	Plots.plot(mnqTempNRCDT[1])
	Plots.plot!(mnqClassNRCDT[1:10])
end

# ╔═╡ 5b4e1589-a52e-4977-93d1-926efa0c1d3a
begin
	Plots.plot(mnqTempNRCDT[2])
	Plots.plot!(mnqClassNRCDT[11:20])
end

# ╔═╡ c14d0e46-9e1d-4652-8f5a-1ffc68a73293
begin
	Plots.plot(mnqTempNRCDT[3])
	Plots.plot!(mnqClassNRCDT[21:30])
end

# ╔═╡ 1f273695-668a-4d97-8f8f-b386cef093f3
begin
	Plots.plot(mnqTempNRCDT[4])
	Plots.plot!(mnqClassNRCDT[31:40])
end

# ╔═╡ 04a755b0-c405-407a-9a8a-b1380c1fa259
begin
	Plots.plot(mnqTempNRCDT[5])
	Plots.plot!(mnqClassNRCDT[41:50])
end

# ╔═╡ c961df5d-4156-4295-9465-21588e5b01bc
begin
	Plots.plot(mnqTempNRCDT[6])
	Plots.plot!(mnqClassNRCDT[51:60])
end

# ╔═╡ 3e9ba4cb-8384-4c0d-adcf-e00244ac7f5a
begin
	Plots.plot(mnqTempNRCDT[7])
	Plots.plot!(mnqClassNRCDT[61:70])
end

# ╔═╡ 0c22db67-7c57-4795-9506-c0c2d018f644
begin
	Plots.plot(mnqTempNRCDT[8])
	Plots.plot!(mnqClassNRCDT[71:80])
end

# ╔═╡ 4a995a31-c010-4953-aa02-0b58fef41596
begin
	Plots.plot(mnqTempNRCDT[9])
	Plots.plot!(mnqClassNRCDT[81:90])
end

# ╔═╡ ebc7894c-49eb-4a05-b4fa-af3f99be1ed0
begin
	Plots.plot(mnqTempNRCDT[10])
	Plots.plot!(mnqClassNRCDT[91:100])
end

# ╔═╡ f7114e5e-b407-4952-8f3f-1c3fcaef4a2e
begin
	function distance_map(quantiles::AbstractArray,quantiles_temp::AbstractArray,ord::Int64)
		n = length(quantiles)
		m = length(quantiles_temp)
		confmap = zeros(n, m)
		for k in 1:n
			for j in 1:m
				confmap[k, j] = sum(abs.(quantiles[k] .- quantiles_temp[j]).^ord)^(1/ord)
			end
		end
		return confmap
	end
	
	function confusion_map(quantiles::AbstractArray,ord::Int64)
		n = length(quantiles)
		confmap = zeros(n, n)
		for k in 1:n
			for j in 1:n
				confmap[k, j] = sum(abs.(quantiles[k] .- quantiles[j]).^ord)^(1/ord)
			end
		end
		return confmap
	end
end

# ╔═╡ 0775010e-a56c-4a36-81c9-d461d199cf74
begin
	DM₁ = distance_map(mnqClassNRCDT, mnqTempNRCDT, 1)
	DM₂ = distance_map(mnqClassNRCDT, mnqTempNRCDT, 2)
	DM∞ = distance_map(mnqClassNRCDT, mnqTempNRCDT, 100)

	DA₁ = distance_map(anqClassNRCDT, anqTempNRCDT, 1)
	DA₂ = distance_map(anqClassNRCDT, anqTempNRCDT, 2)
	DA∞ = distance_map(anqClassNRCDT, anqTempNRCDT, 100)
end

# ╔═╡ 3f39ad0f-aed1-4d67-9731-42914d54ecdf
begin
	category_names = ["Bath Tub", "Bed", "Chair", "Desk", "Dresser", "Monitor", "Night Stand", "Sofa", "Table", "Toilet"]
	y_labels_moved = [n * "     " for n in category_names]
	block_size = 10
	midpoints = [ (i - 1) * block_size + (block_size + 1) / 2 for i in 1:length(category_names) ]
	shift_amount = 2.0 
	x_coords = midpoints .+ shift_amount  # Move right
	y_coords = midpoints .- shift_amount  # Move down
end

# ╔═╡ 6189c160-fd6d-4284-a932-20d7239877a1
begin
	pdm₁ = Plots.heatmap(DM₁, 
	fontfamily="Computer Modern",
	xticks = (1:10, category_names), # Apply custom positions and labels
    xrotation = 45,                       # Rotate labels diagonally
    yticks = (midpoints, category_names), # Same for Y-axis if needed
    yrotation = 45,
	halign = :right,
 	size = (500, 400),)
	#Plots.savefig(pdm₁, "dm_1_128_128.pdf")
end

# ╔═╡ 87f4e467-e510-4b78-9124-5f64a43bce8b
begin
	pdm₂ = Plots.heatmap(DM₂, 
	fontfamily="Computer Modern",
	xticks = (1:10, category_names), # Apply custom positions and labels
    xrotation = 45,                       # Rotate labels diagonally
    yticks = (midpoints, category_names), # Same for Y-axis if needed
    yrotation = 45,
	halign = :right,
 	size = (500, 400),)
	#Plots.savefig(pdm₂, "dm_2_128_128.pdf")
end

# ╔═╡ aa46a01c-6969-4881-a53c-c95b20514636
begin
	pdm∞ = Plots.heatmap(DM∞, 
	fontfamily="Computer Modern",
	xticks = (1:10, category_names), # Apply custom positions and labels
    xrotation = 45,                       # Rotate labels diagonally
    yticks = (midpoints, category_names), # Same for Y-axis if needed
    yrotation = 45,
	halign = :right,
 	size = (500, 400),)
	#Plots.savefig(pdm∞, "dm_∞_128_128.pdf")
end

# ╔═╡ 1f44e79e-c03a-463b-96e4-9db58a5e341b
begin
	pda₁ = Plots.heatmap(DA₁, 
	fontfamily="Computer Modern",
	xticks = (1:10, category_names), # Apply custom positions and labels
    xrotation = 45,                       # Rotate labels diagonally
    yticks = (midpoints, category_names), # Same for Y-axis if needed
    yrotation = 45,
	halign = :right,
 	size = (500, 400),)
	#Plots.savefig(pda₁, "da_1_128_128.pdf")
end

# ╔═╡ 914156f3-6367-462c-9566-88e243350fb4
begin
	pda₂ = Plots.heatmap(DA₂, 
	fontfamily="Computer Modern",
	xticks = (1:10, category_names), # Apply custom positions and labels
    xrotation = 45,                       # Rotate labels diagonally
    yticks = (midpoints, category_names), # Same for Y-axis if needed
    yrotation = 45,
	halign = :right,
 	size = (500, 400),)
	#Plots.savefig(pda₂, "da_2_128_128.pdf")
end

# ╔═╡ 0dcba065-8d28-4106-bc1c-9911226fbd79
begin
	pda∞ = Plots.heatmap(DA∞, 
	fontfamily="Computer Modern",
	xticks = (1:10, category_names), # Apply custom positions and labels
    xrotation = 45,                       # Rotate labels diagonally
    yticks = (midpoints, category_names), # Same for Y-axis if needed
    yrotation = 45,
	halign = :right,
 	size = (500, 400),)
	#Plots.savefig(pda∞, "da_∞_256_128.pdf")
end

# ╔═╡ 67372c90-c5df-4efd-bcac-824c0527241e
begin
	CM₁ = confusion_map(mnqClassNRCDT, 1);
	CM₂ = confusion_map(mnqClassNRCDT, 2);
	CM∞ = confusion_map(mnqClassNRCDT, 100);

	CA₁ = confusion_map(anqClassNRCDT, 1);
	CA₂ = confusion_map(anqClassNRCDT, 2);
	CA∞ = confusion_map(anqClassNRCDT, 100);
end

# ╔═╡ 45b69587-1056-4268-affc-d7a6e8475208
begin
	pcm₁ = Plots.heatmap(CM₁, 
	fontfamily="Computer Modern",
	xticks = (midpoints, category_names), # Apply custom positions and labels
    xrotation = 45,                       # Rotate labels diagonally
    yticks = (midpoints, category_names), # Same for Y-axis if needed
    yrotation = 45,
	halign = :right,
 	size = (500, 400),)
	#Plots.savefig(pcm₁, "cm_1_128_128.pdf")
end

# ╔═╡ cf87c9a1-f9c3-4d02-a984-90d5a5506779
begin
	pcm₂ = Plots.heatmap(CM₂, 
	fontfamily="Computer Modern",
	xticks = (midpoints, category_names), # Apply custom positions and labels
    xrotation = 45,                       # Rotate labels diagonally
    yticks = (midpoints, category_names), # Same for Y-axis if needed
    yrotation = 45,
	halign = :right,
 	size = (500, 400),)
	#Plots.savefig(pcm₂, "cm_2_128_128.pdf")
end

# ╔═╡ fa13a0bb-a54b-4696-8598-2c5acd3995f9
begin
	pcm∞ = Plots.heatmap(CM∞, 
	fontfamily="Computer Modern",
	xticks = (midpoints, category_names), # Apply custom positions and labels
    xrotation = 45,                       # Rotate labels diagonally
    yticks = (midpoints, category_names), # Same for Y-axis if needed
    yrotation = 45,
	halign = :right,
 	size = (500, 400),)
	#Plots.savefig(pcm∞, "cm_∞_128_128.pdf")
end

# ╔═╡ 37ed9041-c2ab-48ee-83a3-2a4a6c156083
begin
	pca₁ = Plots.heatmap(CA₁, 
	fontfamily="Computer Modern",
	xticks = (midpoints, category_names), # Apply custom positions and labels
    xrotation = 45,                       # Rotate labels diagonally
    yticks = (midpoints, category_names), # Same for Y-axis if needed
    yrotation = 45,
	halign = :right,
 	size = (500, 400),)
	#Plots.savefig(pca₁, "ca_1_128_128.pdf")
end

# ╔═╡ 3dce15c5-062f-4be5-8bce-856622f1b527
begin
	pca₂ = Plots.heatmap(CA₂, 
	fontfamily="Computer Modern",
	xticks = (midpoints, category_names), # Apply custom positions and labels
    xrotation = 45,                       # Rotate labels diagonally
    yticks = (midpoints, category_names), # Same for Y-axis if needed
    yrotation = 45,
	halign = :right,
 	size = (500, 400),)
	#Plots.savefig(pca₂, "ca_2_128_128.pdf")
end

# ╔═╡ 2543cc8c-1631-4f76-ac17-7422544f723b
begin
	pca∞ = Plots.heatmap(CA∞, 
	fontfamily="Computer Modern",
	xticks = (midpoints, category_names), # Apply custom positions and labels
    xrotation = 45,                       # Rotate labels diagonally
    yticks = (midpoints, category_names), # Same for Y-axis if needed
    yrotation = 45,
	halign = :right,
 	size = (500, 400),)
	#Plots.savefig(pca∞, "ca_∞_128_128.pdf")
end

# ╔═╡ 32223c3f-da6b-46ab-8c25-8b62d1a253bb
md"""
## Nearest Template and Neighbour Classification
Use the nearest template or neighbour classification
with respect to the chosen templates
to classify the generated dataset.
The normalized RCDT is applied
with different numbers of used angles, normes, and variantes.
"""

# ╔═╡ 92496979-16b9-47b6-8eb1-7a68aef851ef
begin
	function plot_quantiles(temp_q::AbstractArray, temp_lab::AbstractArray, data_q::AbstractArray, data_lab::AbstractArray)
    dd = length(data_q[1])
	    plt = Plots.plot(size = (500,350))
	    color = [:blue, :red, :green, :orange, :purple, :brown]
	    for i in 1:length(temp_lab), j in 1:length(data_lab)
	        if data_lab[j] == temp_lab[i]
	            Plots.plot!(plt, data_q[j], label=false, linecolor = RGBA((abs(0.5-(i-1)/(length(temp_lab)-1)))*(1-(i-1)/(length(temp_lab)-1))*0.99, ((i-1)/(length(temp_lab)-1))*(1-(i-1)/(length(temp_lab)-1))*0.99, (abs(0.5-(i-1)/(length(temp_lab)-1)))*((i-1)/(length(temp_lab)-1))*0.99, 0.5), yticks=true, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
	            # plot!(plt, data_q[j], label=false, linecolor = color[i], yticks=true, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
	        end
	    end
	    for i in 1:length(temp_lab)
	        k = temp_lab[i]
	        Plots.plot!(plt, temp_q[i], fontfamily="Computer Modern", label=["class $k" "k"], linewidth=2 , linecolor = RGBA((abs(0.5-(i-1)/(length(temp_lab)-1)))*(1-(i-1)/(length(temp_lab)-1))*0.99, ((i-1)/(length(temp_lab)-1))*(1-(i-1)/(length(temp_lab)-1))*0.85, (abs(0.5-(i-1)/(length(temp_lab)-1)))*((i-1)/(length(temp_lab)-1))*0.99, 1), yticks=true, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));    
	        # plot!(plt, temp_q[i], legend=false, fontfamily="Computer Modern", label=["Class $k" "k"], linewidth=2 , linecolor = color[i], yticks=true, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));        
	    end
	    return plt
	end

	function accuracy_nearest_neighbour(temp_q::AbstractArray, temp_lab::AbstractArray, data_q::AbstractArray, data_lab::AbstractArray, norm::String; ret::Int64=0)
	    pred_rcdt = zeros(length(temp_q),length(data_q))
	    for k in 1:length(data_q), kk in 1:length(temp_q)
	        if norm=="inf"
	            pred_rcdt[kk,k] = maximum(abs.(temp_q[kk] .- data_q[k]))
	        end
	        if norm=="euclidean"
	            pred_rcdt[kk,k] = sqrt(sum((temp_q[kk] .- data_q[k]).*(temp_q[kk] .- data_q[k])))
	        end
	        if norm=="manhattan"
	            pred_rcdt[kk,k] = sum(abs.(temp_q[kk] .- data_q[k]))
	        end
	    end
	    pred_label_rcdt = argmin(pred_rcdt, dims=1)
	    label_rcdt = zeros(length(data_q))
	    for k in 1:length(data_q)
	        label_rcdt[k] = temp_lab[pred_label_rcdt[k][1]]
	    end 
	    acc_rcdt = mean(data_lab .== label_rcdt)
	    if ret==1
	        @info "Acc. using $(norm)-norm : \t $(acc_rcdt)"
	    end
	    return acc_rcdt
	end

	function argmink(a::AbstractArray, k::Int64)
	    b = partialsortperm(a, length(a)-k+1:length(a), rev=true)
	    # return collect(zip(b, a[b]))
	    return collect(b[end:-1:1])
	end
	
	function argmaxk(a::AbstractArray, k::Int64)
	    b = partialsortperm(a, 1:k, rev=true)
	    # return collect(zip(b, a[b]))
	    return collect(b)
	end
	
	function slice_matrix(A::AbstractMatrix)
	    return [c[:] for c in eachcol(A)]
	end
	
	function accuracy_k_nearest_neighbour(
	    temp_q::AbstractArray, 
	    temp_lab::AbstractArray, 
	    data_q::AbstractArray, 
	    data_lab::AbstractArray,
	    norm::String;
	    K::Int64=1, 
	    ret::Int64=0)
	
	    temp_q = collect.(Iterators.flatten.(collect.(temp_q)))
	    data_q = collect.(Iterators.flatten.(collect.(data_q)))
	
	    pred_rcdt = zeros(length(temp_q),length(data_q))
	    for k in 1:length(data_q), kk in 1:length(temp_q)
	        if norm=="inf"
	            pred_rcdt[kk,k] = maximum(abs.(temp_q[kk] .- data_q[k]))
	        end
	        if norm=="euclidean"
	            pred_rcdt[kk,k] = sqrt(sum((temp_q[kk] .- data_q[k]).*(temp_q[kk] .- data_q[k])))
	        end
	        if norm=="manhattan"
	            pred_rcdt[kk,k] = sum(abs.(temp_q[kk] .- data_q[k]))
	        end
	    end
	    # pred_label_rcdt = argmin(pred_rcdt, dims=1)
	    K_pred_index_rcdt = argmink.(slice_matrix(pred_rcdt), K)
	    K_pred_label_rcdt = zeros(length(K_pred_index_rcdt),K)
	    for l in 1:K, ll in 1:length(K_pred_index_rcdt)
	        K_pred_label_rcdt[ll,l] = temp_lab[K_pred_index_rcdt[ll][l]]
	    end
	    dict_K_pred_label_rcdt = countmap.(slice_matrix(transpose(Int64.(K_pred_label_rcdt))))
	    pred_label_rcdt = findmax.(dict_K_pred_label_rcdt) 
	    label_rcdt = zeros(length(pred_label_rcdt))
	    for k in 1:length(data_q)
	        label_rcdt[k] = pred_label_rcdt[k][2]
	    end 
	    acc_rcdt = mean(data_lab .== label_rcdt)
	    # conf_matrix = confusion_matrix(data_lab, label_rcdt) 
	    if ret==1
	        @info "Acc. using $(norm)-norm : \t $(acc_rcdt)"
	    end
	    return acc_rcdt #, conf_matrix
	end

	function accuracy_k_nearest_part_neighbour(
	    part::Int64, 
	    samp::Int64, 
	    class_size::Int64, 
	    class_number::Int64, 
	    data_q::AbstractArray, 
	    data_lab::AbstractArray, 
	    norm::String; 
	    K::Int64=1,
	    ret::Int64=0)
	
	    # variabel cross <-- TO DO
	
	    if samp > class_size
	        return "Error : unexpected devision of classes."
	    end
	    if ret==1
	        conf_matrix = zeros(length(unique(data_lab)), length(unique(data_lab)))
	    end
	
	    acc_rcdt = zeros(part)
	    size_data = length(data_lab)
	    
	    for i in 1:part
	        split_range = []
	        for l in 1:class_number
	            choice = collect(1+(l-1)*class_size:l*class_size)
	            shuffle!(choice)
	            append!(split_range, choice[1:samp])
	        end

			train_data_q = data_q[split_range]
	        train_labels = data_lab[split_range]
	
	        test_range = Array(1:size_data)
	        test_range = filter(e->!(e in split_range),test_range)
	
	        test_data_q = data_q[test_range]
	        test_labels = data_lab[test_range]
	
	        # acc_rcdt[i], conf_matrix_part = accuracy_k_nearest_neighbour(train_data_q, train_labels, test_data_q, test_labels, norm, K=K)
	        acc_rcdt[i] = accuracy_k_nearest_neighbour(train_data_q, train_labels, test_data_q, test_labels, norm, K=K)
	        if ret==1
	            conf_matrix += conf_matrix_part 
	        end
	    end
	    @info "Acc. using $(norm)-norm : \t $(mean(acc_rcdt)) +/- $(std(acc_rcdt))"
	    if ret==1
	        return conf_matrix
	    end
	end
end

# ╔═╡ 51942786-6a6c-4cd6-8baf-681936516ad1
begin
	accuracy_nearest_neighbour(qTempNRCDT, Label, qClassNRCDT, Labels, "manhattan", ret=1);
	accuracy_nearest_neighbour(qTempNRCDT, Label, qClassNRCDT, Labels, "euclidean", ret=1);
	accuracy_nearest_neighbour(qTempNRCDT, Label, qClassNRCDT, Labels, "inf", ret=1);
end

# ╔═╡ cfa71e36-b75b-47f4-b073-fcac156fc005
begin
	accuracy_nearest_neighbour(mnqTempNRCDT, Label, mnqClassNRCDT, Labels, "manhattan", ret=1);
	accuracy_nearest_neighbour(mnqTempNRCDT, Label, mnqClassNRCDT, Labels, "euclidean", ret=1);
	accuracy_nearest_neighbour(mnqTempNRCDT, Label, mnqClassNRCDT, Labels, "inf", ret=1);
end

# ╔═╡ 7b98dd59-2c47-42ce-adfe-29b207f4c6fe
begin
	accuracy_nearest_neighbour(anqTempNRCDT, Label, anqClassNRCDT, Labels, "manhattan", ret=1)
	accuracy_nearest_neighbour(anqTempNRCDT, Label, anqClassNRCDT, Labels, "euclidean", ret=1)
	accuracy_nearest_neighbour(anqTempNRCDT, Label, anqClassNRCDT, Labels, "inf", ret=1)
end

# ╔═╡ 0306c9e1-528f-4652-9c31-28762b8d23d1
begin
	accuracy_k_nearest_part_neighbour(20, 1, 10, 10, qClassNRCDT, Labels, "manhattan", K=1, ret=0)
	accuracy_k_nearest_part_neighbour(20, 1, 10, 10, qClassNRCDT, Labels, "euclidean", K=1, ret=0)
	accuracy_k_nearest_part_neighbour(20, 1, 10, 10, qClassNRCDT, Labels, "inf", K=1, ret=0)
end

# ╔═╡ a6c48107-e019-412f-88b2-e14057ec42f3
begin
	K = 5
	accuracy_k_nearest_part_neighbour(20, K, 10, 10, mnqClassNRCDT, Labels, "manhattan", K=1, ret=0)
	accuracy_k_nearest_part_neighbour(20, K, 10, 10, mnqClassNRCDT, Labels, "euclidean", K=1, ret=0)
	accuracy_k_nearest_part_neighbour(20, K, 10, 10, mnqClassNRCDT, Labels, "inf", K=1, ret=0)
end

# ╔═╡ db826ec5-58b4-4ff3-84c7-ad62d1c91d70
begin
	accuracy_k_nearest_part_neighbour(20, 1, 10, 10, anqClassNRCDT, Labels, "manhattan", K=1, ret=0)
	accuracy_k_nearest_part_neighbour(20, 1, 10, 10, anqClassNRCDT, Labels, "euclidean", K=1, ret=0)
	accuracy_k_nearest_part_neighbour(20, 1, 10, 10, anqClassNRCDT, Labels, "inf", K=1, ret=0)
end

# ╔═╡ ebc40d73-6ae0-4dea-bbea-8618fa136bed
pq14 = plot_quantiles(anqTempNRCDT[[1,4]], [1,4], anqClassNRCDT[vcat(1:10,31:40)], Labels[vcat(1:10,31:40)])

# ╔═╡ 1e415a55-0089-45c2-a8a0-592bdc8424b3
# ╠═╡ disabled = true
#=╠═╡
Plots.savefig(pq14, "a_quantiles_1_4_128_128.pdf")
  ╠═╡ =#

# ╔═╡ dbad2a0d-ac54-4d35-90bc-48613b7bd130
begin
	function aargmin(D)
		l = []
		q = argmin(D, dims=2)
		for k in 1:length(q)
			l = vcat(l, q[k][2])
		end
		return l
	end

	adm₁ = aargmin(DM₁)
	adm₂ = aargmin(DM₂)
	adm∞ = aargmin(DM∞)

	ada₁ = aargmin(DA₁)
	ada₂ = aargmin(DA₂)
	ada∞ = aargmin(DA∞)
end

# ╔═╡ d6e844b8-6cdb-4888-a4af-f4ece20a7c48
begin
	Plots.plot(Labels, labels="True Labeling", fontfamily="Computer Modern")
	Plots.plot!(adm₁, labels="1-Norm", fontfamily="Computer Modern")
	Plots.plot!(adm₂, labels="2-Norm", fontfamily="Computer Modern")
	ppm = Plots.plot!(adm∞, labels="∞-Norm", fontfamily="Computer Modern")
end

# ╔═╡ c77c2478-73fd-47a9-a0a1-713e534ec712
begin
	Plots.plot(Labels, labels="True Labeling", fontfamily="Computer Modern")
	Plots.plot!(ada₁, labels="1-Norm", fontfamily="Computer Modern")
	Plots.plot!(ada₂, labels="2-Norm", fontfamily="Computer Modern")
	ppa = Plots.plot!(ada∞, labels="∞-Norm", fontfamily="Computer Modern")
end

# ╔═╡ 23c841ef-d77b-4f38-81be-2fdb22c784db
# ╠═╡ disabled = true
#=╠═╡
Plots.savefig(ppa, "anRCDT_labeling_256.pdf")
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╠═eb9a0617-8519-4ce4-a611-485bcbd723fb
# ╠═c20ec051-9ae9-40f7-97cb-7abf2baf4fe4
# ╠═1fa304a4-ecf5-4645-8f3a-90aedde4323e
# ╠═178b1a7f-ceeb-43ba-9006-e2c5fd7f6b05
# ╠═62aa8854-c7b6-4da4-a904-22d56ed5364c
# ╠═3208b1c3-d726-4027-aee9-5875b0f79965
# ╠═4a94351b-d4d4-4a34-9505-7a40697c6704
# ╠═7de17916-6d95-44d7-959b-a48f504f8f07
# ╠═085e2183-97b3-4a7a-873a-560e9e305f36
# ╠═5c07bf3d-b1bc-499f-8c6f-60821fe19cef
# ╠═d8c1637c-3d72-4124-8267-79b036051ca2
# ╠═9e927481-a86b-486d-967f-57786d5c36a1
# ╠═8a45a3b5-1005-407a-b51d-3a0369beff0f
# ╠═9fa792fc-a766-4c0c-bf97-71034c790cd5
# ╠═7df0a0ed-762b-48cf-b7d8-1ab8d3f86076
# ╠═3a2fbfdf-5aa6-4a5a-b58a-b6ac49d1d076
# ╠═9a88aa43-f6b6-458a-939b-a91510bdf571
# ╠═a6fcea5b-3f86-42ac-b104-f4709ba2ee95
# ╠═8d783205-ffa0-4ac6-957c-706b7505806b
# ╠═0c46cd8e-57eb-46fe-8426-ef2aa69a2c56
# ╠═5db931a4-f5ef-4f0e-8729-3648c119147f
# ╠═9359f6f1-ae2c-4313-b480-ffc433e98eb4
# ╠═aa720f14-298a-45df-a6f3-921fd5f77ac2
# ╠═4beb1c43-5d2e-4580-a740-ac4edb625e17
# ╠═654dc837-3c6d-4b87-8ccb-6cbfadd2bb78
# ╠═fae811bb-55ee-4300-9e63-f52fef455b1f
# ╠═08c9d81b-206c-4d73-b0be-eb3361873010
# ╠═da8a20b8-fa41-4b4e-88c6-426bba656d34
# ╠═2456fa7b-e1ab-41f3-abfc-16d11bc26636
# ╠═bca4915d-9da6-4929-98b2-dadc17b170b0
# ╠═2d6b2930-f83b-4a60-9a59-8cb7308cf78b
# ╠═acc458e9-41d4-48be-87f7-199580d2ee10
# ╠═4e2f1004-6fbe-4151-94ad-81f428250493
# ╠═b8b48b65-4cff-4e31-be11-3f67522b2322
# ╠═dd38aed4-157f-451e-90a0-811a9d680bbc
# ╠═ea35d682-b34f-463c-995b-011c2673a061
# ╠═974bde48-c1de-4011-a3cc-c3a406c6904a
# ╠═6cce609b-fc68-4eb9-a056-e9d75c2cc1e0
# ╠═5b4e1589-a52e-4977-93d1-926efa0c1d3a
# ╠═c14d0e46-9e1d-4652-8f5a-1ffc68a73293
# ╠═1f273695-668a-4d97-8f8f-b386cef093f3
# ╠═04a755b0-c405-407a-9a8a-b1380c1fa259
# ╠═c961df5d-4156-4295-9465-21588e5b01bc
# ╠═3e9ba4cb-8384-4c0d-adcf-e00244ac7f5a
# ╠═0c22db67-7c57-4795-9506-c0c2d018f644
# ╠═4a995a31-c010-4953-aa02-0b58fef41596
# ╠═ebc7894c-49eb-4a05-b4fa-af3f99be1ed0
# ╠═f7114e5e-b407-4952-8f3f-1c3fcaef4a2e
# ╠═0775010e-a56c-4a36-81c9-d461d199cf74
# ╠═3f39ad0f-aed1-4d67-9731-42914d54ecdf
# ╠═6189c160-fd6d-4284-a932-20d7239877a1
# ╠═87f4e467-e510-4b78-9124-5f64a43bce8b
# ╠═aa46a01c-6969-4881-a53c-c95b20514636
# ╠═1f44e79e-c03a-463b-96e4-9db58a5e341b
# ╠═914156f3-6367-462c-9566-88e243350fb4
# ╠═0dcba065-8d28-4106-bc1c-9911226fbd79
# ╠═67372c90-c5df-4efd-bcac-824c0527241e
# ╠═45b69587-1056-4268-affc-d7a6e8475208
# ╠═cf87c9a1-f9c3-4d02-a984-90d5a5506779
# ╠═fa13a0bb-a54b-4696-8598-2c5acd3995f9
# ╠═37ed9041-c2ab-48ee-83a3-2a4a6c156083
# ╠═3dce15c5-062f-4be5-8bce-856622f1b527
# ╠═2543cc8c-1631-4f76-ac17-7422544f723b
# ╠═32223c3f-da6b-46ab-8c25-8b62d1a253bb
# ╠═92496979-16b9-47b6-8eb1-7a68aef851ef
# ╠═51942786-6a6c-4cd6-8baf-681936516ad1
# ╠═cfa71e36-b75b-47f4-b073-fcac156fc005
# ╠═7b98dd59-2c47-42ce-adfe-29b207f4c6fe
# ╠═0306c9e1-528f-4652-9c31-28762b8d23d1
# ╠═a6c48107-e019-412f-88b2-e14057ec42f3
# ╠═db826ec5-58b4-4ff3-84c7-ad62d1c91d70
# ╠═ebc40d73-6ae0-4dea-bbea-8618fa136bed
# ╠═1e415a55-0089-45c2-a8a0-592bdc8424b3
# ╠═dbad2a0d-ac54-4d35-90bc-48613b7bd130
# ╠═d6e844b8-6cdb-4888-a4af-f4ece20a7c48
# ╠═c77c2478-73fd-47a9-a0a1-713e534ec712
# ╠═23c841ef-d77b-4f38-81be-2fdb22c784db
