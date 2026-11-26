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

# ╔═╡ 7a209b02-782f-4b79-a4b4-d89029206c23
using FileIO, MeshIO, GLMakie, PlotlyJS, JLD2 # CairoMakie

# ╔═╡ 0791c621-08c5-4772-bbcb-906957a721be
using Markdown, InteractiveUtils

# ╔═╡ b67c6b62-8cec-4f62-8709-07ddc17354d3
using Interpolations: LinearInterpolation

# ╔═╡ dde2b58d-f858-4201-8fa8-94165d256b66
using Statistics, Random, LinearAlgebra

# ╔═╡ 2d6b3f00-18f6-4500-820f-2c931ddb7dd1
using FFTW

# ╔═╡ fa74a3c9-24a0-41ab-951e-dd35755808f0
using MultivariateStats

# ╔═╡ 213c7f64-bd0e-4485-aed0-f36d2a398f66
using Clustering

# ╔═╡ e3fbd24c-7e4d-4841-b028-53fda048ac00
using StatsBase

# ╔═╡ 80b0dd3a-69ae-48df-abac-c8de5625fe2e
md"""
# arXiv:xxxx.xxxxx -- Section III.E
This pluto notebook reproduces the numerical visualizations
with tamplates for Figure 4 and 5 (first template, affinely transformed) from

- Robert Beinert, Jonas Bresch, and Michael Quellmalz
  '*A Discrete Radon Transform Based on the Area of Cube-Plane Intersection*',
  arXiv:xxxx.xxxxx, 2026.
"""

# ╔═╡ 1c634e17-e792-47ec-bab8-2bcdb342bed7
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

# ╔═╡ 010cf880-a1c8-49f2-bad2-80fcfecbc7ce
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

# ╔═╡ c86866df-5730-4f2a-86d0-88e4b9bdba1a
begin
	smallModelNet10 = [voxel_bathtub[1],voxel_bed[1],voxel_chair[1],voxel_desk[1],voxel_dresser[1],voxel_monitor[1],voxel_night_stand[1],voxel_sofa[1],voxel_table[1],voxel_toilet[1]]
	Label = 1:10
end

# ╔═╡ 06c66cd3-1324-4592-930e-73587ec0c5b6
begin
	function extend_3d(A::AbstractArray,d::Int64)
		dd = max(d,128)
		voxel_array = zeros(dd,dd,dd)
		voxel_array[dd÷4:dd÷4+63,dd÷4:dd÷4+63,dd÷4:dd÷4+63] = A
		return voxel_array
	end
	ModelNet10 = extend_3d.(smallModelNet10,128)
end

# ╔═╡ 7cd2d106-cbe1-4ef6-8efc-a3340d059793
md"""
## Template Dataset
Generate the template dataset
using the submodule `TestImages`.
"""

# ╔═╡ 7b8ca11e-b3a8-4455-8811-c74097d2d50f
Class, Labels = generate_academic_classes_3d(ModelNet10, Label, class_size=10);

# ╔═╡ 5f8ec9ae-d81c-4335-85de-f15048d9bbcf
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

# ╔═╡ 41f9d8c0-5cb5-4068-b253-033147dd3449
AClass = A.(Class)

# ╔═╡ 05c285a0-328a-48f4-bb43-888f4d180339
for k in 1:length(AClass)
	AClass[k][isnan.(AClass[k])] .= 0.0;
	k += 1
end

# ╔═╡ 03fc0177-5d3d-4ff0-abe9-3ba7c9e11740
begin
	PNet = Phantom.(ModelNet10)
	PClass = Phantom.(AClass)
end

# ╔═╡ 5999b3b7-7a53-40a2-b8df-1e05887e91ac
md"""
### Help functions for the Daras-style preprocessing
"""

# ╔═╡ a6d430f2-1f9b-4b44-9e22-2e064849aa12
begin
	function voxel_centroid(vol)
	    nx, ny, nz = size(vol)
	
	    cx = cy = cz = 0.0
	    mass = 0.0
	
	    for i in 1:nx, j in 1:ny, k in 1:nz
	        w = vol[i,j,k]
	        if w != 0
	            cx += w * i
	            cy += w * j
	            cz += w * k
	            mass += w
	        end
	    end
	
	    return (cx/mass, cy/mass, cz/mass)
	end

	function shift_volume(vol, dx, dy, dz)
	    nx, ny, nz = size(vol)
	    out = zeros(eltype(vol), size(vol))
	
	    for i in 1:nx, j in 1:ny, k in 1:nz
	        ii = i - dx
	        jj = j - dy
	        kk = k - dz
	
	        if 1 ≤ ii ≤ nx && 1 ≤ jj ≤ ny && 1 ≤ kk ≤ nz
	            out[i,j,k] = vol[ii,jj,kk]
	        end
	    end
	
	    return out
	end

	function center_volume(vol)
	    nx, ny, nz = size(vol)
	    cx, cy, cz = voxel_centroid(vol)
	
	    gx = (nx + 1)/2
	    gy = (ny + 1)/2
	    gz = (nz + 1)/2
	
	    dx = round(Int, cx - gx)
	    dy = round(Int, cy - gy)
	    dz = round(Int, cz - gz)
	
	    return shift_volume(vol, dx, dy, dz)
	end

	function scale_to_unit_sphere(vol)
	    nx, ny, nz = size(vol)
	    cx, cy, cz = (nx+1)/2, (ny+1)/2, (nz+1)/2
	
	    rmax = 0.0
	    for i in 1:nx, j in 1:ny, k in 1:nz
	        if vol[i,j,k] != 0
	            r = sqrt((i-cx)^2 + (j-cy)^2 + (k-cz)^2)
	            rmax = max(rmax, r)
	        end
	    end
	
	    scale = (min(nx,ny,nz)/2 - 1) / rmax
	
	    return rescale_volume(vol, scale)
	end

	function rescale_volume(vol, s)
	    nx, ny, nz = size(vol)
	    cx, cy, cz = (nx+1)/2, (ny+1)/2, (nz+1)/2
	
	    out = zeros(eltype(vol), size(vol))
	
	    for i in 1:nx, j in 1:ny, k in 1:nz
	        ii = round(Int, (i - cx)/s + cx)
	        jj = round(Int, (j - cy)/s + cy)
	        kk = round(Int, (k - cz)/s + cz)
	
	        if 1 ≤ ii ≤ nx && 1 ≤ jj ≤ ny && 1 ≤ kk ≤ nz
	            out[i,j,k] = vol[ii,jj,kk]
	        end
	    end
	
	    return out
	end
	
	function voxel_covariance(vol)
	    nx, ny, nz = size(vol)
	    cx, cy, cz = (nx+1)/2, (ny+1)/2, (nz+1)/2
	
	    C = zeros(3,3)
	    mass = 0.0
	
	    for i in 1:nx, j in 1:ny, k in 1:nz
	        w = vol[i,j,k]
	        if w != 0
	            x = i - cx
	            y = j - cy
	            z = k - cz
	            v = [x,y,z]
	            C .+= w * (v * v')
	            mass += w
	        end
	    end
	
	    return C / mass
	end

	function pca_align_volume(vol)
	    C = voxel_covariance(vol)
	    eig = eigen(Symmetric(C))
	
	    idx = sortperm(eig.values, rev=true)
	    R = eig.vectors[:, idx]
	
	    return rotate_volume(vol, R), R
	end

	function rotate_volume(vol, R)
	    nx, ny, nz = size(vol)
	    cx, cy, cz = (nx+1)/2, (ny+1)/2, (nz+1)/2
	
	    out = zeros(eltype(vol), size(vol))
	
	    for i in 1:nx, j in 1:ny, k in 1:nz
	        x = [i-cx, j-cy, k-cz]
	        y = R' * x
	
	        ii = round(Int, y[1] + cx)
	        jj = round(Int, y[2] + cy)
	        kk = round(Int, y[3] + cz)
	
	        if 1 ≤ ii ≤ nx && 1 ≤ jj ≤ ny && 1 ≤ kk ≤ nz
	            out[i,j,k] = vol[ii,jj,kk]
	        end
	    end
	
	    return out
	end

	function smooth_volume(vol; σ=1.0)
	    kernel = Kernel.gaussian((σ, σ, σ))
	    return imfilter(vol, kernel)
	end

	function axis_flip_matrices()
	    flips = [
	        Diagonal([ 1,  1,  1]),
	        Diagonal([-1, -1,  1]),
	        Diagonal([-1,  1, -1]),
	        Diagonal([ 1, -1, -1])
	    ]
	    return flips
	end

	function pca_align_volume_with_flips(vol)
	    C = voxel_covariance(vol)
	    eig = eigen(Symmetric(C))
	
	    idx = sortperm(eig.values, rev=true)
	    R0 = eig.vectors[:, idx]
	
	    flips = axis_flip_matrices()
	
	    aligned_volumes = Vector{Array{eltype(vol),3}}()
	    rotations = Matrix{Float64}[]
	
	    for F in flips
	        R = R0 * F
	        push!(aligned_volumes, rotate_volume(vol, R))
	        push!(rotations, R)
	    end
	
	    return aligned_volumes, rotations
	end

	function daras_preprocess_volume(vol;
                                 smoothing=true,
                                 σ=1.0)

	    v = copy(vol)
	
	    v = center_volume(v)
	    v = scale_to_unit_sphere(v)
	
	    vols, rotations = pca_align_volume_with_flips(v)
	
	    if smoothing
	        vols = [smooth_volume(vv, σ=σ) for vv in vols]
	    end
	
	    return vols, rotations
	end
end

# ╔═╡ a91c4339-41f2-469c-9e55-50c1732906ec
md"""
## Daras-style preprocessing with optional PCA alignment.
Further information can be found in

- Daras, P. and Zarpalas, D. and Tzovaras, D. and Strintzis, M.G., 
  '[Shape matching using the 3D Radon transform]
  (https://doi.org/10.1109/TDPVT.2004.1335419)',
  *Proceedings. 2nd International Symposium on 3D Data Processing, Visualization and Transmission (3DPVT)*,
  953--960 (2004). 

###### Style giude:

- If pca_align == true:
    1. translate to centroid
    2. PCA-align (NO sign fixing or axis reordering)
    3. uniform scale to unit ball (max radius)
    4. re-voxelize by accumulation
    5. normalize mass

- If pca_align == false:
    behaves like the original preprocess_daras: translate -> scale -> re-voxelize -> normalize

Arguments:
 - voxels :: 3D array (binary or density)
 - grid_size :: Int (output resolution; default 64)
 - pca_align :: Bool (apply PCA alignment? default true)

Returns:
 - output :: Array{Float64,3} of size (grid_size,grid_size,grid_size)
"""

# ╔═╡ 4d629451-66b7-44a9-8693-9f48d5f324a3
function preprocess_daras(voxels::AbstractArray{<:Real,3};
                          grid_size::Int = 64,
                          pca_align::Bool = true)

    # --- 1. Extract occupied voxels ---
    inds = findall(voxels .> 0)
    if isempty(inds)
        return zeros(Float64, grid_size, grid_size, grid_size)
    end

    # Coordinates as 3 × M matrix
    X = hcat([[i[1], i[2], i[3]] for i in inds]...)
    X = Float64.(X)
    M = size(X, 2)

    # --- 2. Translation: centroid ---
    μ = mean(X, dims=2)
    Xc = X .- μ

    # Optionally apply PCA alignment (Daras-style)
    if pca_align
        # covariance / inertia matrix
        C = (Xc * Xc') / max(M, 1)
        eigres = eigen(Symmetric(C))
        # IMPORTANT (paper-faithful): do not reorder or flip eigenvectors here
        R = eigres.vectors
        Xp = R' * Xc        # rotate to principal axes (no sign/order fix)
    else
        Xp = Xc
    end

    # --- 3. Scaling: fit into unit ball (MAX radius) ---
    radii = sqrt.(sum(Xp .^ 2, dims=1))
    rmax = maximum(radii)
    if rmax == 0.0
        rmax = 1.0
    end
    Xs = Xp ./ rmax   # now inside unit ball

    # --- 4. Re-voxelization (accumulate, not overwrite) ---
    out = zeros(Float64, grid_size, grid_size, grid_size)
    half = grid_size ÷ 2

    for i in 1:size(Xs, 2)
        x = Xs[1, i] * half + half
        y = Xs[2, i] * half + half
        z = Xs[3, i] * half + half

        xi = clamp(round(Int, x), 1, grid_size)
        yi = clamp(round(Int, y), 1, grid_size)
        zi = clamp(round(Int, z), 1, grid_size)

        out[xi, yi, zi] += 1.0
    end

    # --- 5. Normalize volume mass ---
    s = sum(out)
    s > 0 && (out ./= s)

    return out
end

# ╔═╡ a2d9776f-3e8a-428f-af93-dd16ff86ad75
# ╠═╡ disabled = true
#=╠═╡
begin
	pvoxel_bathtub = preprocess_daras.(AClass[1:10]; pca_align=true)
	pvoxel_bed = preprocess_daras.(AClass[11:20]; pca_align=true)
	pvoxel_chair = preprocess_daras.(AClass[21:30]; pca_align=true)
	pvoxel_desk = preprocess_daras.(AClass[31:40]; pca_align=true)
	pvoxel_dresser = preprocess_daras.(AClass[41:50]; pca_align=true)
	pvoxel_monitor = preprocess_daras.(AClass[51:60]; pca_align=true)
	pvoxel_night_stand = preprocess_daras.(AClass[61:70]; pca_align=true)
	pvoxel_sofa = preprocess_daras.(AClass[71:80]; pca_align=true)
	pvoxel_table = preprocess_daras.(AClass[81:90]; pca_align=true)
	pvoxel_toilet = preprocess_daras.(AClass[91:100]; pca_align=true)
end
  ╠═╡ =#

# ╔═╡ 13832bcd-831f-445a-bb50-fab64300211f
#=╠═╡
GLMakie.volume(pvoxel_bathtub[1], algorithm=:mip, colormap=:magma)
  ╠═╡ =#

# ╔═╡ 35949af4-e037-4e68-9e54-4cfc3a8c7316
#=╠═╡
GLMakie.volume(pvoxel_bed[1], algorithm=:mip, colormap=:magma)
  ╠═╡ =#

# ╔═╡ 47b98719-a983-4480-ab09-23676f8ee9da
function sphere_angles(d₁::Int64,d₂::Int64)
	θ = zeros((d₁-1)*d₂,3)
	for k in 1:d₁-1
		for j in 1:d₂
			#θ[j+d₂*(k-1),1] = sin(coss(k/d₁*π))*sin(j/d₂*2π)
			#θ[j+d₂*(k-1),2] = sin(coss(k/d₁*π))*cos(j/d₂*2π)
			#θ[j+d₂*(k-1),3] = cos(coss(k/d₁*π))
			θ[j+d₂*(k-1),1] = sin(k/d₁*π)*sin(j/d₂*2π)
			θ[j+d₂*(k-1),2] = sin(k/d₁*π)*cos(j/d₂*2π)
			θ[j+d₂*(k-1),3] = cos(k/d₁*π)
		end
	end
	return θ
end

# ╔═╡ 837b8fa2-b7e8-4850-a513-69c03d35a45b
TT = sphere_angles(20,16)

# ╔═╡ 3daa2232-071d-493c-aa58-f1b8d0cef86a
Plots.scatter3d(TT[:,1], TT[:,2], TT[:,3])

# ╔═╡ a75c6ac8-86a0-479d-aa87-e5bde563d22b
md"""
## Radon transform
Setup the Radon transform in `3d`
Discretization, given by 

- number of equispaced radii in [-√3⋅d/2,√3⋅d/2],
- Fibonacci sphere points (nearly equispaced on the S^2).
"""

# ╔═╡ aff5fbc3-2724-45fd-802b-3b60e509a999
begin
	d = 128
	r = 2048
	radii = collect(LinRange(-sqrt(3)/2*(d-1),sqrt(3)/2*(d-1),r))
	angles = sphere_angles(30,21)
	R = RadonTransform(radii,angles)
end

# ╔═╡ 107302bd-01ad-4787-aa96-61d6e93b08b7
#=╠═╡
begin
	phvoxel_bathtub = Phantom.(pvoxel_bathtub[1:10])
	phvoxel_bed = Phantom.(pvoxel_bed[1:10])
	phvoxel_chair = Phantom.(pvoxel_chair[1:10])
	phvoxel_desk = Phantom.(pvoxel_desk[1:10])
	phvoxel_dresser = Phantom.(pvoxel_dresser[1:10])
	phvoxel_monitor = Phantom.(pvoxel_monitor[1:10])
	phvoxel_night_stand = Phantom.(pvoxel_night_stand[1:10])
	phvoxel_sofa = Phantom.(pvoxel_sofa[1:10])
	phvoxel_table = Phantom.(pvoxel_table[1:10])
	phvoxel_toilet = Phantom.(pvoxel_toilet[1:10])
end
  ╠═╡ =#

# ╔═╡ 96ae91b4-fafd-47e5-b2a9-e19f61b21f84
#=╠═╡
begin
	rvoxel_bathtub = R.(phvoxel_bathtub)
	rvoxel_bed = R.(phvoxel_bed)
	rvoxel_chair = R.(phvoxel_chair)
	rvoxel_desk = R.(phvoxel_desk)
	rvoxel_dresser = R.(phvoxel_dresser)
	rvoxel_monitor = R.(phvoxel_monitor)
	rvoxel_night_stand = R.(phvoxel_night_stand)
	rvoxel_sofa = R.(phvoxel_sofa)
	rvoxel_table = R.(phvoxel_table)
	rvoxel_toilet = R.(phvoxel_toilet)
end
  ╠═╡ =#

# ╔═╡ cbfc4e51-031f-46f3-92fc-32f3335ef22d
#=╠═╡
begin
	svoxel_bathtub = [reshape(r.data, (2048,29,21)) for r in rvoxel_bathtub]
	svoxel_bed = [reshape(r.data, (2048,29,21)) for r in rvoxel_bed]
	svoxel_chair = [reshape(r.data, (2048,29,21)) for r in rvoxel_chair]
	svoxel_desk = [reshape(r.data, (2048,29,21)) for r in rvoxel_desk]
	svoxel_dresser = [reshape(r.data, (2048,29,21)) for r in rvoxel_dresser]
	svoxel_monitor = [reshape(r.data, (2048,29,21)) for r in rvoxel_monitor]
	svoxel_night_stand = [reshape(r.data, (2048,29,21)) for r in rvoxel_night_stand]
	svoxel_sofa = [reshape(r.data, (2048,29,21)) for r in rvoxel_sofa]
	svoxel_table = [reshape(r.data, (2048,29,21)) for r in rvoxel_table]
	svoxel_toilet = [reshape(r.data, (2048,29,21)) for r in rvoxel_toilet]
end
  ╠═╡ =#

# ╔═╡ baa5029f-428f-4e48-8c6a-c31b643fa5cf
#=╠═╡
svoxel = vcat(svoxel_bathtub, svoxel_bed, svoxel_chair, svoxel_desk, svoxel_dresser, svoxel_monitor, svoxel_night_stand, svoxel_sofa, svoxel_table, svoxel_toilet)
  ╠═╡ =#

# ╔═╡ da9fbc60-77eb-4952-acb5-7bba2e1fa045
md"""
## Daras-style descriptors.
Further information can be found in

- Daras, P. and Zarpalas, D. and Tzovaras, D. and Strintzis, M.G., 
  '[Shape matching using the 3D Radon transform]
  (https://doi.org/10.1109/TDPVT.2004.1335419)',
  *Proceedings. 2nd International Symposium on 3D Data Processing, Visualization and Transmission (3DPVT)*,
  953--960 (2004). 

###### Style giude:

- Tab. 1 (51 descriptors) and the full combination of all functions

Input: rt_result :: Array{<:Real,3}  [rho, theta, phi]

Returns:
 - output :: 64-dimensional descriptor
"""

# ╔═╡ 88a06a44-b384-4e41-a3cf-b90130521749
begin
	# --- Functionals (Daras 2004) ---
	F1(g) = maximum(g)
	# F2(g) = length(g) < 2 ? 0.0 : sum(abs.(diff(g)))
	F2(g) = length(g) < 3 ? 0.0 : sum(abs.((g[3:end] .- g[1:end-2]) ./ 2))
	F3(g) = sum(g)
	F4(g) = maximum(g) - minimum(g)
	
	const FUNC = Dict("F1"=>F1, "F2"=>F2, "F3"=>F3, "F4"=>F4)
	
	# --- Reduce along axis for 3D, 2D, 1D ---
	function reduce_axis(A, axis::Int, f)
	    nd = ndims(A)
	
	    if nd == 3
	        if axis == 1
	            T,P = size(A,2), size(A,3)
	            B = zeros(Float64, T, P)
	            for t in 1:T, p in 1:P
	                B[t,p] = f(@view A[:,t,p])
	            end
	            return B
	        elseif axis == 2
	            S,P = size(A,1), size(A,3)
	            B = zeros(Float64, S, P)
	            for r in 1:S, p in 1:P
	                B[r,p] = f(@view A[r,:,p])
	            end
	            return B
	        elseif axis == 3
	            S,T = size(A,1), size(A,2)
	            B = zeros(Float64, S, T)
	            for r in 1:S, t in 1:T
	                B[r,t] = f(@view A[r,t,:])
	            end
	            return B
	        end
	
	    elseif nd == 2
	        if axis == 1
	            return [f(@view A[:,j]) for j in 1:size(A,2)]
	        elseif axis == 2
	            return [f(@view A[i,:]) for i in 1:size(A,1)]
	        end
	
	    elseif nd == 1
	        return f(A)
	
	    else
	        error("Unexpected dimensionality")
	    end
	end
	
	# --- EXACT Table 1 from Daras 2004 (51 entries) ---
	const TABLE1 = [
	    ("rho_phi_theta","F1,F1,F1"), ("rho_theta_phi","F1,F1,F2"),
	    ("theta_rho_phi","F1,F1,F4"), ("phi_rho_theta","F1,F1,F4"),
	
	    ("rho_phi_theta","F1,F1,F2"), ("rho_theta_phi","F1,F1,F3"),
	    ("theta_rho_phi","F1,F2,F1"), ("phi_rho_theta","F1,F2,F1"),
	
	    ("rho_phi_theta","F1,F1,F4"), ("rho_theta_phi","F1,F1,F4"),
	    ("theta_rho_phi","F1,F2,F4"), ("phi_rho_theta","F1,F3,F1"),
	
	    ("rho_phi_theta","F1,F2,F1"), ("rho_theta_phi","F1,F2,F1"),
	    ("theta_rho_phi","F2,F1,F1"), ("phi_rho_theta","F1,F3,F2"),
	
	    ("rho_phi_theta","F1,F3,F1"), ("rho_theta_phi","F1,F2,F4"),
	    ("theta_rho_phi","F2,F2,F1"), ("phi_rho_theta","F2,F1,F1"),
	
	    ("rho_phi_theta","F1,F3,F4"), ("rho_theta_phi","F1,F4,F1"),
	    ("theta_rho_phi","F3,F1,F1"), ("phi_rho_theta","F2,F2,F1"),
	
	    ("rho_phi_theta","F1,F4,F1"), ("rho_theta_phi","F1,F4,F3"),
	    ("theta_rho_phi","F3,F1,F2"), ("phi_rho_theta","F3,F1,F1"),
	
	    ("rho_phi_theta","F1,F4,F2"), ("rho_theta_phi","F2,F1,F1"),
	    ("theta_rho_phi","F3,F1,F4"), ("phi_rho_theta","F3,F2,F1"),
	
	    ("rho_phi_theta","F2,F1,F1"), ("rho_theta_phi","F2,F1,F3"),
	    ("theta_rho_phi","F3,F2,F4"),
	
	    ("rho_phi_theta","F2,F1,F4"), ("rho_theta_phi","F2,F1,F4"),
	    ("theta_rho_phi","F3,F3,F4"),
	
	    ("rho_phi_theta","F2,F3,F4"), ("rho_theta_phi","F2,F2,F2"),
	
	    ("rho_phi_theta","F2,F4,F1"), ("rho_theta_phi","F2,F4,F1"),
	
	    ("rho_phi_theta","F2,F4,F2"), ("rho_theta_phi","F2,F4,F3"),
	
	    ("rho_phi_theta","F3,F1,F2"), ("rho_theta_phi","F3,F1,F1"),
	
	    ("rho_phi_theta","F3,F1,F4"), ("rho_theta_phi","F3,F2,F4"),
	
	    ("rho_phi_theta","F3,F2,F1"), ("rho_theta_phi","F3,F4,F4"),
	
	    ("rho_phi_theta","F3,F4,F1")
	]

	const TABLE1_FULL = [
	    ("rho_phi_theta","F1,F1,F1"), ("rho_theta_phi","F1,F1,F2"),
	    ("theta_rho_phi","F1,F1,F4"), ("phi_rho_theta","F1,F1,F4"),
	    ("rho_phi_theta","F1,F1,F2"), ("rho_theta_phi","F1,F1,F3"),
	    ("theta_rho_phi","F1,2,F1"),   ("phi_rho_theta","F1,F2,F1"),
	    ("rho_phi_theta","F1,F1,F4"), ("rho_theta_phi","F1,F1,F4"),
	    ("theta_rho_phi","F1,F2,F4"), ("phi_rho_theta","F1,F3,F1"),
	    ("rho_phi_theta","F1,F2,F1"), ("rho_theta_phi","F1,F2,F1"),
	    ("theta_rho_phi","F2,F1,F1"), ("phi_rho_theta","F1,F3,F2"),
	
	    ("rho_phi_theta","F1,F3,F1"), ("rho_theta_phi","F1,F2,F4"),
	    ("theta_rho_phi","F2,F2,F1"), ("phi_rho_theta","F2,F1,F1"),
	    ("rho_phi_theta","F1,F3,F4"), ("rho_theta_phi","F1,F4,F1"),
	    ("theta_rho_phi","F3,F1,F1"), ("phi_rho_theta","F2,F2,F1"),
	    ("rho_phi_theta","F1,F4,F1"), ("rho_theta_phi","F1,F4,F3"),
	    ("theta_rho_phi","F3,F1,2"),   ("phi_rho_theta","F3,F1,F1"),
	    ("rho_phi_theta","F1,F4,F2"), ("rho_theta_phi","F2,F1,F1"),
	    ("theta_rho_phi","F3,F1,F4"), ("phi_rho_theta","F3,F2,F1"),
	
	    ("rho_phi_theta","F2,F1,F1"), ("rho_theta_phi","F2,F1,F3"),
	    ("theta_rho_phi","F3,F2,F4"), ("phi_rho_theta","F3,F2,F4"), 
	    ("rho_phi_theta","F2,F1,F4"), ("rho_theta_phi","F2,F1,F4"),
	    ("theta_rho_phi","F3,F3,F4"), ("phi_rho_theta","F3,F3,F1"),
	    ("rho_phi_theta","F2,F3,F4"), ("rho_theta_phi","F2,F2,F2"),
	    ("rho_phi_theta","F2,4,F1"),   ("rho_theta_phi","F2,F4,F1"),
	    ("rho_phi_theta","F2,4,F2"),   ("rho_theta_phi","F2,F4,F3"),
	    ("rho_phi_theta","F3,F1,F2"), ("rho_theta_phi","F3,F1,F1"),
	    ("rho_phi_theta","F3,F1,F4"), ("rho_theta_phi","F3,2,F4"),
	    ("rho_phi_theta","F3,F2,F1"), ("rho_theta_phi","F3,F4,F4"),
	    ("rho_phi_theta","F3,F4,F1"), ("rho_theta_phi","F3,F4,F2")  
	]
		
	"""
	Exact Daras (2004) descriptor extractor
	Input: R[rho,theta,phi]
	Output: 51-dimensional vector
	"""
	function daras_descriptors(R::Array{<:Real,3})
	    descriptors = Float64[]
	
	    for (order, triple) in TABLE1
	        axes = split(order, "_")               # e.g. ["rho","phi","theta"]
	        fs   = split(triple, ",")
	
	        A = Array{Float64}(R)
	        curr_axes = [:rho, :phi, :theta]
	
	        for k in 1:3
	            ax = Symbol(axes[k])
	            f  = FUNC[fs[k]]
	            pos = findfirst(==(ax), curr_axes)
	            A = reduce_axis(A, pos, f)
	            deleteat!(curr_axes, pos)
	        end
	
	        push!(descriptors, float(A))
	    end
	
	    return descriptors
	end
end

# ╔═╡ b8bd9a31-6571-48ee-a345-33f247aa9e55
#=╠═╡
begin
	evoxel = daras_descriptors.(svoxel)
	Evoxel = hcat(evoxel... )
end
  ╠═╡ =#

# ╔═╡ b5b4ae57-a437-4c40-8983-252ae845abf7
#=╠═╡
Plots.plot(evoxel[1])
  ╠═╡ =#

# ╔═╡ 07884f89-cb26-4612-8530-a65c19146c52
#=╠═╡
begin
	μ = mean(Evoxel, dims=2)
	σ = std(Evoxel, dims=2) .+ eps()
	
	Fz = (Evoxel .- μ) ./ σ
	
	pca = fit(PCA, Fz; maxoutdim=15)
	Zvoxel = transform(pca, Fz)
end
  ╠═╡ =#

# ╔═╡ f108a5d8-b97a-4020-b170-7b28d2ea1d5b
#=╠═╡
zvoxel = [Zvoxel[:,k] for k in 1:100]
  ╠═╡ =#

# ╔═╡ 6c878ab2-af36-4bda-a3b9-5903cc293e6f
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

# ╔═╡ 4dcacaf4-7ffe-4dae-ab51-95351481f968
#=╠═╡
CZ = confusion_map(zvoxel,1)
  ╠═╡ =#

# ╔═╡ d94c8928-8fc6-4605-802c-d5998881c1f9
#=╠═╡
Plots.heatmap(CZ)
  ╠═╡ =#

# ╔═╡ 3f5714c9-b4fc-464e-8caf-8008b4bbf593
#=╠═╡
C = confusion_map(evoxel,2)
  ╠═╡ =#

# ╔═╡ 995cb023-6400-46e3-a691-4680f745a632
#=╠═╡
pc = Plots.heatmap(C)
  ╠═╡ =#

# ╔═╡ 1a7082c4-bfcf-40cc-9263-7b5bc71f9156
# ╠═╡ disabled = true
#=╠═╡
Plots.savefig(pc, "shape_matching_conf_map_2048_20_16_ord_2.pdf")
  ╠═╡ =#

# ╔═╡ 37567a53-6a72-4d9f-996f-98e31af48feb
md"""
### Data similarity, used in Daras' paper.
`daras_similarity`(DA, DB; eps_den=1e-12)

Compute the similarity score from Daras et al. (2004), Eq. (17).

Inputs:
 - DA, DB : Vectors of equal length (e.g. length 51)
 - eps_den : small value to avoid division by zero (default 1e-12)

Returns similarity in percent (0..100).
"""

# ╔═╡ 4045055c-7a1e-488f-a528-6a9f62f450b7
function daras_similarity(DA::AbstractVector{<:Real}, DB::AbstractVector{<:Real}; eps_den=1e-12)
    @assert length(DA) == length(DB)
    N = length(DA)

    # Ensure Float64 arithmetic
    a = Float64.(DA)
    b = Float64.(DB)

    # denom = (a + b)/2 ; per-paper denominator
    denom = (a .+ b) ./ 2.0

    # If denom is zero (both a and b zero) then the fractional term is defined as 0.
    # We'll compute safely by masking.
    diff = abs.(a .- b)

    # safe fraction: diff ./ denom with denom==0 -> 0
    frac = similar(diff)
    for i in eachindex(diff)
        if abs(denom[i]) <= eps_den
            frac[i] = 0.0
        else
            frac[i] = diff[i] / denom[i]
        end
    end

    score = (1.0 - sum(frac) / N) * 100.0
    # clamp to [0,100] for numerical safety
    return max(0.0, min(100.0, score))
end

# ╔═╡ 39bbfac0-7969-4c85-81d0-2e2100c54312
function confusion_daras(quantiles::AbstractArray)
	n = length(quantiles)
	confmap = zeros(n, n)
	for k in 1:n
		for j in 1:n
			confmap[k, j] = daras_similarity(quantiles[k],quantiles[j])
		end
	end
	return confmap
end

# ╔═╡ 708524ca-e335-4214-a32f-161b777a210f
#=╠═╡
CD = confusion_daras(evoxel)
  ╠═╡ =#

# ╔═╡ a12a388f-4977-4cf2-b047-a6a984e37b05
#=╠═╡
cd = Plots.heatmap(CD)
  ╠═╡ =#

# ╔═╡ 7e5d8b7a-4613-4066-afcf-a7ab40a2f6b7
# ╠═╡ disabled = true
#=╠═╡
Plots.savefig(cd, "shape_matching_similarity_map_2048_20_16.pdf")
  ╠═╡ =#

# ╔═╡ 08b9b025-6797-48bf-8e50-8eb02352076a
#=╠═╡
Plots.heatmap(CD.>66.6)
  ╠═╡ =#

# ╔═╡ 4284e5b2-552b-4da7-9138-f88bdd4f4413
#=╠═╡
kEvoxel = kmeans(Zvoxel, 10, maxiter=100, display=:iter)
  ╠═╡ =#

# ╔═╡ 6631451f-2729-49c2-92f3-e4aa83ab5462
#=╠═╡
Plots.plot(kEvoxel.assignments)
  ╠═╡ =#

# ╔═╡ 4ce0ef20-8c5d-40b9-bc49-53f7cae65c4d
md"""
## Nearest Template and Neighbour Classification
Use the nearest template or neighbour classification
with respect to the chosen templates
to classify the generated dataset.
The normalized RCDT is applied
with different numbers of used angles, normes, and variantes.
"""

# ╔═╡ 1242f93b-edf6-4163-8147-221cff3c9c17
begin
	function plot_quantiles(temp_q::AbstractArray, temp_lab::AbstractArray, data_q::AbstractArray, data_lab::AbstractArray, text)
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
	        Plots.plot!(plt, temp_q[i], fontfamily="Computer Modern", 
				# label=["class $k" "k"], 
				label=text[i],
				linewidth=2 , linecolor = RGBA((abs(0.5-(i-1)/(length(temp_lab)-1)))*(1-(i-1)/(length(temp_lab)-1))*0.99, ((i-1)/(length(temp_lab)-1))*(1-(i-1)/(length(temp_lab)-1))*0.85, (abs(0.5-(i-1)/(length(temp_lab)-1)))*((i-1)/(length(temp_lab)-1))*0.99, 1), yticks=true, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));    
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
	    conf_matrix = confusion_mat(data_lab, label_rcdt) 
	    if ret==1
	        @info "Acc. using $(norm)-norm : \t $(acc_rcdt)"
	    end
	    return acc_rcdt, conf_matrix
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
	
	        acc_rcdt[i], conf_matrix_part = accuracy_k_nearest_neighbour(train_data_q, train_labels, test_data_q, test_labels, norm, K=K)
	        # acc_rcdt[i] = accuracy_k_nearest_neighbour(train_data_q, train_labels, test_data_q, test_labels, norm, K=K)
	        if ret==1
	            conf_matrix += conf_matrix_part 
	        end
	    end
	    @info "Acc. using $(norm)-norm : \t $(mean(acc_rcdt)) +/- $(std(acc_rcdt))"
	    if ret==1
	        return conf_matrix
	    end
	end

	function confusion_mat(true_labels::AbstractArray, predicted_labels::AbstractArray)
	    # Get the sorted unique labels from both true and predicted labels
	    labels = sort(unique(vcat(true_labels, predicted_labels)))
	    n_labels = length(labels)
	    
	    # Create a mapping from label to index
	    label_to_index = Dict(label => idx for (idx, label) in enumerate(labels))
	    
	    # Initialize the confusion matrix as an integer matrix
	    matrix = zeros(Int, n_labels, n_labels)
	    
	    # Fill in the confusion matrix
	    for (t, p) in zip(true_labels, predicted_labels)
	        i = label_to_index[t]
	        j = label_to_index[p]
	        matrix[i, j] += 1
	    end
	    
	    return matrix
	end
end

# ╔═╡ e06795a0-e9de-4bd4-a8da-6585348126eb
#=╠═╡
begin
	K = 5
	CC₁ = accuracy_k_nearest_part_neighbour(20, K, 10, 10, evoxel, Labels, "manhattan", K=1, ret=1)
	CC₂ = accuracy_k_nearest_part_neighbour(20, K, 10, 10, evoxel, Labels, "euclidean", K=1, ret=1)
	CC∞ = accuracy_k_nearest_part_neighbour(20, K, 10, 10, evoxel, Labels, "inf", K=1, ret=1)
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╟─80b0dd3a-69ae-48df-abac-c8de5625fe2e
# ╠═7a209b02-782f-4b79-a4b4-d89029206c23
# ╠═0791c621-08c5-4772-bbcb-906957a721be
# ╠═b67c6b62-8cec-4f62-8709-07ddc17354d3
# ╠═dde2b58d-f858-4201-8fa8-94165d256b66
# ╟─1c634e17-e792-47ec-bab8-2bcdb342bed7
# ╠═010cf880-a1c8-49f2-bad2-80fcfecbc7ce
# ╠═c86866df-5730-4f2a-86d0-88e4b9bdba1a
# ╠═06c66cd3-1324-4592-930e-73587ec0c5b6
# ╟─7cd2d106-cbe1-4ef6-8efc-a3340d059793
# ╠═7b8ca11e-b3a8-4455-8811-c74097d2d50f
# ╠═5f8ec9ae-d81c-4335-85de-f15048d9bbcf
# ╠═41f9d8c0-5cb5-4068-b253-033147dd3449
# ╠═05c285a0-328a-48f4-bb43-888f4d180339
# ╠═03fc0177-5d3d-4ff0-abe9-3ba7c9e11740
# ╟─5999b3b7-7a53-40a2-b8df-1e05887e91ac
# ╠═a6d430f2-1f9b-4b44-9e22-2e064849aa12
# ╟─a91c4339-41f2-469c-9e55-50c1732906ec
# ╠═4d629451-66b7-44a9-8693-9f48d5f324a3
# ╠═a2d9776f-3e8a-428f-af93-dd16ff86ad75
# ╠═13832bcd-831f-445a-bb50-fab64300211f
# ╠═35949af4-e037-4e68-9e54-4cfc3a8c7316
# ╠═47b98719-a983-4480-ab09-23676f8ee9da
# ╠═837b8fa2-b7e8-4850-a513-69c03d35a45b
# ╠═3daa2232-071d-493c-aa58-f1b8d0cef86a
# ╟─a75c6ac8-86a0-479d-aa87-e5bde563d22b
# ╠═aff5fbc3-2724-45fd-802b-3b60e509a999
# ╠═107302bd-01ad-4787-aa96-61d6e93b08b7
# ╠═96ae91b4-fafd-47e5-b2a9-e19f61b21f84
# ╠═cbfc4e51-031f-46f3-92fc-32f3335ef22d
# ╠═baa5029f-428f-4e48-8c6a-c31b643fa5cf
# ╠═2d6b3f00-18f6-4500-820f-2c931ddb7dd1
# ╟─da9fbc60-77eb-4952-acb5-7bba2e1fa045
# ╠═88a06a44-b384-4e41-a3cf-b90130521749
# ╠═b8bd9a31-6571-48ee-a345-33f247aa9e55
# ╠═b5b4ae57-a437-4c40-8983-252ae845abf7
# ╠═fa74a3c9-24a0-41ab-951e-dd35755808f0
# ╠═07884f89-cb26-4612-8530-a65c19146c52
# ╠═f108a5d8-b97a-4020-b170-7b28d2ea1d5b
# ╠═6c878ab2-af36-4bda-a3b9-5903cc293e6f
# ╠═4dcacaf4-7ffe-4dae-ab51-95351481f968
# ╠═d94c8928-8fc6-4605-802c-d5998881c1f9
# ╠═3f5714c9-b4fc-464e-8caf-8008b4bbf593
# ╠═995cb023-6400-46e3-a691-4680f745a632
# ╠═1a7082c4-bfcf-40cc-9263-7b5bc71f9156
# ╟─37567a53-6a72-4d9f-996f-98e31af48feb
# ╠═4045055c-7a1e-488f-a528-6a9f62f450b7
# ╠═39bbfac0-7969-4c85-81d0-2e2100c54312
# ╠═708524ca-e335-4214-a32f-161b777a210f
# ╠═a12a388f-4977-4cf2-b047-a6a984e37b05
# ╠═7e5d8b7a-4613-4066-afcf-a7ab40a2f6b7
# ╠═08b9b025-6797-48bf-8e50-8eb02352076a
# ╠═213c7f64-bd0e-4485-aed0-f36d2a398f66
# ╠═4284e5b2-552b-4da7-9138-f88bdd4f4413
# ╠═6631451f-2729-49c2-92f3-e4aa83ab5462
# ╟─4ce0ef20-8c5d-40b9-bc49-53f7cae65c4d
# ╠═e3fbd24c-7e4d-4841-b028-53fda048ac00
# ╠═1242f93b-edf6-4163-8147-221cff3c9c17
# ╠═e06795a0-e9de-4bd4-a8da-6585348126eb
