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

# ╔═╡ 9026057c-ac03-41a9-b68b-5b44b2937eb1
md"""
# arXiv:xxxx.xxxxx -- Table x, Figure x
This pluto notebook reproduces the numerical visualizations
with tamplates for Figure x from

- Robert Beinert, Jonas Bresch, and Michael Quellmalz
  '*A Discrete Radon Transform Based on the Area of Cube-Plane Intersection*',
  arXiv:xxxx.xxxxx, 2026.
"""

# ╔═╡ 66f1e84b-dd99-4b7b-8ccd-b3f6b7282fa9
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

# ╔═╡ 89a09bb5-f806-4c64-9eda-7387cd851688
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

# ╔═╡ 08f1d9df-8ea6-48a2-9528-e7d5416c50a2
begin
	ℓ = 10
	ModelNet10 = vcat(voxel_bathtub[1:ℓ],voxel_bed[1:ℓ],voxel_chair[1:ℓ],voxel_desk[1:ℓ],voxel_dresser[1:ℓ],voxel_monitor[1:ℓ],voxel_night_stand[1:ℓ],voxel_sofa[1:ℓ],voxel_table[1:ℓ],voxel_toilet[1:ℓ])
end

# ╔═╡ 9a88aa43-f6b6-458a-939b-a91510bdf571
begin
	PNet = Phantom.(ModelNet10)
end

# ╔═╡ 88faa132-5788-450b-bbda-5c4ab6dc16c1
md"""
## Radon transform
Setup the Radon transform in `3d`
Discretization, given by 

- number of equispaced radii in [-√3⋅d/2,√3⋅d/2],
- Fibonacci sphere points (nearly equispaced on the S^2).
"""

# ╔═╡ 8d783205-ffa0-4ac6-957c-706b7505806b
begin
	d = 64
	r = 64
	radii = collect(LinRange(-sqrt(3)/2*(d-1),sqrt(3)/2*(d-1),r))
	angles = fibonacci_spherical_directions(64)
	R = RadonTransform(radii,angles)
end

# ╔═╡ 2127d551-0be7-447b-952c-640e4af509af
md"""
## Normalized RCDT
Setup the normalized RCDT,
and apply it 
to the dataset and templates.
"""

# ╔═╡ 9359f6f1-ae2c-4313-b480-ffc433e98eb4
begin
	struct RadonCDT 
	    quantiles::Int64
	    Radon::XdRadonTransform.RadonTransform
	    signal_raising::Float64
	    function RadonCDT(q, R, ϵ)
	        if q <= 0
	            error("non-positive number of quantiles")
	        elseif ϵ <= 0
	            error("non-positive signal raising")
	        end
	        return new(q, R, ϵ)
	    end
	end
	
	
	RadonCDT(q, R) = RadonCDT(q, R, 1e-10)
	
	
	function (RCDT::RadonCDT)(image::XdRadonTransform.Phantom)
	    S = RCDT.Radon(image)
		I = S.data
	    Q = zeros(RCDT.quantiles, RCDT.Radon.num_angles)
	    for k in axes(Q, 2)
	        Q[:, k] = cdt(I[:, k], RCDT)
	    end
	    return Q
	end
	
	function cdt(s::Vector{Float64}, RCDT::RadonCDT)
	    p = signal_to_density(s, RCDT.signal_raising)
	    P = cumsum(p)
	    s_grid = collect(LinRange(-1 - RCDT.Radon.width / 2, 1 + RCDT.Radon.width / 2, length(s)))
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
	
	function (NRCDT::NormRadonCDT)(image::XdRadonTransform.Phantom)
	    rcdt = NRCDT.RCDT(image)
	    nrcdt = (rcdt .- mean(rcdt, dims=1)) ./ std(rcdt, dims=1)
	    return nrcdt
	end
	
	struct MaxNormRadonCDT
	    RCDT::RadonCDT
	end
	
	function (mNRCDT::MaxNormRadonCDT)(image::XdRadonTransform.Phantom)
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
	RCDT = RadonCDT(128, R);
	NRCDT = NormRadonCDT(RCDT);
end

# ╔═╡ 654dc837-3c6d-4b87-8ccb-6cbfadd2bb78
qTempNRCDT = NRCDT.RCDT.(PNet);

# ╔═╡ 08c9d81b-206c-4d73-b0be-eb3361873010
pₜ = Plots.heatmap(qTempNRCDT[1])

# ╔═╡ da8a20b8-fa41-4b4e-88c6-426bba656d34
# ╠═╡ disabled = true
#=╠═╡
Plots.savefig(pₜ, "rcdt_c_128_128_t1.pdf")
  ╠═╡ =#

# ╔═╡ 2d6b2930-f83b-4a60-9a59-8cb7308cf78b
nqTempNRCDT = normalization.(qTempNRCDT)

# ╔═╡ 4e2f1004-6fbe-4151-94ad-81f428250493
pₙ = Plots.heatmap(nqTempNRCDT[1])

# ╔═╡ b8b48b65-4cff-4e31-be11-3f67522b2322
# ╠═╡ disabled = true
#=╠═╡
Plots.savefig(pₙ, "nrcdt_c_128_128_t1.pdf")
  ╠═╡ =#

# ╔═╡ e1af6bdf-cef1-4963-9821-78a5bafcc04c
begin
	Label = 1:10
	Class, Labels = generate_academic_classes_3d(ModelNet10[[1,11,21,31,41,51,61,71,81,91]], Label, class_size=10)
end

# ╔═╡ 974bde48-c1de-4011-a3cc-c3a406c6904a
begin
	mnqTempNRCDT = maxnormalization.(qTempNRCDT)
	
	anqTempNRCDT = meannormalization.(qTempNRCDT)
end

# ╔═╡ 6cce609b-fc68-4eb9-a056-e9d75c2cc1e0
begin
	Plots.plot(mnqTempNRCDT[21:30])
end

# ╔═╡ f7114e5e-b407-4952-8f3f-1c3fcaef4a2e
begin
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

# ╔═╡ 67372c90-c5df-4efd-bcac-824c0527241e
begin
	CM₁ = confusion_map(mnqTempNRCDT, 1);
	CM₂ = confusion_map(mnqTempNRCDT, 2);
	CM∞ = confusion_map(mnqTempNRCDT, 100);

	CA₁ = confusion_map(anqTempNRCDT, 1);
	CA₂ = confusion_map(anqTempNRCDT, 2);
	CA∞ = confusion_map(anqTempNRCDT, 100);
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
	Plots.savefig(pcm₁, "cm_all_1_256_256.pdf")
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
	Plots.savefig(pcm₂, "cm_all_2_256_256.pdf")
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
	Plots.savefig(pcm∞, "cm_all_∞_256_256.pdf")
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
	Plots.savefig(pca₁, "ca_all_1_256_256.pdf")
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
	Plots.savefig(pca₂, "ca_all_2_256_256.pdf")
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
	Plots.savefig(pca∞, "ca_all_∞_256_256.pdf")
end

# ╔═╡ 74b495d0-6782-4b9e-9293-d024d82e5792
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

# ╔═╡ 47e7e872-37a1-4363-a881-1fac2142d4fd
sel = vcat(1:10,21:30,41:50) # vcat(1:10,41:50,91:100)

# ╔═╡ 1353049f-dc17-4a09-8d33-ae8b9a1202ee
begin
	Plots.plot(anqTempNRCDT[sel])
end

# ╔═╡ a6c48107-e019-412f-88b2-e14057ec42f3
begin
	K₁ = 5
	CC₁ = accuracy_k_nearest_part_neighbour(20, K₁, 10, 3, mnqTempNRCDT[sel], Labels[sel], "manhattan", K=1)
	CC₁ = accuracy_k_nearest_part_neighbour(20, K₁, 10, 3, mnqTempNRCDT[sel], Labels[sel], "euclidean", K=1)
	CC₁ = accuracy_k_nearest_part_neighbour(20, K₁, 10, 3, mnqTempNRCDT[sel], Labels[sel], "inf", K=1)
end

# ╔═╡ 4fa3ed4d-e651-4ac3-b69a-b164cf9676ef
begin
	K₂ = 5
	CC₂ = accuracy_k_nearest_part_neighbour(20, K₂, 10, 3, anqTempNRCDT[sel], Labels[sel], "manhattan", K=1)
	CC₂ = accuracy_k_nearest_part_neighbour(20, K₂, 10, 3, anqTempNRCDT[sel], Labels[sel], "euclidean", K=1)
	CC₂ = accuracy_k_nearest_part_neighbour(20, K₂, 10, 3, anqTempNRCDT[sel], Labels[sel], "inf", K=1)
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
# ╠═9026057c-ac03-41a9-b68b-5b44b2937eb1
# ╠═c20ec051-9ae9-40f7-97cb-7abf2baf4fe4
# ╠═1fa304a4-ecf5-4645-8f3a-90aedde4323e
# ╠═178b1a7f-ceeb-43ba-9006-e2c5fd7f6b05
# ╠═62aa8854-c7b6-4da4-a904-22d56ed5364c
# ╠═66f1e84b-dd99-4b7b-8ccd-b3f6b7282fa9
# ╠═89a09bb5-f806-4c64-9eda-7387cd851688
# ╠═08f1d9df-8ea6-48a2-9528-e7d5416c50a2
# ╠═9a88aa43-f6b6-458a-939b-a91510bdf571
# ╠═88faa132-5788-450b-bbda-5c4ab6dc16c1
# ╠═8d783205-ffa0-4ac6-957c-706b7505806b
# ╠═2127d551-0be7-447b-952c-640e4af509af
# ╠═9359f6f1-ae2c-4313-b480-ffc433e98eb4
# ╠═aa720f14-298a-45df-a6f3-921fd5f77ac2
# ╠═4beb1c43-5d2e-4580-a740-ac4edb625e17
# ╠═654dc837-3c6d-4b87-8ccb-6cbfadd2bb78
# ╠═08c9d81b-206c-4d73-b0be-eb3361873010
# ╠═da8a20b8-fa41-4b4e-88c6-426bba656d34
# ╠═2d6b2930-f83b-4a60-9a59-8cb7308cf78b
# ╠═4e2f1004-6fbe-4151-94ad-81f428250493
# ╠═b8b48b65-4cff-4e31-be11-3f67522b2322
# ╠═e1af6bdf-cef1-4963-9821-78a5bafcc04c
# ╠═974bde48-c1de-4011-a3cc-c3a406c6904a
# ╠═6cce609b-fc68-4eb9-a056-e9d75c2cc1e0
# ╠═1353049f-dc17-4a09-8d33-ae8b9a1202ee
# ╠═f7114e5e-b407-4952-8f3f-1c3fcaef4a2e
# ╠═3f39ad0f-aed1-4d67-9731-42914d54ecdf
# ╠═67372c90-c5df-4efd-bcac-824c0527241e
# ╠═45b69587-1056-4268-affc-d7a6e8475208
# ╠═cf87c9a1-f9c3-4d02-a984-90d5a5506779
# ╠═fa13a0bb-a54b-4696-8598-2c5acd3995f9
# ╠═37ed9041-c2ab-48ee-83a3-2a4a6c156083
# ╠═3dce15c5-062f-4be5-8bce-856622f1b527
# ╠═2543cc8c-1631-4f76-ac17-7422544f723b
# ╠═74b495d0-6782-4b9e-9293-d024d82e5792
# ╠═92496979-16b9-47b6-8eb1-7a68aef851ef
# ╠═47e7e872-37a1-4363-a881-1fac2142d4fd
# ╠═a6c48107-e019-412f-88b2-e14057ec42f3
# ╠═4fa3ed4d-e651-4ac3-b69a-b164cf9676ef
# ╠═ebc40d73-6ae0-4dea-bbea-8618fa136bed
# ╠═1e415a55-0089-45c2-a8a0-592bdc8424b3
# ╠═dbad2a0d-ac54-4d35-90bc-48613b7bd130
# ╠═d6e844b8-6cdb-4888-a4af-f4ece20a7c48
# ╠═c77c2478-73fd-47a9-a0a1-713e534ec712
# ╠═23c841ef-d77b-4f38-81be-2fdb22c784db
