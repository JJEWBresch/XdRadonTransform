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
	using Augmentor
	using IterTools
	using LinearAlgebra
	using GLMakie
	# plotlyjs()
end

# ╔═╡ 62aa8854-c7b6-4da4-a904-22d56ed5364c
using Statistics, Random, StatsBase

# ╔═╡ d07cf61a-edd0-4176-a2da-836f511490a1
using Interpolations: LinearInterpolation

# ╔═╡ 29ec4a5a-08a1-4194-a2fe-69e4d07150c5
md"""
# arXiv:xxxx.xxxxx -- Table 2, Figure 3
This pluto notebook reproduces the numerical visualizations
with tamplates for Figure 1 with template from Figure 3 from

- Robert Beinert, Jonas Bresch, and Michael Quellmalz
  '*John Transform for Object Classification*',
  arXiv:xxxx.xxxxx, 2026.
"""

# ╔═╡ 5ba7d520-b0cc-49dc-85b4-f7cb7920ae6c
md"""
## Templates
Generate the three templates
using the submodule `TestImages`.
"""

# ╔═╡ e9133a33-ec57-4ffb-850f-24da6a4d81a1
begin
	stroke_width = 2;
	blur = 3; #15
end

# ╔═╡ fbb64f21-c677-4d7b-bef3-91594f4293f5
begin
	I₁_sharp = render(TestImages.Circle(scale=(0.8, 0.8)), size=(32,32), width=stroke_width);
	I₁ = augment(I₁_sharp, GaussianBlur(blur));
	J₁ = extend_image(I₁, (64, 64))
end

# ╔═╡ bf3adcf5-0d1a-425c-96ba-819f95c281f6
begin
	I₂_sharp = render(TestImages.Square(scale=(0.8, 0.8)), size=(32,32), width=stroke_width);
	I₂ = augment(I₂_sharp, GaussianBlur(blur));
	J₂ = extend_image(I₂, (64, 64))
end

# ╔═╡ f98d22e0-ae6c-4870-b5b4-dab2bc1f3e18
begin
	I₃_sharp = render(TestImages.Triangle(scale=(0.8, 0.8)), size=(32,32), width=stroke_width);
	I₃ = augment(I₃_sharp, GaussianBlur(blur));
	J₃ = extend_image(I₃, (64, 64))
end

# ╔═╡ 7af925b8-29fa-40ae-84bb-3bb95a96d415
md"""
## Dataset
Generate the dataset 
by duplicating the templates
and by applying random affine transformations
using the submodule `DataTransformations`.
"""

# ╔═╡ a23573e6-dfca-4fa2-b167-b131c427293e
begin
	T = [generate_3d_rot(J₁, order=2.0), generate_3d_rot(J₂, order=20.0), generate_3d_rot(J₃, order=1.0)];
	Label = [1, 2, 3];
end

# ╔═╡ c3d8db69-40a7-415e-8324-90080c9aa84b
Class, Labels = generate_academic_classes_3d(T, Label, class_size=5);

# ╔═╡ 8e1c4550-96d0-4e06-92ac-bfe53ecfea09
A = DataTransformations.RandomAffineTransformation3d(
	rotate_x = (-π,π),
	rotate_y = (-π,π),
	rotate_z = (-π,π),
	scale_x = (0.75, 1.25),
	scale_y = (0.75, 1.25),
	scale_z = (0.75, 1.25),
	scale_i = (0.5, 1.5),
	# shear_xy = (-0.15, 0.15),
	# shear_yx = (-0.15, 0.15),
	# shear_xz = (-0.15, 0.15),
	# shear_zx = (-0.15, 0.15),
	# shear_yz = (-0.15, 0.15),
	# shear_zy = (-0.15, 0.15),
	shift_x = (-10,10),
	shift_y = (-10,10),
	shift_z = (-10,10)
	)

# ╔═╡ 22420609-2ed2-4102-8fe3-822783b9e490
begin
	Random.seed!(42424)
	TClass = A.(Class)
end

# ╔═╡ 66130468-077c-4522-a25b-a37e041b7a64
for k in 1:length(TClass)
	TClass[k][isnan.(TClass[k])] .= 0.0;
	k += 1
end

# ╔═╡ 7460bd01-6bd0-4c81-b9cf-e78acfc0e380
GLMakie.volume(TClass[12])

# ╔═╡ c859e69d-dec5-4d52-a875-ea22f28ebfa4
md"""
## John transform
Setup the John transform
and apply it 
to the dataset and templates.

- resolution::Tuple{Int64,...,Int64} in the tagentspace
- boxsize::Tuple{Int64,...,Int64} of the domain of the John transform
"""

# ╔═╡ 3f35dfa9-c4d2-4773-8324-84959b014743
begin
	d = 3
	ϰ = 64
	ℓ = 32
	resolution = Tuple(Int64.(ϰ*ones(d-1)))
	boxsize = Tuple(Int64.(ℓ*ones(d-1)))
	θ = randn(d)
	θ ./ norm(θ)
end

# ╔═╡ e4abc6cc-790b-408f-b94b-7ac1a4a365b0
begin
	Jθ = RestrictedJohnTransform(resolution, boxsize, θ)
end

# ╔═╡ 9f2225ea-42bb-4d83-b74f-9fddf0a6aac5
Jθ.tangentspace

# ╔═╡ 3353e434-20fc-4f76-baf6-1086f69da2a2
[p ⋅ θ for p in Jθ.tangentspace]

# ╔═╡ fa74be1a-e0d4-424e-8bed-2b404123c272
Plots.heatmap(norm.(Jθ.tangentspace))

# ╔═╡ c6d48960-2bd5-417f-baa4-b6a6215e4882
md"""
## Normalized John-CDT -- Figure xxx
Setup the max-normalized RCDT,
and apply it 
to the dataset and templates.
"""

# ╔═╡ 1279f747-8661-45a8-b47f-df3e7927e473
begin
	PJ = PhantomBox.(T)
	JJ = Jθ.(PJ)
end

# ╔═╡ e22aaa88-1c56-4223-ae20-1f68f325871b
Plots.heatmap(JJ[1].data)

# ╔═╡ 71f4f715-61fb-4eb5-9a9c-b83dba91923f
begin
	PTClass = PhantomBox.(TClass)
	JClass = Jθ.(PTClass)
end

# ╔═╡ dee02cf0-1368-4bff-a229-fcc864711137
Plots.heatmap(JClass[13].data)

# ╔═╡ 8bc81f60-824c-44a3-843f-f10ff0e09036
begin
	ϰₜ = 64
	ℓₜ = 32
	resolutionₜ = Tuple(Int64.(ϰₜ*ones(d-2)))
	boxsizeₜ = Tuple(Int64.(ℓₜ*ones(d-2)))
	θₜ = randn(d-1)
	θₜ ./ norm(θₜ)
end

# ╔═╡ 8caf0f7e-03dd-4312-93b1-44bc74fb9f2a
begin
	Jθₜ = RestrictedJohnTransform(resolutionₜ, boxsizeₜ, θₜ)
end

# ╔═╡ 871adfcd-1b4d-4569-a8f0-111a872e6f10
begin
	JJP = []
	JClassP = []
	JJP = [jj.data for jj in JJ]
	JClassP = [jp.data for jp in JClass]
end

# ╔═╡ a4a690d3-60bc-4efc-9fe7-a50d077ed785
begin
	JJPP = PhantomBox.(JJP)
	JJJ = Jθₜ.(JJPP)
	JJClassP = PhantomBox.(JClassP)
	JJClass = Jθₜ.(JJClassP)
end

# ╔═╡ 801bebde-dcdb-4f5f-b45e-04e24a08982f
begin
	Plots.plot(JJJ[1].data)
	Plots.plot!(JJClass[1].data)
	Plots.plot!(JJClass[2].data)
	Plots.plot!(JJClass[3].data)
	Plots.plot!(JJJ[2].data)
	Plots.plot!(JJClass[6].data)
	Plots.plot!(JJClass[7].data)
	Plots.plot!(JJClass[8].data)
	Plots.plot!(JJJ[3].data)
	Plots.plot!(JJClass[11].data)
	Plots.plot!(JJClass[12].data)
	Plots.plot!(JJClass[13].data)
end

# ╔═╡ bc032ecd-e12d-4e80-8d1c-6c10271aa09b
md"""
## Nearest Neighbour Classification -- Table xxx
Use the nearest neighbour classification
with respect to the chosen templates
to classify the generated dataset.
The normalized J-CDT is applied
with different numbers of used angles.
"""

# ╔═╡ 5cd92283-c65e-4d3a-9758-8402b1be7708
begin
	dᵣ = 64
	rᵣ = 256
	radiiᵣ = collect(LinRange(-1/sqrt(2)*(dᵣ),1/sqrt(2)*(dᵣ),rᵣ))
	anglesᵣ = uniform_circle(128)
	Rᵣ = RadonTransform(radiiᵣ,anglesᵣ)
	Rᵣᵣ = RadonTransformRay(radiiᵣ,anglesᵣ)
end

# ╔═╡ 178efc7f-6ffb-4c5f-8346-edcda2f5b968
begin
	RJPP = Phantom.(JJP)
	RJJ = Rᵣ.(RJPP)
	RJClassP = Phantom.(JClassP)
	RJClass = Rᵣ.(RJClassP)
end

# ╔═╡ 98972142-7556-4611-84c7-48ba84f84570
begin
	function rcdt(ss::Sinogram, 
		quantiles::Int64; 
		signal_raising=1e-10::Float64)
		S = ss.data
		angles = size(ss.angles)[1]
	    Q = zeros(quantiles, angles)
	    for k in axes(Q, 2)
	        Q[:, k] = cdt(S[:, k], quantiles; signal_raising)
	    end
	    return Q
	end

	function cdt(ss::RestrictedSinogram, 
		quantiles::Int64; 
		signal_raising=1e-10::Float64)
		s = ss.data
	    p = signal_to_density(s, signal_raising)
	    P = cumsum(p)
	    s_grid = collect(LinRange(-1, 1, length(s)))
	    q_grid = collect(LinRange(0, 1, quantiles + 2))[2:end - 1]
	    q = LinearInterpolation(P, s_grid)
	    return q(q_grid)
	end
		
	function nrcdt(ss::Sinogram, 
		quantiles::Int64; 
		signal_raising=1e-10::Float64)
	    r_cdt = rcdt(ss, quantiles; signal_raising)
	    nr_cdt = (r_cdt .- mean(r_cdt, dims=1)) ./ std(r_cdt, dims=1)
	    return nr_cdt
	end
	
	function mnrcdt(ss::Sinogram, 
		quantiles::Int64; 
		signal_raising=1e-10::Float64)
	    nr_cdt = nrcdt(ss, quantiles; signal_raising)
	    mnr_cdt = dropdims(maximum(nr_cdt, dims=2), dims=2)
	    return mnr_cdt
	end
end

# ╔═╡ 9c1f7df6-864d-474c-bcd4-3be47ad26c84
begin
	function njcdt(ss::RestrictedSinogram, 
		quantiles::Int64; 
		signal_raising=1e-10::Float64)
		jcdt = cdt(ss.data, quantiles; signal_raising)
		njcdt = ncdt(jcdt)
		return njcdt
	end
	
	function cdt(s::AbstractArray, 
		quantiles::Int64; 
		signal_raising=1e-10::Float64)
	    p = signal_to_density(s, signal_raising)
	    P = cumsum(p)
	    s_grid = collect(LinRange(-1, 1, length(s)))
	    q_grid = collect(LinRange(0, 1, quantiles + 2))[2:end - 1]
	    q = LinearInterpolation(P, s_grid)
	    return q(q_grid)
	end
	
	function signal_to_density(s::Vector{Float64}, ϵ::Float64)
	    p = s .+ ϵ
	    return p ./= sum(p)
	end

	function ncdt(jcdt::Vector{Float64})
		njcdt = (jcdt .- mean(jcdt)) ./ std(jcdt)
		return njcdt
	end
end

# ╔═╡ 1fb20e89-8bdd-48fe-a6b3-2da865f72c79
begin
	JCDT_T = njcdt.(JJJ, 128)
	JCDT_TClass = njcdt.(JJClass, 128)
end

# ╔═╡ dce6a3b3-9415-4473-b394-15681e0c6330
begin
	nRJJ = nrcdt.(RJJ, 128)
	nRJClass = nrcdt.(RJClass, 128)

	mnRJJ = mnrcdt.(RJJ, 128)
	mnRJClass = mnrcdt.(RJClass, 128)
end

# ╔═╡ 5793a5cb-35ce-42d4-a196-c8985b61c269
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
			Plots.plot!(plt, temp_q[i], legend=false, fontfamily="Computer Modern", label=["Class $k" "k"], linewidth=2 , linecolor = RGBA((abs(0.5-(i-1)/(length(temp_lab)-1)))*(1-(i-1)/(length(temp_lab)-1))*0.99, ((i-1)/(length(temp_lab)-1))*(1-(i-1)/(length(temp_lab)-1))*0.85, (abs(0.5-(i-1)/(length(temp_lab)-1)))*((i-1)/(length(temp_lab)-1))*0.99, 1), yticks=true, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));    
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

	function confusion_matrix(true_labels::AbstractArray, predicted_labels::AbstractArray)
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
	    conf_matrix = confusion_matrix(data_lab, label_rcdt) 
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
end

# ╔═╡ e08bf1ad-323b-4e21-932b-61c942170d3d
plot_quantiles(JCDT_T, Label, JCDT_TClass, Labels)

# ╔═╡ 7aba5b91-9476-419d-af7c-7e646c3090bb
begin
	accuracy_nearest_neighbour(JCDT_T, Label, JCDT_TClass, Labels, "manhattan", ret=1);
	accuracy_nearest_neighbour(JCDT_T, Label, JCDT_TClass, Labels, "euclidean", ret=1);
	accuracy_nearest_neighbour(JCDT_T, Label, JCDT_TClass, Labels, "inf", ret=1);
end

# ╔═╡ 64cea78e-48c5-4fcd-9e09-02040f55d56f
begin
	accuracy_k_nearest_part_neighbour(20, 1, 5, 3, JCDT_TClass, Labels, "manhattan", K=1)
	accuracy_k_nearest_part_neighbour(20, 1, 5, 3, JCDT_TClass, Labels, "euclidean", K=1)
	accuracy_k_nearest_part_neighbour(20, 1, 5, 3, JCDT_TClass, Labels, "inf", K=1)
end

# ╔═╡ ce7191e1-b560-44dc-9170-a39964b466a2
plot_quantiles(mnRJJ, Label, mnRJClass, Labels)

# ╔═╡ b301da2d-4368-4dda-8733-c1d40cf27671
begin
	accuracy_nearest_neighbour(mnRJJ, Label, mnRJClass, Labels, "manhattan", ret=1);
	accuracy_nearest_neighbour(mnRJJ, Label, mnRJClass, Labels, "euclidean", ret=1);
	accuracy_nearest_neighbour(mnRJJ, Label, mnRJClass, Labels, "inf", ret=1);
end

# ╔═╡ 1349301a-16f3-4071-bb4e-4478bca4aeef
begin
	accuracy_k_nearest_part_neighbour(20, 1, 5, 3, mnRJClass, Labels, "manhattan", K=1)
	accuracy_k_nearest_part_neighbour(20, 1, 5, 3, mnRJClass, Labels, "euclidean", K=1)
	accuracy_k_nearest_part_neighbour(20, 1, 5, 3, mnRJClass, Labels, "inf", K=1)
end

# ╔═╡ 9a9feb1c-8685-4c58-8177-352cd79fdd41
md"""
### Momentum-based feature extractors
For more details see 
- J. Flusser and T. Suk, 
  '[Pattern recognition by affine moment invariants]
  (https://doi.org/10.1016/0031-3203(93)90098-H)',
  *Pattern Recognition* **26**(1),
  167--174 (1993). 

`flusser_affine_features(img::AbstractMatrix{<:Real})`


Compute the three Flusser–Suk affine moment invariants (I1, I2, I3)
for an image `img` (d x d, nonnegative intensities).

Returns a NamedTuple: (I1=..., I2=..., I3=...).

Behavior:
 - if the total mass μ00 == 0, returns zeros for all features.
 - any NaN/Inf in final results is replaced with 0.
"""

# ╔═╡ 8b17e769-baae-4960-8f0a-001dabf59f97
begin
	function central_moment(X::Matrix{Float64}, 
			Y::Matrix{Float64}, 
			I::Matrix{Float64}, 
			cx::Float64,
			cy::Float64,
			p::Int, 
			q::Int)
        # compute (X - cx).^p .* (Y - cy).^q .* I and sum
        # use broadcasting for small p,q
        return sum(((X .- cx) .^ p) .* ((Y .- cy) .^ q) .* I)
    end
	
	function moment_features(img::RestrictedSinogram; I4_true::Bool=true)
		I = img.data
	    d1, d2 = size(I)
		I ./= sum(I)
	
	    # coordinates: x = row index, y = col index
	    X = zeros(Float64, d1, d2)
	    Y = zeros(Float64, d1, d2)
	    for i in 1:d1, j in 1:d2
	        X[i,j] = i
	        Y[i,j] = j
	    end
	
	    mu00 = sum(I)
	    if mu00 == 0.0
	        return [0.0, 0.0, 0.0, 0.0]
	    end

	    m10 = sum(X .* I)
	    m01 = sum(Y .* I)
	    cx = m10 / mu00
	    cy = m01 / mu00
	
	    μ20 = central_moment(X,Y,I,cx,cy,2,0)
	    μ02 = central_moment(X,Y,I,cx,cy,0,2)
	    μ11 = central_moment(X,Y,I,cx,cy,1,1)
	    μ30 = central_moment(X,Y,I,cx,cy,3,0)
	    μ03 = central_moment(X,Y,I,cx,cy,0,3)
	    μ21 = central_moment(X,Y,I,cx,cy,2,1)
	    μ12 = central_moment(X,Y,I,cx,cy,1,2)
	
	    # compute invariants (using Flusser & Suk formulas)
		# I1 numerator as in the paper 
	    I1 = (μ20 * μ02 - μ11^2) / (mu00^4)
	
	    # I2 numerator as in the paper 
	    num2 = μ30^2 * μ03^2 -
	           6.0 * μ30 * μ21 * μ12 * μ03 +
	           4.0 * μ30 * μ12^3 +
	           4.0 * μ21^3 * μ03 -
	           3.0 * μ21^2 * μ12^2
	    I2 = num2 / (mu00^10)
	
	    # I3 numerator as in the paper 
	    num3 = μ20 * (μ21 * μ03 - μ12^2) -
	           μ11 * (μ30 * μ03 - μ21 * μ12) +
	           μ02 * (μ30 * μ12 - μ21^2)
	    I3 = num3 / (mu00^7)

		t1  =  (μ20^3) * (μ03^2)
	    t2  = -6.0 * (μ20^2) * μ11 * μ12 * μ03
	    t3  = -6.0 * (μ20^2) * μ02 * μ21 * μ03
		t4  = 9.0 * (μ20^2) * μ02 * (μ12^2)
	    t13  = 12.0 * μ20 * (μ11^2) * μ21 * μ03
	    t5  =  6.0 * μ20 * μ11 * μ02 * μ30 * μ03
	    t6  = -18.0 * μ20 * μ11 * μ02 * μ21 * μ12
	    t7  = -8.0  * (μ11^3) * μ30 * μ03
	    t8  = -6.0  * μ20 * (μ02^2) * μ30 * μ12
	    t9  =  9.0  * μ20 * (μ02^2) * (μ21^2)
	    t10 = 12.0  * (μ11^2) * μ02 * μ30 * μ12
	    t11 = -6.0  * μ11 * (μ02^2) * μ30 * μ21
	    t12 =       (μ02^3) * (μ30^2)
		
	
	    num4 = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + t10 + t11 + t12 + t13
	
	    I4 = num4 / (mu00^11)
	
	    # replace any NaN/Inf with 0 (user requested 0 for 0/0)
	    function clean(x)
	        if !isfinite(x) || isnan(x)
	            return 0.0
	        else
	            return x
	        end
	    end
	
	    if I4_true
			return clean.([I1, I2, I3, I4])
		else
			return clean.([I1, I2, I3])
		end
	end
end

# ╔═╡ c8a44724-ff30-4e23-a064-530acf930a00
begin
	MBJJ = moment_features.(JJ; I4_true=true)
	MBJClass = moment_features.(JClass; I4_true=true)
end

# ╔═╡ da431825-87f8-40ec-b721-66f57c7a283e
plot_quantiles(MBJJ, Label, MBJClass, Labels)

# ╔═╡ 40bef521-36c6-49e3-bde4-44416dc0d60e
begin
	accuracy_nearest_neighbour(MBJJ, Label, MBJClass, Labels, "manhattan", ret=1);
	accuracy_nearest_neighbour(MBJJ, Label, MBJClass, Labels, "euclidean", ret=1);
	accuracy_nearest_neighbour(MBJJ, Label, MBJClass, Labels, "inf", ret=1);
end

# ╔═╡ 75970283-62f4-4b1c-805b-74f8d32cf3c6
begin
	accuracy_k_nearest_part_neighbour(20, 1, 5, 3, MBJClass, Labels, "manhattan", K=1)
	accuracy_k_nearest_part_neighbour(20, 1, 5, 3, MBJClass, Labels, "euclidean", K=1)
	accuracy_k_nearest_part_neighbour(20, 1, 5, 3, MBJClass, Labels, "inf", K=1)
end

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╟─29ec4a5a-08a1-4194-a2fe-69e4d07150c5
# ╠═62aa8854-c7b6-4da4-a904-22d56ed5364c
# ╠═d07cf61a-edd0-4176-a2da-836f511490a1
# ╟─5ba7d520-b0cc-49dc-85b4-f7cb7920ae6c
# ╠═e9133a33-ec57-4ffb-850f-24da6a4d81a1
# ╠═fbb64f21-c677-4d7b-bef3-91594f4293f5
# ╠═bf3adcf5-0d1a-425c-96ba-819f95c281f6
# ╠═f98d22e0-ae6c-4870-b5b4-dab2bc1f3e18
# ╟─7af925b8-29fa-40ae-84bb-3bb95a96d415
# ╠═a23573e6-dfca-4fa2-b167-b131c427293e
# ╠═c3d8db69-40a7-415e-8324-90080c9aa84b
# ╠═8e1c4550-96d0-4e06-92ac-bfe53ecfea09
# ╠═22420609-2ed2-4102-8fe3-822783b9e490
# ╠═66130468-077c-4522-a25b-a37e041b7a64
# ╠═7460bd01-6bd0-4c81-b9cf-e78acfc0e380
# ╟─c859e69d-dec5-4d52-a875-ea22f28ebfa4
# ╠═3f35dfa9-c4d2-4773-8324-84959b014743
# ╠═e4abc6cc-790b-408f-b94b-7ac1a4a365b0
# ╠═9f2225ea-42bb-4d83-b74f-9fddf0a6aac5
# ╠═3353e434-20fc-4f76-baf6-1086f69da2a2
# ╠═fa74be1a-e0d4-424e-8bed-2b404123c272
# ╟─c6d48960-2bd5-417f-baa4-b6a6215e4882
# ╠═9c1f7df6-864d-474c-bcd4-3be47ad26c84
# ╠═1279f747-8661-45a8-b47f-df3e7927e473
# ╠═e22aaa88-1c56-4223-ae20-1f68f325871b
# ╠═71f4f715-61fb-4eb5-9a9c-b83dba91923f
# ╠═dee02cf0-1368-4bff-a229-fcc864711137
# ╠═8bc81f60-824c-44a3-843f-f10ff0e09036
# ╠═8caf0f7e-03dd-4312-93b1-44bc74fb9f2a
# ╠═871adfcd-1b4d-4569-a8f0-111a872e6f10
# ╠═a4a690d3-60bc-4efc-9fe7-a50d077ed785
# ╠═801bebde-dcdb-4f5f-b45e-04e24a08982f
# ╠═1fb20e89-8bdd-48fe-a6b3-2da865f72c79
# ╠═e08bf1ad-323b-4e21-932b-61c942170d3d
# ╟─bc032ecd-e12d-4e80-8d1c-6c10271aa09b
# ╠═7aba5b91-9476-419d-af7c-7e646c3090bb
# ╠═64cea78e-48c5-4fcd-9e09-02040f55d56f
# ╠═5cd92283-c65e-4d3a-9758-8402b1be7708
# ╠═178efc7f-6ffb-4c5f-8346-edcda2f5b968
# ╠═98972142-7556-4611-84c7-48ba84f84570
# ╠═dce6a3b3-9415-4473-b394-15681e0c6330
# ╠═5793a5cb-35ce-42d4-a196-c8985b61c269
# ╠═ce7191e1-b560-44dc-9170-a39964b466a2
# ╠═b301da2d-4368-4dda-8733-c1d40cf27671
# ╠═1349301a-16f3-4071-bb4e-4478bca4aeef
# ╟─9a9feb1c-8685-4c58-8177-352cd79fdd41
# ╠═8b17e769-baae-4960-8f0a-001dabf59f97
# ╠═c8a44724-ff30-4e23-a064-530acf930a00
# ╠═da431825-87f8-40ec-b721-66f57c7a283e
# ╠═40bef521-36c6-49e3-bde4-44416dc0d60e
# ╠═75970283-62f4-4b1c-805b-74f8d32cf3c6
