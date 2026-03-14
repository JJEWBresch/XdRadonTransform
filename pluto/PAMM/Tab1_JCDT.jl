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

# ╔═╡ acdbfebc-569f-4775-a57c-2d06fcf567a5
md"""
# arXiv:xxxx.xxxxx -- Table 1, Figure 2
This pluto notebook reproduces the numerical visualizations
with tamplates for Figure 1 with templates from Figure 2 from

- Robert Beinert, Jonas Bresch, and Michael Quellmalz
  '*John Transform for Object Classification*',
  arXiv:xxxx.xxxxx, 2026.
"""

# ╔═╡ e88b16dd-b1b4-443b-b369-638e2cc741f1
md"""
## Templates
Generate the three templates
using the submodule `TestImages`.
"""

# ╔═╡ e9133a33-ec57-4ffb-850f-24da6a4d81a1
begin
	stroke_width = 4;
	blur = 5; #15
end

# ╔═╡ fbb64f21-c677-4d7b-bef3-91594f4293f5
begin
	I₁_sharp = render(OrbAndCross(TestImages.Circle(),Star(1)), width=stroke_width, size=(64,64));
	I₁ = augment(I₁_sharp, GaussianBlur(blur));
	J₁ = extend_image(I₁, (128, 128))
end

# ╔═╡ bf3adcf5-0d1a-425c-96ba-819f95c281f6
begin
	I₂_sharp = render(OrbAndCross(Square(),Star(4)), width=stroke_width, size=(64,64));
	I₂ = augment(I₂_sharp, GaussianBlur(blur));
	J₂ = extend_image(I₂, (128, 128))
end

# ╔═╡ f98d22e0-ae6c-4870-b5b4-dab2bc1f3e18
begin
	I₃_sharp = render(Shield(Triangle()), width=stroke_width, size=(64,64));
	I₃ = augment(I₃_sharp, GaussianBlur(blur));
	J₃ = extend_image(I₃, (128, 128))
end

# ╔═╡ a5056c59-52d5-4082-9c78-9571744f85cf
md"""
## Dataset
Generate the dataset 
by duplicating the templates
and by applying random affine transformations
using the submodule `DataTransformations`.
"""

# ╔═╡ 5490758d-1a00-443d-acf0-098caa280dd9
J = [J₁, J₂, J₃]; Label = [1, 5, 12];

# ╔═╡ c3d8db69-40a7-415e-8324-90080c9aa84b
Class, Labels = generate_academic_classes(J, Label, class_size=10);

# ╔═╡ 8e1c4550-96d0-4e06-92ac-bfe53ecfea09
A = DataTransformations.RandomAffineTransformation(
	# scale_x = (0.75, 1.25), 
	# scale_y = (0.75, 1.25),
	scale_i = (0.5, 1.5),
	# rotate=(-10.0, 10.0), 
	# shear_x=(-30.0, 30.0),
	# shear_y=(-30.0, 30.0),
	shift_x=(-20, 20),
	shift_y=(-20, 20)
	)

# ╔═╡ 22420609-2ed2-4102-8fe3-822783b9e490
begin
	Random.seed!(42424)
	TClass = A.(Class)
end

# ╔═╡ 7b5cac8e-c3d7-48c0-b3ab-bcc3a6f964e9
p = Plots.heatmap(TClass[24], axis=false, grid=false)

# ╔═╡ aeeb495e-ffa3-4176-b628-9a5b101a5d19
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
	d = 2
	ϰ = 256
	ℓ = 64
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
Plots.plot(norm.(Jθ.tangentspace))

# ╔═╡ 6eac02a9-2679-43b5-be9f-bbe54632021d
md"""
## Normalized John-CDT -- Figure xxx
Setup the max-normalized RCDT,
and apply it 
to the dataset and templates.
"""

# ╔═╡ 9c1f7df6-864d-474c-bcd4-3be47ad26c84
begin
	function njcdt(ss::RestrictedSinogram, 
		quantiles::Int64; 
		signal_raising=1e-10::Float64)
		jcdt = cdt(ss, quantiles; signal_raising)
		njcdt = ncdt(jcdt)
		return njcdt
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
	
	function signal_to_density(s::Vector{Float64}, ϵ::Float64)
	    p = s .+ ϵ
	    return p ./= sum(p)
	end

	function ncdt(jcdt::Vector{Float64})
		njcdt = (jcdt .- mean(jcdt)) ./ std(jcdt)
		return njcdt
	end
end

# ╔═╡ 1279f747-8661-45a8-b47f-df3e7927e473
begin
	PJ = PhantomBox.(J)
	PTClass = PhantomBox.(TClass)
	JJ = Jθ.(PJ)
	JClass = Jθ.(PTClass)
end

# ╔═╡ 801bebde-dcdb-4f5f-b45e-04e24a08982f
begin
	Plots.plot(JJ[1].data)
	Plots.plot!(JJ[2].data)
	Plots.plot!(JJ[3].data)
end

# ╔═╡ b8ff68fd-b7e2-4765-bd1d-cd773503ff75
begin
	Plots.plot(JClass[1].data)
	Plots.plot!(JClass[2].data)
	Plots.plot!(JClass[3].data)
end

# ╔═╡ 1fb20e89-8bdd-48fe-a6b3-2da865f72c79
begin
	JCDT_T = njcdt.(JJ, 128)
	JCDT_TClass = njcdt.(JClass, 128)
end

# ╔═╡ 8a844c4e-4e36-46c6-b611-f820827a345c
begin
	Plots.plot(JCDT_TClass[1], c=:red)
	Plots.plot!(JCDT_TClass[2], c=:red)
	Plots.plot!(JCDT_TClass[3], c=:red)
	Plots.plot!(JCDT_TClass[11], c=:green)
	Plots.plot!(JCDT_TClass[12], c=:green)
	Plots.plot!(JCDT_TClass[13], c=:green)
	Plots.plot!(JCDT_TClass[21], c=:blue)
	Plots.plot!(JCDT_TClass[22], c=:blue)
	Plots.plot!(JCDT_TClass[23], c=:blue)
end

# ╔═╡ 8ff6fd3e-0b73-4b87-8fa8-bed3936fe4ca
begin
	Plots.plot(JCDT_TClass[1], c=:red)
	Plots.plot!(JCDT_TClass[2], c=:red)
	Plots.plot!(JCDT_TClass[3], c=:red)
	Plots.plot!(JCDT_TClass[4], c=:red)
end

# ╔═╡ a47d598b-5f0a-456a-98e0-e7b1a241fc1a
md"""
## Nearest Neighbour Classification -- Table xxx
Use the nearest neighbour classification
with respect to the chosen templates
to classify the generated dataset.
The normalized J-CDT is applied
with different numbers of used angles.
"""

# ╔═╡ fd6076a0-a970-43a9-9a10-ff98b37ae53e
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

# ╔═╡ 17648cfc-2ddb-4ca3-b1c4-d11b5e796440
plot_quantiles(JCDT_T, Label, JCDT_TClass, Labels)

# ╔═╡ 2343c9eb-bd00-4747-bdc1-96b8bfcd0440
begin
	accuracy_nearest_neighbour(JCDT_T, Label, JCDT_TClass, Labels, "manhattan", ret=1);
	accuracy_nearest_neighbour(JCDT_T, Label, JCDT_TClass, Labels, "euclidean", ret=1);
	accuracy_nearest_neighbour(JCDT_T, Label, JCDT_TClass, Labels, "inf", ret=1);
end

# ╔═╡ a66946af-387c-48fa-baf0-b41689385bae
begin
	accuracy_k_nearest_part_neighbour(20, 5, 10, 3, JCDT_TClass, Labels, "manhattan", K=1)
	accuracy_k_nearest_part_neighbour(20, 5, 10, 3, JCDT_TClass, Labels, "euclidean", K=1)
	accuracy_k_nearest_part_neighbour(20, 5, 10, 3, JCDT_TClass, Labels, "inf", K=1)
end

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╟─acdbfebc-569f-4775-a57c-2d06fcf567a5
# ╠═62aa8854-c7b6-4da4-a904-22d56ed5364c
# ╠═d07cf61a-edd0-4176-a2da-836f511490a1
# ╟─e88b16dd-b1b4-443b-b369-638e2cc741f1
# ╠═e9133a33-ec57-4ffb-850f-24da6a4d81a1
# ╠═fbb64f21-c677-4d7b-bef3-91594f4293f5
# ╠═bf3adcf5-0d1a-425c-96ba-819f95c281f6
# ╠═f98d22e0-ae6c-4870-b5b4-dab2bc1f3e18
# ╟─a5056c59-52d5-4082-9c78-9571744f85cf
# ╠═5490758d-1a00-443d-acf0-098caa280dd9
# ╠═c3d8db69-40a7-415e-8324-90080c9aa84b
# ╠═8e1c4550-96d0-4e06-92ac-bfe53ecfea09
# ╠═22420609-2ed2-4102-8fe3-822783b9e490
# ╠═7b5cac8e-c3d7-48c0-b3ab-bcc3a6f964e9
# ╟─aeeb495e-ffa3-4176-b628-9a5b101a5d19
# ╠═3f35dfa9-c4d2-4773-8324-84959b014743
# ╠═e4abc6cc-790b-408f-b94b-7ac1a4a365b0
# ╠═9f2225ea-42bb-4d83-b74f-9fddf0a6aac5
# ╠═3353e434-20fc-4f76-baf6-1086f69da2a2
# ╠═fa74be1a-e0d4-424e-8bed-2b404123c272
# ╟─6eac02a9-2679-43b5-be9f-bbe54632021d
# ╠═9c1f7df6-864d-474c-bcd4-3be47ad26c84
# ╠═1279f747-8661-45a8-b47f-df3e7927e473
# ╠═801bebde-dcdb-4f5f-b45e-04e24a08982f
# ╠═b8ff68fd-b7e2-4765-bd1d-cd773503ff75
# ╠═1fb20e89-8bdd-48fe-a6b3-2da865f72c79
# ╠═8a844c4e-4e36-46c6-b611-f820827a345c
# ╠═8ff6fd3e-0b73-4b87-8fa8-bed3936fe4ca
# ╟─a47d598b-5f0a-456a-98e0-e7b1a241fc1a
# ╠═fd6076a0-a970-43a9-9a10-ff98b37ae53e
# ╠═17648cfc-2ddb-4ca3-b1c4-d11b5e796440
# ╠═2343c9eb-bd00-4747-bdc1-96b8bfcd0440
# ╠═a66946af-387c-48fa-baf0-b41689385bae
