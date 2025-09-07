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
	Random.seed!(42)
	using GLMakie
	using LinearAlgebra
	using Interpolations: LinearInterpolation
	using Statistics
	using StatsBase
end

# ╔═╡ 9bc6bbed-558b-471e-bfdb-62d18897a910
I₁ = render(TestImages.Circle(scale=(0.35, 0.35)), size=(32,32), width=9.0);

# ╔═╡ d272426c-53c1-46f4-8d39-ab2f635ad683
J₁ = extend_image(I₁, (50, 50))

# ╔═╡ df653bcd-ed94-4909-91c9-b4b69d87d8c2
I₂ = render(TestImages.Square(scale=(0.4, 0.4)), size=(32,32), width=9.0);

# ╔═╡ 58dc61a9-8fe9-40e2-a201-0d0110c2eb6e
J₂ = extend_image(I₂, (50, 50))

# ╔═╡ 2819e101-b296-4286-90f5-3cc9f2988d49
I₂₁ = render(TestImages.Square(scale=(0.8, 0.8)), size=(50,50), width=4.0);

# ╔═╡ 7672b41f-e463-419f-8f54-b1f59cb28cf3
J₂₁ = extend_image(I₂₁, (70, 70))

# ╔═╡ a5299295-00de-4fee-9e23-3cf6d365da06
I₃ = render(TestImages.Triangle(scale=(0.35, 0.35)), size=(32,32), width=9.0);

# ╔═╡ 381b97d1-ff34-45f4-a8fa-1e640ad5e499
J₃ = extend_image(I₃, (50, 50))

# ╔═╡ 7253550c-918b-4031-830c-b688a678f13a
I₃₁ = render(TestImages.Triangle(scale=(0.8, 0.8)), size=(50,50), width=4.0);

# ╔═╡ 87216979-f9e5-4fa3-9174-5f6b8239e70d
J₃₁ = extend_image(I₃₁, (70, 70))

# ╔═╡ 1c01e7d8-967c-466a-9c34-5072e05e8cd6
I₄₁ = render(TestImages.Shield(TestImages.Circle(scale=(0.0, 0.0)),scale=(0.8, 0.8)), size=(50,50), width=4.0);

# ╔═╡ b8bbd2b1-d313-44c6-a052-483fc5df90a4
J₄₁ = extend_image(I₄₁, (70, 70))

# ╔═╡ 8d138aee-c81b-4c1d-b033-2e7bfe93bc92
I₁₁ = render(TestImages.Circle(scale=(0.8, 0.8)), size=(50,50), width=4.0);

# ╔═╡ 91143d01-9651-4e22-9feb-fbe005657cc3
J₁₁ = extend_image(I₁₁, (70, 70))

# ╔═╡ 50ef63a9-f9f9-4edd-a7aa-d24d24f56043
#Label = [1,2,3]; J = [ifelse.(J₁.>0.0, 1.0, 0.0), ifelse.(J₂.>0.0, 1.0, 0.0), ifelse.(J₃.>0.0, 1.0, 0.0)];
Label = [1,2,3]; J = [J₁, J₂, J₃];

# ╔═╡ fe5c32e2-1461-40ce-ac87-547816bf94af
#T3d = [generate_3d_rot(J₁₁, order=2.0), generate_3d_rot(J₂₁, order=20.0), generate_3d_rot(J₃₁, order=2.0)];
T3d = [generate_3d_rot(J₁₁, order=2.0), generate_3d_rot(J₂₁, order=20.0), generate_3d_rot(J₃₁, order=1.0)];

# ╔═╡ b6e1d0b8-0ce3-4a69-993d-cb3120e1d9be
Class3d, Labels3d = generate_academic_classes_3d(T3d, Label, class_size=5);

# ╔═╡ 4d870eb0-7e00-4786-ae1b-2588bbff1afd
# ╠═╡ disabled = true
#=╠═╡
for i in 1:n, j in 1:n, k in 1:n
	if sum(abs(i-(n+1)/2)^p + abs(j - (n+1)/2)^p + abs(k - (n+1)/2)^p)^(1/p) < n/4
		B[i,j,k] = 1.0
	end
	if sum(abs(i-(n+1)/2)^q + abs(j - (n+1)/2)^q + abs(k - (n+1)/2)^q)^(1/q) < n/4
		if sum(abs(i-(n+1)/2)^q + abs(j - (n+1)/2)^q + abs(k - (n+1)/2)^q)^(1/q) > n/6
			C[i,j,k] = 1.0
		end
	end
	if sum(abs(i-(n+1)/2)^(p) + abs(j - (n+1)/2)^(p))^(1/p) < n/4*max(1 - 2*(k-20)/n,0)
		if k > 20
			D[i,j,k] = 1.0
		end
	end
end
  ╠═╡ =#

# ╔═╡ 80a6eed2-1536-45ad-aff6-7376813b405b
volume(T3d[1], algorithm=:mip, colormap=:magma)

# ╔═╡ 9d98b722-0622-4e7d-a871-759e14e593d1
volume(T3d[2], algorithm=:mip, colormap=:magma)

# ╔═╡ a21c33ae-9bf4-45d3-baea-1b4000ea8d10
volume(T3d[3], algorithm=:mip, colormap=:magma)

# ╔═╡ 7a9b6963-3ecc-4ffb-9adc-9e32baa48981
A3d = RandomAffineTransformation3d(
	rotate_x = (-π,π),
	rotate_y = (-π,π),
	rotate_z = (-π,π),
	scale_x = (0.75, 1.1),
	scale_y = (0.75, 1.1),
	scale_z = (0.75, 1.1),
	shear_xy = (-0.15, 0.15),
	shear_yx = (-0.15, 0.15),
	shear_xz = (-0.15, 0.15),
	shear_zx = (-0.15, 0.15),
	shear_yz = (-0.15, 0.15),
	shear_zy = (-0.15, 0.15),
	shift_x = (-10,10),
	shift_y = (-10,10),
	shift_z = (-10,10)
)

# ╔═╡ 5049bc96-83b0-4260-a617-dd1e5d5d57e0
TClass3d = A3d.(Class3d);

# ╔═╡ ea8c4b3d-1c46-4349-9bf4-232874fbcaf7
for k in 1:length(TClass3d)
	TClass3d[k][isnan.(TClass3d[k])] .= 0.0;
	k += 1
end

# ╔═╡ 6efb2987-751a-4c7c-a05b-f243c67656d2
volume(TClass3d[1], algorithm=:mip, colormap=:magma)

# ╔═╡ d9b1fa46-18e7-43d5-90bc-ab6c70bab8c0
volume(TClass3d[6], algorithm=:mip, colormap=:magma)

# ╔═╡ b874e6a8-a12e-4327-a992-dcb620dd6bd9
volume(TClass3d[11], algorithm=:mip, colormap=:magma)

# ╔═╡ 636cd33d-93b2-42d4-9a66-26a89f89fc32
Plots.heatmap(TClass3d[11][:,30,:])

# ╔═╡ 8a161f3b-8d6d-4b6e-b113-3d40230dc609
R3d = RadonTransform3d(61,15,15)

# ╔═╡ 850f80f1-1d0e-4dbf-900b-837f8082b0af
begin
	struct RadonCDT3d 
	    quantiles::Int64
	    Radon3d::RadonTransform3d
	    signal_raising::Float64
	    function RadonCDT3d(q, R, ϵ)
	        if q <= 0
	            error("non-positive number of quantiles")
	        elseif ϵ <= 0
	            error("non-positive signal raising")
	        end
	        return new(q, R, ϵ)
	    end
	end
	
	
	RadonCDT3d(q, R) = RadonCDT3d(q, R, 1e-10)
	
	
	function (RCDT3d::RadonCDT3d)(image::AbstractArray)
	    S = RCDT3d.Radon3d(image)
	    Q = zeros(RCDT3d.quantiles, RCDT3d.Radon3d.angles_theta, RCDT3d.Radon3d.angles_phi)
	    for k in axes(Q, 2), l in axes(Q, 3)
	        Q[:, k, l] = cdt3d(S[:, k, l], RCDT3d)
	    end
	    return Q
	end
	
	function cdt3d(s::Vector{Float64}, RCDT3d::RadonCDT3d)
	    p = signal_to_density3d(s, RCDT3d.signal_raising)
	    P = cumsum(p)
	    s_grid = collect(LinRange(-1 - RCDT3d.Radon3d.width / 2, 1 + RCDT3d.Radon3d.width / 2, length(s)))
	    q_grid = collect(LinRange(0, 1, RCDT3d.quantiles + 2))[2:end - 1]
	    q = LinearInterpolation(P, s_grid)
	    return q(q_grid)
	end
	
	function signal_to_density3d(s::Vector{Float64}, ϵ::Float64)
	    p = s .+ ϵ
	    return p ./= sum(p)
	end
	
	struct NormRadonCDT3d
	    RCDT3d::RadonCDT3d
	end
	
	function (NRCDT3d::NormRadonCDT3d)(image::AbstractArray)
	    rcdt = NRCDT3d.RCDT3d(image)
	    nrcdt = (rcdt .- mean(rcdt, dims=1)) ./ std(rcdt, dims=1)
	    return nrcdt
	end
	
	struct MaxNormRadonCDT3d
	    RCDT3d::RadonCDT3d
	end
	
	function (mNRCDT3d::MaxNormRadonCDT3d)(image::AbstractArray)
	    NRCDT3d = NormRadonCDT3d(mNRCDT3d.RCDT3d)
	    nrcdt = NRCDT3d(image)
	    mnrcdt = dropdims(maximum(dropdims(maximum(nrcdt, dims=2), dims=2), dims=2))
	    return mnrcdt
	end
	
	function normalization3d(rcdt::AbstractArray)
	    nrcdt = (rcdt .- mean(rcdt, dims=1)) ./ std(rcdt, dims=1)
	    return nrcdt
	end
	
	function max_normalization3d(rcdt::AbstractArray)
	    nrcdt = normalization3d(rcdt)
	    (nq, nt, np) = size(nrcdt)
	    mnrcdt = dropdims(maximum(reshape(nrcdt, (nq, nt*np)), dims=2), dims=2)
	    return mnrcdt
	end
	
	function mean_normalization3d(rcdt::AbstractArray)
	    nrcdt = normalization3d(rcdt)
	    (nq, nt, np) = size(nrcdt)
	    mnrcdt = dropdims(mean(reshape(nrcdt, (nq, nt*np)), dims=2), dims=2)
	    return mnrcdt
	end
end

# ╔═╡ 731ffb61-93f6-424a-927e-c1f24f878a75
RCDT3d = RadonCDT3d(64, R3d);

# ╔═╡ 7fba80c5-6ab2-4fec-a7aa-b6924ef73e83
NRCDT3d = NormRadonCDT3d(RCDT3d);

# ╔═╡ 01021e2f-b2e5-49ac-a26c-9f771f481d5c
qTempNRCDT3d = NRCDT3d.RCDT3d.(T3d);

# ╔═╡ c1cca820-4187-4d52-a787-3b8637314dd5
mqTempNRCDT3d = max_normalization3d.(qTempNRCDT3d);

# ╔═╡ 80306a6d-18ac-4081-8c9e-6ca24ed714c9
aqTempNRCDT3d = mean_normalization3d.(qTempNRCDT3d);

# ╔═╡ 34005741-e70c-4551-b8a1-91523224ec47
Plots.plot(mqTempNRCDT3d[1]);Plots.plot!(mqTempNRCDT3d[2]);Plots.plot!(mqTempNRCDT3d[3])

# ╔═╡ 2ec72be6-4fd3-48cd-9413-2c45db23920b
Plots.plot(aqTempNRCDT3d[1]);Plots.plot!(aqTempNRCDT3d[2]);Plots.plot!(aqTempNRCDT3d[3])

# ╔═╡ be66aa46-d1f0-407f-be31-d1fc00301a59
qClassNRCDT3d = NRCDT3d.RCDT3d.(TClass3d);

# ╔═╡ db1f64c4-0abb-4a43-b3cc-0e4e2d1e9e5b
mqClassNRCDT3d = max_normalization3d.(qClassNRCDT3d);

# ╔═╡ 3c2be1fa-679a-4cd8-8330-3e9ab7325cc8
aqClassNRCDT3d = mean_normalization3d.(qClassNRCDT3d);

# ╔═╡ 0af3a515-3f5f-41e2-9bde-0b564a911d4e
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

# ╔═╡ 370c4169-ec37-496b-bda9-aa268c25fffd
p1 = plot_quantiles([mqTempNRCDT3d[3]],[3],mqClassNRCDT3d[11:15], Labels3d[11:15])

# ╔═╡ 8531c205-80e4-4357-96ba-d3e6b07db6cd
# ╠═╡ disabled = true
#=╠═╡
savefig(p1, "3d_max_quantiles_61_shield_3_large.pdf")
  ╠═╡ =#

# ╔═╡ fd92a21c-69e5-483c-9d1a-291b021d886c
p = plot_quantiles(mqTempNRCDT3d,[1,2,3],mqClassNRCDT3d, Labels3d)

# ╔═╡ 5a9cee73-f2bc-4de6-989b-1f09cf0d4c90
# ╠═╡ disabled = true
#=╠═╡
savefig(p, "3d_max_quantiles_61_shield_large.pdf")
  ╠═╡ =#

# ╔═╡ 4311880a-278b-492a-8d3a-c225cb56d9ae
ap = plot_quantiles(aqTempNRCDT3d,[1,2,3],aqClassNRCDT3d, Labels3d)

# ╔═╡ cc7655c2-37ff-4a00-9ef4-49d8fefeef50
# ╠═╡ disabled = true
#=╠═╡
savefig(ap, "3d_mean_quantiles_61_shield_large.pdf")
  ╠═╡ =#

# ╔═╡ 3c940e91-99c5-44d5-8ea6-2f1ff616ed36
QQQ2 = zeros(15,3); QQQi = zeros(15,3); QQQ1 = zeros(15,3); QQQ2_a = zeros(15,3); QQQi_a = zeros(15,3); QQQ1_a = zeros(15,3);

# ╔═╡ b06db212-d791-437c-a5c8-0073f84fa4ed
for il in 1:15, jl in 1:3
	QQQ2[il, jl] = sqrt(sum(abs.(mqClassNRCDT3d[il].-mqTempNRCDT3d[jl]).^2))
	QQQi[il, jl] = maximum(abs.(mqClassNRCDT3d[il].-mqTempNRCDT3d[jl]))
	QQQ1[il, jl] = sum(abs.(mqClassNRCDT3d[il].-mqTempNRCDT3d[jl]))
	QQQ2_a[il, jl] = sqrt(sum(abs.(aqClassNRCDT3d[il].-aqTempNRCDT3d[jl]).^2))
	QQQi_a[il, jl] = maximum(abs.(aqClassNRCDT3d[il].-aqTempNRCDT3d[jl]))
	QQQ1_a[il, jl] = sum(abs.(aqClassNRCDT3d[il].-aqTempNRCDT3d[jl]))
end

# ╔═╡ dad9ea86-d816-4061-937e-f64aedd8a0a2
fig = Figure(resolution=(1200,400)); ax1 = Makie.Axis(fig[1,1], title = "L₂"); h1 = Makie.heatmap!(ax1, QQQ2, colormap=:heat) ; ax1.yticks= (1:3,["1", "2", "3"]); Colorbar(fig[1,2], h1); ax2 = Makie.Axis(fig[2,1], title = "L∞"); h2 = Makie.heatmap!(ax2, QQQi, colormap=:heat); ax2.yticks= (1:3,["1", "2", "3"]); ax2.yticks= (1:3,["1", "2", "3"]); Colorbar(fig[2,2], h2); ax3 = Makie.Axis(fig[3,1], title="L₁"); h3= Makie.heatmap!(ax3, QQQ1, colormap=:heat); ax3.yticks= (1:3,["1", "2", "3"]); ax3.yticks= (1:3,["1", "2", "3"]); Colorbar(fig[3,2], h3);

# ╔═╡ 2660f3cb-6ab2-433d-8e69-51d14b024826
fig

# ╔═╡ 3f6a7c2c-a0b1-451f-a6a0-00dfbd99da33
# ╠═╡ disabled = true
#=╠═╡
save("3d_max_mdis_61_shield_large.png", fig)
  ╠═╡ =#

# ╔═╡ 775926fc-9545-40ca-8140-7c5b79f950b2
fig_a = Figure(resolution=(1200,400)); ax1_a = Makie.Axis(fig_a[1,1], title = "L₂"); h1_a = Makie.heatmap!(ax1_a, QQQ2_a, colormap=:heat) ; ax1_a.yticks= (1:3,["1", "2", "3"]); Colorbar(fig_a[1,2], h1_a); ax2_a = Makie.Axis(fig_a[2,1], title = "L∞"); h2_a = Makie.heatmap!(ax2_a, QQQi_a, colormap=:heat); ax2_a.yticks= (1:3,["1", "2", "3"]); ax2_a.yticks= (1:3,["1", "2", "3"]); Colorbar(fig_a[2,2], h2_a); ax3_a = Makie.Axis(fig_a[3,1], title="L₁"); h3_a= Makie.heatmap!(ax3_a, QQQ1_a, colormap=:heat); ax3_a.yticks= (1:3,["1", "2", "3"]); ax3_a.yticks= (1:3,["1", "2", "3"]); Colorbar(fig_a[3,2], h3_a);

# ╔═╡ 68824724-6dde-43af-8361-94d914884fc3
fig_a

# ╔═╡ 05c9666a-5f1d-484b-b0e0-78430dd6b097
# ╠═╡ disabled = true
#=╠═╡
save("3d_mean_mdis_61_shield_large.png", fig_a)
  ╠═╡ =#

# ╔═╡ 3ccdda15-c2ef-43e3-a5f9-de4dafbeda9a
aQ1 = argmin(QQQ1, dims=2); aQi = argmin(QQQi, dims=2); aQ2 = argmin(QQQ2, dims=2);

# ╔═╡ 439a3cfd-894c-4242-bd2e-44f7cd157ba0
accuracy_nearest_neighbour(mqTempNRCDT3d, [1,2,3], mqClassNRCDT3d, Labels3d, "inf", ret=1);

# ╔═╡ adb2172c-47f4-4164-91ba-91497346dea4
accuracy_nearest_neighbour(mqTempNRCDT3d, [1,2,3], mqClassNRCDT3d, Labels3d, "manhattan", ret=1);

# ╔═╡ b557ae0a-38e2-430d-ac2d-4493dcc542a4
accuracy_nearest_neighbour(mqTempNRCDT3d, [1,2,3], mqClassNRCDT3d, Labels3d, "euclidean", ret=1);

# ╔═╡ c10f0dd4-77c5-4cb3-b7dc-f0c9e410538e
accuracy_nearest_neighbour(aqTempNRCDT3d, [1,2,3], aqClassNRCDT3d, Labels3d, "inf", ret=1);

# ╔═╡ ee6a18ab-b845-4910-90a3-786e073b6a5a
accuracy_nearest_neighbour(aqTempNRCDT3d, [1,2,3], aqClassNRCDT3d, Labels3d, "manhattan", ret=1);

# ╔═╡ 73887103-8c49-4b94-8056-2706db9369d5
accuracy_nearest_neighbour(aqTempNRCDT3d, [1,2,3], aqClassNRCDT3d, Labels3d, "euclidean", ret=1);

# ╔═╡ 74205cd4-ab30-4c7f-bc22-2bc261026649
accuracy_k_nearest_part_neighbour(10, 1, 5, 3, mqClassNRCDT3d, Labels3d, "inf", K=1)

# ╔═╡ afbb15c7-e4b9-4086-be65-733511005192
accuracy_k_nearest_part_neighbour(10, 1, 5, 3, mqClassNRCDT3d, Labels3d, "euclidean", K=1)

# ╔═╡ cd6537bd-8d4c-44f0-858e-1878776dbabf
accuracy_k_nearest_part_neighbour(10, 1, 5, 3, aqClassNRCDT3d, Labels3d, "inf", K=1)

# ╔═╡ 3ac4e1a5-a0af-47c4-8f94-e51ac61f2c9b
accuracy_k_nearest_part_neighbour(10, 1, 5, 3, aqClassNRCDT3d, Labels3d, "euclidean", K=1)

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╠═9bc6bbed-558b-471e-bfdb-62d18897a910
# ╠═d272426c-53c1-46f4-8d39-ab2f635ad683
# ╠═df653bcd-ed94-4909-91c9-b4b69d87d8c2
# ╠═58dc61a9-8fe9-40e2-a201-0d0110c2eb6e
# ╠═2819e101-b296-4286-90f5-3cc9f2988d49
# ╠═7672b41f-e463-419f-8f54-b1f59cb28cf3
# ╠═a5299295-00de-4fee-9e23-3cf6d365da06
# ╠═381b97d1-ff34-45f4-a8fa-1e640ad5e499
# ╠═7253550c-918b-4031-830c-b688a678f13a
# ╠═87216979-f9e5-4fa3-9174-5f6b8239e70d
# ╠═1c01e7d8-967c-466a-9c34-5072e05e8cd6
# ╠═b8bbd2b1-d313-44c6-a052-483fc5df90a4
# ╠═8d138aee-c81b-4c1d-b033-2e7bfe93bc92
# ╠═91143d01-9651-4e22-9feb-fbe005657cc3
# ╠═50ef63a9-f9f9-4edd-a7aa-d24d24f56043
# ╠═fe5c32e2-1461-40ce-ac87-547816bf94af
# ╠═b6e1d0b8-0ce3-4a69-993d-cb3120e1d9be
# ╠═4d870eb0-7e00-4786-ae1b-2588bbff1afd
# ╠═80a6eed2-1536-45ad-aff6-7376813b405b
# ╠═9d98b722-0622-4e7d-a871-759e14e593d1
# ╠═a21c33ae-9bf4-45d3-baea-1b4000ea8d10
# ╠═7a9b6963-3ecc-4ffb-9adc-9e32baa48981
# ╠═5049bc96-83b0-4260-a617-dd1e5d5d57e0
# ╠═ea8c4b3d-1c46-4349-9bf4-232874fbcaf7
# ╠═6efb2987-751a-4c7c-a05b-f243c67656d2
# ╠═d9b1fa46-18e7-43d5-90bc-ab6c70bab8c0
# ╠═b874e6a8-a12e-4327-a992-dcb620dd6bd9
# ╠═636cd33d-93b2-42d4-9a66-26a89f89fc32
# ╠═8a161f3b-8d6d-4b6e-b113-3d40230dc609
# ╠═850f80f1-1d0e-4dbf-900b-837f8082b0af
# ╠═731ffb61-93f6-424a-927e-c1f24f878a75
# ╠═7fba80c5-6ab2-4fec-a7aa-b6924ef73e83
# ╠═01021e2f-b2e5-49ac-a26c-9f771f481d5c
# ╠═c1cca820-4187-4d52-a787-3b8637314dd5
# ╠═80306a6d-18ac-4081-8c9e-6ca24ed714c9
# ╠═34005741-e70c-4551-b8a1-91523224ec47
# ╠═2ec72be6-4fd3-48cd-9413-2c45db23920b
# ╠═be66aa46-d1f0-407f-be31-d1fc00301a59
# ╠═db1f64c4-0abb-4a43-b3cc-0e4e2d1e9e5b
# ╠═3c2be1fa-679a-4cd8-8330-3e9ab7325cc8
# ╠═0af3a515-3f5f-41e2-9bde-0b564a911d4e
# ╠═370c4169-ec37-496b-bda9-aa268c25fffd
# ╠═8531c205-80e4-4357-96ba-d3e6b07db6cd
# ╠═fd92a21c-69e5-483c-9d1a-291b021d886c
# ╠═5a9cee73-f2bc-4de6-989b-1f09cf0d4c90
# ╠═4311880a-278b-492a-8d3a-c225cb56d9ae
# ╠═cc7655c2-37ff-4a00-9ef4-49d8fefeef50
# ╠═3c940e91-99c5-44d5-8ea6-2f1ff616ed36
# ╠═b06db212-d791-437c-a5c8-0073f84fa4ed
# ╠═dad9ea86-d816-4061-937e-f64aedd8a0a2
# ╠═2660f3cb-6ab2-433d-8e69-51d14b024826
# ╠═3f6a7c2c-a0b1-451f-a6a0-00dfbd99da33
# ╠═775926fc-9545-40ca-8140-7c5b79f950b2
# ╠═68824724-6dde-43af-8361-94d914884fc3
# ╠═05c9666a-5f1d-484b-b0e0-78430dd6b097
# ╠═3ccdda15-c2ef-43e3-a5f9-de4dafbeda9a
# ╠═439a3cfd-894c-4242-bd2e-44f7cd157ba0
# ╠═adb2172c-47f4-4164-91ba-91497346dea4
# ╠═b557ae0a-38e2-430d-ac2d-4493dcc542a4
# ╠═c10f0dd4-77c5-4cb3-b7dc-f0c9e410538e
# ╠═ee6a18ab-b845-4910-90a3-786e073b6a5a
# ╠═73887103-8c49-4b94-8056-2706db9369d5
# ╠═74205cd4-ab30-4c7f-bc22-2bc261026649
# ╠═afbb15c7-e4b9-4086-be65-733511005192
# ╠═cd6537bd-8d4c-44f0-858e-1878776dbabf
# ╠═3ac4e1a5-a0af-47c4-8f94-e51ac61f2c9b
