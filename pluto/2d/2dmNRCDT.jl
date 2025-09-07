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
	#using GLMakie
	using RadonKA
	using LinearAlgebra
	using Interpolations: LinearInterpolation
	using Statistics
end

# ╔═╡ b2a669f2-34c3-48db-9a72-feefbc3b86aa
# 32/50

# ╔═╡ 9bc6bbed-558b-471e-bfdb-62d18897a910
I₁ = render(OrbAndCross(TestImages.Circle(scale=(0.7, 0.7)),Star(1)), size=(128,128), width=2.0);

# ╔═╡ d272426c-53c1-46f4-8d39-ab2f635ad683
J₁ = extend_image(I₁, (256, 256))

# ╔═╡ 61d09b61-1038-42dc-84b1-cbcd37b10a6e
I₂ = render(OrbAndCross(TestImages.Circle(scale=(0.7, 0.7)),Star(4)), size=(128,128), width=2.0);

# ╔═╡ aa47398a-b248-48fd-943e-b349b6e8f6a8
J₂ = extend_image(I₂, (256, 256))

# ╔═╡ 29700517-53ff-493a-acec-e4eac297eee1
I₃ = render(OrbAndCross(TestImages.Circle(scale=(0.7, 0.7)),Star(5)), size=(128,128), width=2.0);

# ╔═╡ 0f0ba691-f135-4647-8039-c6ba6ff09eea
J₃ = extend_image(I₃, (256, 256))

# ╔═╡ 5a04ab89-1ab9-480c-97a1-b47bdf0baedb
I₅ = render(OrbAndCross(TestImages.Circle(scale=(0.7, 0.7)),Star(1)), size=(32,32), width=1.0);

# ╔═╡ 6ae8e856-1eed-4fe0-be60-49d346ce7ecb
J₅ = extend_image(I₅, (50, 50))

# ╔═╡ e1239f0f-246c-44a8-95f8-efca6a01d0c8
I₆ = render(OrbAndCross(TestImages.Circle(scale=(0.7, 0.7)),Star(4)), size=(32,32), width=1.0);

# ╔═╡ 21d5fd97-3d8b-4bc5-8a6a-0f77ca9ef8d5
J₆ = extend_image(I₆, (50, 50))

# ╔═╡ 6924e56a-973d-4398-83a6-be235fa61252
I₇ = render(OrbAndCross(TestImages.Circle(scale=(0.7, 0.7)),Star(5)), size=(32,32), width=1.0);

# ╔═╡ 2a3b392c-e818-44d2-939a-5f90354141ca
J₇ = extend_image(I₇, (50, 50))

# ╔═╡ 50ef63a9-f9f9-4edd-a7aa-d24d24f56043
# ╠═╡ disabled = true
#=╠═╡
Label=[1,2,3,4]; J = [J₁, J₂, J₃, J₄]
  ╠═╡ =#

# ╔═╡ 9026e79e-ce2c-4b5b-987b-c4ba7e340d09
Label=[1,2,3]; J = [J₅, J₆, J₇] #J = [J₁, J₂, J₃] 

# ╔═╡ 4f5b3b5e-e4b5-497c-8a59-c40e05249274
Class, Labels = generate_academic_classes(J, Label, class_size=10);

# ╔═╡ 76b23278-16ea-411a-8771-8606d8e14b96
A = DataTransformations.RandomAffineTransformation(
	scale_x = (0.5, 1.15), 
	scale_y = (0.5, 1.15),
	rotate=(-180.0, 180.0), 
	shear_x=(-25.0, 25.0),
	shear_y=(-25.0, 25.0),
	shift_x=(-10, 10),
	shift_y=(-10, 10))

# ╔═╡ affee424-d71b-4b18-81f6-f6a7c6ab95ac
TClass = A.(Class)

# ╔═╡ 6b12a034-c756-47cd-aab2-7be920fd0fc1
# ╠═╡ disabled = true
#=╠═╡
savefig(Plots.heatmap(TClass[26], axis=nothing, frame=:none), "academic_33_large.pdf")
  ╠═╡ =#

# ╔═╡ 73584882-fb8b-44bc-9769-a8b6f715ae25
begin
	struct RadonCDT 
	    quantiles::Int64
	    Radon::RadonTransform
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
	
	function (RCDT::RadonCDT)(image::AbstractMatrix)
	    S = RCDT.Radon(image)
	    Q = zeros(RCDT.quantiles, RCDT.Radon.angles)
	    for k in axes(Q, 2)
	        Q[:, k] = cdt(S[:, k], RCDT)
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
	
	function (NRCDT::NormRadonCDT)(image::AbstractMatrix)
	    rcdt = NRCDT.RCDT(image)
	    nrcdt = (rcdt .- mean(rcdt, dims=1)) ./ std(rcdt, dims=1)
	    return nrcdt
	end
	
	struct MaxNormRadonCDT
	    RCDT::RadonCDT
	end
	
	function (mNRCDT::MaxNormRadonCDT)(image::AbstractMatrix)
	    NRCDT = NormRadonCDT(mNRCDT.RCDT)
	    nrcdt = NRCDT(image)
	    mnrcdt = dropdims(maximum(nrcdt, dims=2), dims=2)
	    return mnrcdt
	end
	
	struct MeanNormRadonCDT
	    RCDT::RadonCDT
	end
	
	function (aNRCDT::MeanNormRadonCDT)(image::AbstractMatrix)
	    NRCDT = NormRadonCDT(aNRCDT.RCDT)
	    nrcdt = NRCDT(image)
	    anrcdt = dropdims(mean(nrcdt, dims=2), dims=2)
	    return anrcdt
	end
	
	function filter_angles(rcdt::AbstractMatrix, num_angle::Int64, angles::Int64)
	    choice_angles = sort(Int64.(angles+1 .- collect(range(0,angles,length=num_angle+1))))[1:num_angle]
	    filter_rcdt = rcdt[:,choice_angles]
	    return filter_rcdt
	end
	
	function normalization(rcdt::AbstractArray)
	    nrcdt = (rcdt .- mean(rcdt, dims=1)) ./ std(rcdt, dims=1)
	    return nrcdt
	end
	
	function max_normalization(rcdt::AbstractArray)
	    mnrcdt = dropdims(maximum(normalization(rcdt), dims=2), dims=2)
	    return mnrcdt
	end
	
	function mean_normalization(rcdt::AbstractArray)
	    mnrcdt = dropdims(mean(normalization(rcdt), dims=2), dims=2)
	    return mnrcdt
	end
end

# ╔═╡ 4beeac67-a9bb-45c5-95b1-d0b556af9561
R = RadonTransform(501,16,0.0); RCDT = RadonCDT(64, R); qRCDT = RCDT.(TClass); qtRCDT = RCDT.(J); #501,128

# ╔═╡ 328b0da3-2319-4bf4-8536-64ca2b38534e
mqTemp = max_normalization.(qtRCDT); mqClass = max_normalization.(qRCDT);

# ╔═╡ f329081e-fa20-495d-ad78-db03ed6d7859
"""
    radon(f::Matrix{T}; θ=Vector(0:(π/180):π-π/180)::Vector{T}, in_circle=true) where {T<:Real}

return Radon transform of `f`.

# Keyword arguments

- `f`: 2-dimension array.(phantom image)
- `θ`: projection angle.(radian)
- `in_circle`: 
"""
function radon_transform(f::AbstractArray, θ::AbstractArray; in_circle=true)
    ℛf = zeros(maximum(size(f)), length(θ))
    
    slim, θlim = size(ℛf)
    ylim, xlim = size(f)

    if in_circle == true
        L = ylim
    else
        L = round(Int64, hypot(xlim, ylim))
    end
    
    cos_ = cos.(θ)
    sin_ = sin.(θ)

    s = LinRange(-L/2, L/2, slim)
    t = LinRange(-L/2, L/2, L)
    
    for sᵢ ∈ 1:slim, θᵢ ∈ 1:θlim
        for l ∈ 1:L
            x = round(Int64, xlim/2 + s[sᵢ]*cos_[θᵢ] + t[l]*sin_[θᵢ])
            y = round(Int64, ylim/2 - t[l]*cos_[θᵢ] + s[sᵢ]*sin_[θᵢ])
            if 1 ≤ x ≤ xlim && 1 ≤ y ≤ ylim
                ℛf[sᵢ, θᵢ] += f[end-y+1, x]
            end
        end
    end
    
    return ℛf
end

# ╔═╡ d6fff828-24ca-40e6-b16f-a09c70c7650e
pra = radon_transform(AbstractArray{Float64}(J[1]),AbstractArray{Float64}(range(-π,π,128)));

# ╔═╡ fb09010e-ae5e-44eb-a5b6-f20d33afac84
function RKA(Class::AbstractArray,radii::Int64,angles::Int64)
	rangeangles=Array(range(-π,π,angles+1))[1:end-1]
	RJ = []
	for k in 1:length(Class)
		push!(RJ, RadonKA.radon(AbstractArray{Float64}(Class[k]), rangeangles))
		#geometry=RadonParallelCircle(size(Class[k])[1], range(-1,1,100)))
		#push!(RJ, radon_transform(AbstractArray{Float64}(Class[k]),radii ,angles))
	end
	return RJ
end

# ╔═╡ dfdecf7b-b81e-4da9-bdf5-0d6937d26ac0
function RPA(Class::AbstractArray,radii::Int64,angles::Int64)
	rangeangles=Array(range(-π,π,angles+1))[1:end-1]
	RJ = []
	for k in 1:length(Class)
		push!(RJ, radon_transform(AbstractArray{Float64}(Class[k]), rangeangles))
		#geometry=RadonParallelCircle(size(Class[k])[1], range(-1,1,100)))
		#push!(RJ, radon_transform(AbstractArray{Float64}(Class[k]),radii ,angles))
	end
	return RJ
end

# ╔═╡ d3fb5c62-118c-41f5-87a9-5efe5d79e6a8
RKTemp = RKA(J,501,16); RKClass = RKA(TClass,501,16);

# ╔═╡ 0f1d67da-4e22-4c5e-8992-fcd4555e0e2d
RPTemp = RPA(J,501,16); RPClass = RPA(TClass,501,16);

# ╔═╡ 3a733559-65b9-427c-8a72-0039fb2bdb7e
# ╠═╡ disabled = true
#=╠═╡
function signal_to_density(s::Vector{Float64}, ϵ::Float64)
    p = s .+ ϵ
    return p ./= sum(p)
end
  ╠═╡ =#

# ╔═╡ bc9bd9aa-cf6a-4fb3-9d95-9d7766fb8585
function cdtka(s::Vector{Float64}, S::AbstractArray)
    p = signal_to_density(s, 1e-10)
    P = cumsum(p)
    s_grid = collect(LinRange(-1, 1, length(s)))
    q_grid = collect(LinRange(0, 1, size(S)[1] + 2))[2:end - 1]
    q = LinearInterpolation(P, s_grid)
    return q(q_grid)
end

# ╔═╡ 4504ef7e-0cc7-49a5-ad54-7cb7ff3502fc
function RICDT(Class::AbstractArray)
	qQ = []
    for k in 1:length(Class)
		S = Class[k]
		q,a = size(S)
		Q = zeros(q, a)
	    for j in axes(Q, 2)
	        Q[:, j] = cdtka(S[:, j], S)
	    end
		push!(qQ,Q)
	end 
    return qQ
end

# ╔═╡ 367c2ba2-4246-41e2-a6e4-e71a72f3f281
qTempRKACDT = RICDT(RKTemp); qClassRKACDT = RICDT(RKClass);

# ╔═╡ 4d9aa396-2b35-46d5-9e21-405edb5860bc
qTempRPACDT = RICDT(RPTemp); qClassRPACDT = RICDT(RPClass);

# ╔═╡ 121077da-4eee-4eec-88bd-c8b9851ba5f8
mqTempRPACDT = max_normalization.(qTempRPACDT); mqClassRPACDT = max_normalization.(qClassRPACDT);

# ╔═╡ 2fa9b619-dfe8-4ba9-9e96-2c0be84e32c3
mqTempRKACDT = max_normalization.(qTempRKACDT); mqClassRKACDT = max_normalization.(qClassRKACDT);

# ╔═╡ e5f19621-f786-4b27-afc6-3468f2647a1b
begin
	function plot_quantiles(temp_q::AbstractArray, temp_lab::AbstractArray, data_q::AbstractArray, data_lab::AbstractArray)
	    dd = length(data_q[1])
	    plt = plot(size = (500,350))
	    color = [:blue, :red, :green, :orange, :purple, :brown]
	    for i in 1:length(temp_lab), j in 1:length(data_lab)
	        if data_lab[j] == temp_lab[i]
	            plot!(plt, data_q[j], label=false, linecolor = RGBA((abs(0.5-(i-1)/(length(temp_lab)-1)))*(1-(i-1)/(length(temp_lab)-1))*0.99, ((i-1)/(length(temp_lab)-1))*(1-(i-1)/(length(temp_lab)-1))*0.99, (abs(0.5-(i-1)/(length(temp_lab)-1)))*((i-1)/(length(temp_lab)-1))*0.99, 0.5), yticks=true, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
	            # plot!(plt, data_q[j], label=false, linecolor = color[i], yticks=true, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));
	        end
	    end
	    for i in 1:length(temp_lab)
	        k = temp_lab[i]
	        plot!(plt, temp_q[i], legend=false, fontfamily="Computer Modern", label=["Class $k" "k"], linewidth=2 , linecolor = RGBA((abs(0.5-(i-1)/(length(temp_lab)-1)))*(1-(i-1)/(length(temp_lab)-1))*0.99, ((i-1)/(length(temp_lab)-1))*(1-(i-1)/(length(temp_lab)-1))*0.85, (abs(0.5-(i-1)/(length(temp_lab)-1)))*((i-1)/(length(temp_lab)-1))*0.99, 1), yticks=true, xticks = (LinRange(0,dd,4), ["0", "0.25", "0.75", "1"]));    
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
end

# ╔═╡ 2e8bebc7-f987-413d-9dcc-2cb4e790b253
p = plot_quantiles(mqTemp, Label, mqClass, Labels)

# ╔═╡ eb4cb74c-5253-4d54-8b62-49fb5c0b0622
# ╠═╡ disabled = true
#=╠═╡
savefig(p, "academic_quantiles_exact_large.pdf")
  ╠═╡ =#

# ╔═╡ 5fba40e3-cdf2-403e-8685-9641be3ee2ca
accuracy_nearest_neighbour(mqTemp, Label, mqClass, Labels, "inf", ret=1)

# ╔═╡ 28f28ee6-a487-4e39-8ac3-98353cc0bb85
accuracy_nearest_neighbour(mqTemp, Label, mqClass, Labels, "euclidean", ret=1)

# ╔═╡ e133440b-55dd-4ac3-994c-3da15c2f16dd
accuracy_nearest_neighbour(mqTemp, Label, mqClass, Labels, "manhattan", ret=1)

# ╔═╡ dcc7abe5-c528-4a3e-ad59-e77c62ef6f59
mean(abs.(mean(mqClass[1:10]) .- mqTemp[1]))

# ╔═╡ 4a0751b9-b16d-42e8-9b23-8663cddaf0d3
mean(abs.(mean(mqClass[11:20]) .- mqTemp[2]))

# ╔═╡ 2d0178bb-fb98-404a-b93d-dca756aec960
mean(abs.(mean(mqClass[21:30]) .- mqTemp[3]))

# ╔═╡ efa802c3-aec1-4a65-b551-13d0d8b44b2b
pi = plot_quantiles(mqTempRKACDT, Label, mqClassRKACDT, Labels)

# ╔═╡ 43d13deb-a992-4b54-9256-d8368b741306
# ╠═╡ disabled = true
#=╠═╡
savefig(pi, "academic_quantiles_inter_large.pdf")
  ╠═╡ =#

# ╔═╡ 8082dd9c-f557-4149-94ef-da54969b3679
mean(abs.(mean(mqClassRKACDT[1:10]) .- mqTempRKACDT[1]))

# ╔═╡ d628bab5-df7c-499d-a3d0-022cfcd69de2
mean(abs.(mean(mqClassRKACDT[11:20]) .- mqTempRKACDT[2]))

# ╔═╡ 86785da0-0d96-44dd-bf37-175ff97e3902
mean(abs.(mean(mqClassRKACDT[21:30]) .- mqTempRKACDT[3]))

# ╔═╡ 4db78f1b-d418-49ff-ab8c-c595ce53896a
accuracy_nearest_neighbour(mqTempRKACDT, Label, mqClassRKACDT, Labels, "inf", ret=1)

# ╔═╡ 0654303f-8ecf-4e9f-b8c1-47516afc62bf
accuracy_nearest_neighbour(mqTempRKACDT, Label, mqClassRKACDT, Labels, "euclidean", ret=1)

# ╔═╡ 3acabac7-ddc6-425f-8c1a-b641a00127c3
accuracy_nearest_neighbour(mqTempRKACDT, Label, mqClassRKACDT, Labels, "manhattan", ret=1)

# ╔═╡ fcd40e6a-ec39-44f4-8a05-ca4cfdfe42d6
pp = plot_quantiles(mqTempRPACDT, Label, mqClassRPACDT, Labels)

# ╔═╡ 9b2a1681-06ea-4007-9505-77bb82bf1241
# ╠═╡ disabled = true
#=╠═╡
savefig(pp, "academic_quantiles_project_large.pdf")
  ╠═╡ =#

# ╔═╡ d14f2566-6cf0-4998-9d30-3ecc8b0dfa59
accuracy_nearest_neighbour(mqTempRPACDT, Label, mqClassRPACDT, Labels, "inf", ret=1)

# ╔═╡ ff670d21-852a-466c-90dd-748a60df8e53
accuracy_nearest_neighbour(mqTempRPACDT, Label, mqClassRPACDT, Labels, "euclidean", ret=1)

# ╔═╡ f77a3a29-b47e-4d1a-8ce8-771834b61b24
accuracy_nearest_neighbour(mqTempRPACDT, Label, mqClassRPACDT, Labels, "manhattan", ret=1)

# ╔═╡ Cell order:
# ╠═8cbe0300-edff-11ef-2fad-d3b8cca171a9
# ╠═b2a669f2-34c3-48db-9a72-feefbc3b86aa
# ╠═9bc6bbed-558b-471e-bfdb-62d18897a910
# ╠═d272426c-53c1-46f4-8d39-ab2f635ad683
# ╠═61d09b61-1038-42dc-84b1-cbcd37b10a6e
# ╠═aa47398a-b248-48fd-943e-b349b6e8f6a8
# ╠═29700517-53ff-493a-acec-e4eac297eee1
# ╠═0f0ba691-f135-4647-8039-c6ba6ff09eea
# ╠═5a04ab89-1ab9-480c-97a1-b47bdf0baedb
# ╠═6ae8e856-1eed-4fe0-be60-49d346ce7ecb
# ╠═e1239f0f-246c-44a8-95f8-efca6a01d0c8
# ╠═21d5fd97-3d8b-4bc5-8a6a-0f77ca9ef8d5
# ╠═6924e56a-973d-4398-83a6-be235fa61252
# ╠═2a3b392c-e818-44d2-939a-5f90354141ca
# ╠═50ef63a9-f9f9-4edd-a7aa-d24d24f56043
# ╠═9026e79e-ce2c-4b5b-987b-c4ba7e340d09
# ╠═4f5b3b5e-e4b5-497c-8a59-c40e05249274
# ╠═76b23278-16ea-411a-8771-8606d8e14b96
# ╠═affee424-d71b-4b18-81f6-f6a7c6ab95ac
# ╠═6b12a034-c756-47cd-aab2-7be920fd0fc1
# ╠═73584882-fb8b-44bc-9769-a8b6f715ae25
# ╠═4beeac67-a9bb-45c5-95b1-d0b556af9561
# ╠═328b0da3-2319-4bf4-8536-64ca2b38534e
# ╠═f329081e-fa20-495d-ad78-db03ed6d7859
# ╠═d6fff828-24ca-40e6-b16f-a09c70c7650e
# ╠═fb09010e-ae5e-44eb-a5b6-f20d33afac84
# ╠═dfdecf7b-b81e-4da9-bdf5-0d6937d26ac0
# ╠═d3fb5c62-118c-41f5-87a9-5efe5d79e6a8
# ╠═0f1d67da-4e22-4c5e-8992-fcd4555e0e2d
# ╠═3a733559-65b9-427c-8a72-0039fb2bdb7e
# ╠═bc9bd9aa-cf6a-4fb3-9d95-9d7766fb8585
# ╠═4504ef7e-0cc7-49a5-ad54-7cb7ff3502fc
# ╠═367c2ba2-4246-41e2-a6e4-e71a72f3f281
# ╠═4d9aa396-2b35-46d5-9e21-405edb5860bc
# ╠═121077da-4eee-4eec-88bd-c8b9851ba5f8
# ╠═2fa9b619-dfe8-4ba9-9e96-2c0be84e32c3
# ╠═e5f19621-f786-4b27-afc6-3468f2647a1b
# ╠═2e8bebc7-f987-413d-9dcc-2cb4e790b253
# ╠═eb4cb74c-5253-4d54-8b62-49fb5c0b0622
# ╠═5fba40e3-cdf2-403e-8685-9641be3ee2ca
# ╠═28f28ee6-a487-4e39-8ac3-98353cc0bb85
# ╠═e133440b-55dd-4ac3-994c-3da15c2f16dd
# ╠═dcc7abe5-c528-4a3e-ad59-e77c62ef6f59
# ╠═4a0751b9-b16d-42e8-9b23-8663cddaf0d3
# ╠═2d0178bb-fb98-404a-b93d-dca756aec960
# ╠═efa802c3-aec1-4a65-b551-13d0d8b44b2b
# ╠═43d13deb-a992-4b54-9256-d8368b741306
# ╠═8082dd9c-f557-4149-94ef-da54969b3679
# ╠═d628bab5-df7c-499d-a3d0-022cfcd69de2
# ╠═86785da0-0d96-44dd-bf37-175ff97e3902
# ╠═4db78f1b-d418-49ff-ab8c-c595ce53896a
# ╠═0654303f-8ecf-4e9f-b8c1-47516afc62bf
# ╠═3acabac7-ddc6-425f-8c1a-b641a00127c3
# ╠═fcd40e6a-ec39-44f4-8a05-ca4cfdfe42d6
# ╠═9b2a1681-06ea-4007-9505-77bb82bf1241
# ╠═d14f2566-6cf0-4998-9d30-3ecc8b0dfa59
# ╠═ff670d21-852a-466c-90dd-748a60df8e53
# ╠═f77a3a29-b47e-4d1a-8ce8-771834b61b24
