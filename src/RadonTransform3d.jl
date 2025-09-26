using Base.Threads   # makes @threads available

struct RadonTransform3d
    radii::Int64
    angles_theta::Int64
    angles_phi::Int64
    width::Float64
    function RadonTransform3d(radii, angles_theta, angles_phi, width)
        if radii <= 0
            error("non-positive number of radii")
        elseif angles_theta <= 0
            error("non-positive number of angles")
        elseif angles_phi <= 0
            error("non-positive number of angles")
        end 
    width::Float64
        return new(radii, angles_theta, angles_phi, width)
    end
end

RadonTransform3d(radii, angles_theta, angles_phi; width=0.0) = RadonTransform3d(radii, angles_theta, angles_phi, width)

struct Phantom3d
    data::Array{Float64}
    pixel_size::Float64
    dim_x::Int64
    dim_y::Int64
    dim_z::Int64
end

function Phantom3d(image::AbstractArray)
    pixel_size = 1 / sqrt(3) / max(size(image)...)
    return Phantom3d(Float64.(image), pixel_size, size(image, 1), size(image, 2), size(image, 3))    
end

struct Plan
    t::Float64
    θ_1::Float64
    θ_2::Float64
end

struct Plan_t
    t::Float64
end

struct Plan_θ
    θ_1::Float64
    θ_2::Float64
end



function (R::RadonTransform3d)(image::AbstractArray)
    P = Phantom3d(image)
    S = zeros(R.radii, R.angles_theta, R.angles_phi)
    PI = DomRadon(P)
    @threads for k in axes(S, 2)
        for l in axes(S, 3)
            p_θ = determine_corresponding_angle(k, l, R)
            theta = Theta(p_θ)
            TI = map(a -> sum(a), map(a -> a .* theta, PI))
            for j in axes(S,1)
                t = determine_corresponding_ray(j, R)
                t_down, t_up = t.t - 3*sqrt(3)*P.pixel_size, t.t + 3*sqrt(3)*P.pixel_size
                T = findall(x -> t_down <= x <= t_up, TI)
                if size(T)[1] > 0
                    S[j, k, l] = integrate_along_plan(T, TI, P, p_θ, t.t)
                end
            end
        end
    end
    return S
end

function DomRadon(P::Phantom3d)
    # s = 1 / (1/2/P.pixel_size + 1)
    s = P.pixel_size
    PI = [[s*(i - P.dim_x/2),s*(j - P.dim_y/2),s*(k - P.dim_z/2)] for i in 1:P.dim_x, j in 1:P.dim_y, k in 1:P.dim_z]
end

function Theta(p::Plan_θ)
    theta = zeros(3)
    theta[1] = cos(p.θ_1)*sin(p.θ_2)
    theta[2] = sin(p.θ_1)*sin(p.θ_2)
    theta[3] = cos(p.θ_2)
    return theta
end

function IRule(P::Phantom3d, p::Plan_θ)
    I = zeros(P.dim_x, P.dim_y, P.dim_z)
    theta = Theta(p)
    for i in axes(I,1), j in axes(I,2), k in axes(I,3)
        I[i,j,k] = P.pixel_size*(i*theta[1] + j*theta[2] + k*theta[3]) - 1/2*sum(theta)
    end
    return I
end

function determine_corresponding_ray(j::Int64, R::RadonTransform3d)
    t = 2 * (j - 1) / (R.radii - 1) - 1
    return Plan_t(t)
end

function determine_corresponding_angle(k::Int64, l::Int64, R::RadonTransform3d)
    #θ_1 = 2π * (k - 1) / R.angles_theta
    #θ_2 = π * l / (R.angles_phi + 1) - π/2
    R_l = R.angles_theta*R.angles_phi # k*l
    θ_2, θ_1 = fibonacci_sphere_i(k-1+(l-1)*R.angles_theta, R_l)
    θ_2 = θ_2 - π/2
    return Plan_θ(θ_1, θ_2)
end

function determine_corresponding_plan(j::Int64, k::Int64, l::Int64, R::RadonTransform3d)
    t = 2 * (j - 1) / (R.radii - 1) - 1
    θ_1 = 2π * (k - 1) / R.angles_theta
    θ_2 = π * l / (R.angles_phi + 1)
    return Plan(t, θ_1, θ_2)
end

function integrate_along_plan(T::AbstractArray, TI::AbstractArray, P::Phantom3d, p_θ::Plan_θ, t::Float64)
    S = 0.0
    for l in 1:size(T)[1]
        t1 = getindex(T[l],1)
        t2 = getindex(T[l],2)
        t3 = getindex(T[l],3)
        theta = Theta(p_θ)
        weight = plan_integral_unit_pixel_sum(t - TI[t1, t2, t3], theta[1], theta[2], theta[3])
        S += P.data[t1, t2, t3]*weight*P.pixel_size^2
    end
    return S
end 

function plan_integral_unit_pixel_sum(t::Float64, k_1::Float64, k_2::Float64, k_3::Float64)
    #if k_1*k_2*k_3 == 0.0
    #    if max(k_1*k_2, k_1*k_3, k_2*k_3) == 0.0
    #        plan_integral_unit_pixel_sum_one(t::Float64)
    #    else 
    #        if k_1*k_2 == 0.0
    #            if k_1*k_3 == 0.0
    #                plan_integral_unit_pixel_sum_two(t::Float64, k_2::Float64, k_3::Float64)
    #            else
    #                plan_integral_unit_pixel_sum_two(t::Float64, k_1::Float64, k_3::Float64)
    #            end
    #        else
    #            plan_integral_unit_pixel_sum_two(t::Float64, k_1::Float64, k_2::Float64)
    #        end
    #    end
    #else
    #    plan_integral_unit_pixel_sum_three(t::Float64, k_1::Float64, k_2::Float64, k_3::Float64)
    #end
    if k_1*k_2*k_3 != 0
    plan_integral_unit_pixel_sum_three(t::Float64, k_1::Float64, k_2::Float64, k_3::Float64)
    elseif k_1*k_2 != 0
        plan_integral_unit_pixel_sum_two(t::Float64, k_1::Float64, k_2::Float64)
    elseif k_1*k_3 != 0
        plan_integral_unit_pixel_sum_two(t::Float64, k_1::Float64, k_3::Float64)
    elseif k_2*k_3 != 0
        plan_integral_unit_pixel_sum_two(t::Float64, k_2::Float64, k_3::Float64)
    else
        plan_integral_unit_pixel_sum_one(t::Float64)
    end
end

function K_2()
    K = zeros(4,2)
    k = zeros(4)
    for i in 1:2
        for j in 1:2
            K[i + 2*(j-1), 1] = (-1)^i
            K[i + 2*(j-1), 2] = (-1)^j
            k[i + 2*(j-1)] = (-1)^i*(-1)^j
        end
    end
    # k = dropdims(prod(K, dims=2), dims=2)
    return K, k
end

function K_3()
    K = zeros(8,3)
    k = zeros(8)
    for i in 1:2
        for j in 1:2
            for l in 1:2
                K[i + 2*(j-1) + 4*(l-1), 1] = (-1)^i
                K[i + 2*(j-1) + 4*(l-1), 2] = (-1)^j
                K[i + 2*(j-1) + 4*(l-1), 3] = (-1)^l
                k[i + 2*(j-1) + 4*(l-1)] = (-1)^i*(-1)^j*(-1)^l
            end
        end
    end
    # k = dropdims(prod(K, dims=2), dims=2)
    return K, k
end

function plan_integral_unit_pixel_sum_three(t::Float64, k_1::Float64, k_2::Float64, k_3::Float64)
    K, k = K_3()
    # print(1/k_1/k_2/k_3/2 * sum(k .* max.(t .+ K * vec([k_1, k_2, k_3]),0).^2))
    return 1/k_1/k_2/k_3/2 * sum(k .* max.(t .+ 1/2 * K * vec([k_1, k_2, k_3]),0).^2)
end

function plan_integral_unit_pixel_sum_two(t::Float64, k_1::Float64, k_2::Float64)
    K, k = K_2()
    # print(1/k_1/k_2 * sum(k .* max.(t .+ K * vec([k_1, k_2]), 0)))
    return 1/k_1/k_2 * sum(k .* max.(t .+ 1/2 * K * vec([k_1, k_2]), 0))
end

function plan_integral_unit_pixel_sum_one(t::Float64)
    return 1
end

function fibonacci_sphere_i(i::Int64,n::Int64)
    φ = π * (3 - sqrt(5))  # golden angle
	z = 1 - 2*(i+0.5)/n
    x = sqrt(1 - z^2) * cos(φ*i)
	y = sqrt(1 - z^2) * sin(φ*i)
	θ₁ = acos(z)
	θ₂ = atan(y,x)
	return θ₁, θ₂, x, y, z
end

function fibonacci_sphere(n::Int64)
    φ = π * (3 - sqrt(5))  # golden angle
	θ₁ = []
	θ₂ = []
	x = []
	y = []
	z = []
	for i in 0:n-1
		push!(z, 1 - 2*(i+0.5)/n)
    	push!(x, sqrt(1 - z[i+1]^2) * cos(φ*i))
		push!(y, sqrt(1 - z[i+1]^2) * sin(φ*i))
		push!(θ₁, acos(z[i+1]))
		push!(θ₂, atan(y[i+1],x[i+1]))
	end
	return θ₁, θ₂, x, y, z
end
