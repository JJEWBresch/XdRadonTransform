# setprecision(Double64, 64)

struct RadonTransform
    radii::Int64
    angles::Int64
    width::Float64 #Double64
    function RadonTransform(radii, angles, width)
        if radii <= 0
            error("non-positive number of radii")
        elseif angles <= 0
            error("non-positive number of angles")
        elseif width < 0.0
            error("negative width")
        end 
        return new(radii, angles, width)
    end
end

RadonTransform(radii, angles; width=0.0) = RadonTransform(radii, angles, width)

struct Phantom
    data::Matrix{Float64} #Matrix{Double64}
    pixel_size::Float64
    dim_x::Int64
    dim_y::Int64
end

function Phantom(image::AbstractMatrix)
    pixel_size = sqrt(2) / max(size(image)...)
    return Phantom(Float64.(image), Float64(pixel_size), size(image, 2), size(image, 1))  # Phantom(Double64.(image), pixel_size, size(image, 2), size(image, 1))    
end

struct Ray
    t::Float64 #Double64
    θ::Float64 #Double64
    w::Float64 #Double64
end

function (R::RadonTransform)(image::AbstractMatrix)
    P = Phantom(image)
    S = zeros(Float64, R.radii, R.angles)
    for j in axes(S, 1), k in axes(S, 2)
        r = determine_corresponding_ray(j, k, R)
        S[j, k] = integrate_along_ray(P, r)
    end
    return S
end

function determine_corresponding_ray(j::Int64, k::Int64, R::RadonTransform)
    t = (2 + R.width) * (j - 1) / (R.radii - 1) - 1 - R.width / 2
    θ = 2π * (k - 1) / R.angles
    return Ray(t, θ, R.width)
end

function integrate_along_ray(P::Phantom, r::Ray)
    if π / 4 <= r.θ < 3 * π / 4 || 5 * π / 4 <= r.θ < 7 * π / 4
        return integrate_horizontal_ray(P, r)
    else
        return integrate_vertical_ray(P, r)
    end
end 

function integrate_vertical_ray(P::Phantom, r::Ray)
    S = 0.0
    for j in axes(P.data, 1)
        for k in horizontal_pixels(j, P, r)
            S += integrate_pixel_along_ray(j, k, P, r)
        end
    end
    return S
end

function horizontal_pixels(j::Int64, P::Phantom, r::Ray)
    (cos_θ, sin_θ) = (cos(r.θ), sin(r.θ))
    y = index_to_y_coordinate(j, P)
    τ = (r.t * sin_θ - y) / cos_θ
    x = r.t * cos_θ + τ * sin_θ
    x_width = abs(r.w / cos_θ / 2)
    first_index = Int64(floor(x_coordinate_to_index(x - x_width, P)))
    last_index = Int64(ceil(x_coordinate_to_index(x + x_width, P)))
    return max(first_index, 1):min(last_index, P.dim_x)
end

function integrate_horizontal_ray(P::Phantom, r::Ray)
    S = 0.0
    for k in axes(P.data, 2)
        for j in vertical_pixels(k, P, r)
            S += integrate_pixel_along_ray(j, k, P, r)
        end
    end
    return S
end

function vertical_pixels(k::Int64, P::Phantom, r::Ray)
    (cos_θ, sin_θ) = (cos(r.θ), sin(r.θ))
    x = index_to_x_coordinate(k, P)
    τ = (x - r.t * cos_θ) / sin_θ
    y = r.t * sin_θ - τ * cos_θ
    y_width = abs(r.w / sin_θ / 2)
    first_index = Int64(floor(y_coordinate_to_index(y + y_width, P)))
    last_index = Int64(ceil(y_coordinate_to_index(y - y_width, P)))
    return max(first_index, 1):min(last_index, P.dim_y)
end

function index_to_x_coordinate(k::Int64, P::Phantom)
    return (k - (P.dim_x + 1) / 2) * P.pixel_size
end

function x_coordinate_to_index(x::Float64, P::Phantom)
    return x / P.pixel_size + (P.dim_x + 1) / 2
end

function index_to_y_coordinate(j::Int64, P::Phantom)
    return ((P.dim_y + 1) / 2 - j) * P.pixel_size
end

function y_coordinate_to_index(y::Float64, P::Phantom)
    return (P.dim_y + 1) / 2 - y / P.pixel_size
end

function integrate_pixel_along_ray(j::Int64, k::Int64, P::Phantom, r::Ray)
    unit_r = select_unit_ray(j, k, P, r)
    weight = integrate_unit_pixel_along_ray(unit_r)
    scale = (r.w == 0.0) ? P.pixel_size : P.pixel_size^2 / r.w
    return weight * scale * P.data[j, k]
end

function select_unit_ray(j::Int64, k::Int64, P::Phantom, r::Ray)
    t_pixel = index_to_t_coordinate(j, k, P, r)
    unit_t = (r.t - t_pixel) / P.pixel_size
    unit_θ = mod(r.θ, π / 2)
    unit_θ = (0 <= unit_θ < π/4) ? unit_θ : π / 2 - unit_θ
    unit_w = r.w / P.pixel_size
    return Ray(unit_t, unit_θ, unit_w)
end

function index_to_t_coordinate(j::Int64, k::Int64, P::Phantom, r::Ray)
    x = index_to_x_coordinate(k, P)
    y = index_to_y_coordinate(j, P)
    return x * cos(r.θ) + y * sin(r.θ)
end

function integrate_unit_pixel_along_ray(r::Ray)
    if r.w == 0.0
        return line_integral_unit_pixel(r.t, r.θ)
    else
        return area_integral_unit_pixel(r.t, r.θ, r.w)
    end
end

function line_integral_unit_pixel(t::Float64, θ::Float64) #line_integral_unit_pixel(t::Double64, θ::Double64)
    line = 1 / cos(θ)
    t1 = -(cos(θ) + sin(θ)) / 2
    t2 = (sin(θ) - cos(θ)) / 2
    t3 = -t2
    t4 = -t1
    if t <= t1
        return 0.0
    elseif t <= t2
        return line * (t - t1) / (t2 - t1)
    elseif t <= t3
        return line
    elseif t <= t4
        return line * (t4 - t) / (t4 - t3)
    else
        return 0.0
    end
end

function area_integral_unit_pixel(t::Float64, θ::Float64, w::Float64) #area_integral_unit_pixel(t::Double64, θ::Double64, w::Double64)
    return (anti_derivative_area_integral(t + w / 2, θ) - anti_derivative_area_integral(t - w / 2, θ))
end

function anti_derivative_area_integral(t::Float64, θ::Float64) #anti_derivative_area_integral(t::Double64, θ::Double64)
    area_triangle = tan(θ) / 2
    area_parallelogram = 1 - tan(θ)
    t1 = -(cos(θ) + sin(θ)) / 2
    t2 = (sin(θ) - cos(θ)) / 2
    t3 = -t2
    t4 = -t1
    if t <= t1
        return 0.0
    elseif t <= t2
        return area_triangle * ((t - t1) / (t2 - t1))^2
    elseif t <= t3
        return area_triangle + area_parallelogram * (t - t2) / (t3 - t2)
    elseif t <= t4
        return 1 - area_triangle * ((t - t4) / (t4 - t3))^2
    else
        return 1.0
    end
end

####

struct Backprojection
    dim_x::Int64
    dim_y::Int64
    inter::String
    filter::String
    Δs::Float64
    function Backprojection(dim_x, dim_y, inter, filter, Δs)
        if dim_x <= 0
            error("non-positive dimension of x")
        elseif dim_y <= 0
            error("non-positive dimension of y")
        elseif inter ∉ ["line", "linear"]
            error("non-int inter")
        elseif filter ∉ ["none", "ramp"]
            error("non-int filter")
        elseif Δs <= 0
            error("non-positive Δs")
        end 
        return new(dim_x, dim_y, inter, filter, Δs)
    end
end

struct Sinogram
    data::Matrix{Float64}
    angles::AbstractArray
    dim_t::Int64
end

function Sinogram(sinogram::AbstractMatrix)
    dim_t, Nθ = size(sinogram)
    angles = range(0,2*π,Nθ+1)[1:end-1]
    return Sinogram(Float64.(sinogram), angles, dim_t)    
end

function ndgrid(x::Vector, y::Vector)
    X = repeat(x, 1, length(y))
    Y = repeat(y', length(x), 1)
    return X, Y
end

function (B::Backprojection)(sinogram::AbstractMatrix)
    S = Sinogram(sinogram)
    if B.filter=="ramp"
        S = Sinogram(filter_back(S.data, B.Δs))
    end
    img = zeros(B.dim_x, B.dim_y)
    Ns = S.dim_t
    θ = S.angles

    # Coordinate grid centered at (0,0) with axes span in [-1/2,1/2]^2
    x = collect(range(-1, stop=1, length=B.dim_x+2))[2:end-1] # collect(-(B.dim_x-1)/2:(B.dim_x-1)/2)/B.dim_x
    y = collect(range(-1, stop=1, length=B.dim_y+2))[2:end-1] # collect(-(B.dim_y-1)/2:(B.dim_y-1)/2)/B.dim_y
    X, Y = ndgrid(x, y)
    
    for i in 1:length(θ)
        ϑ = θ[i]
        T_vals = X .* cos(ϑ) + Y .* sin(ϑ)
        # map [-sqrt(2), sqrt(2)] to [-1,1] <- [1,Ns] = collect(1:Ns) << (. - (Ns+1)/2)*2/(Ns+1)
        I_vals = T_vals .* (Ns+1)/2 .+ (Ns+1)/2
        I_vals_up = ceil.(I_vals)
        I_vals_down = floor.(I_vals)
        weights = zeros(size(T_vals,1),size(T_vals,2),2)
        dsino = zeros(size(T_vals,1),size(T_vals,2),2)
        for k in axes(T_vals, 1), l in axes(T_vals, 2)
            if B.inter == "line"
                weights[k,l,1] = line_integral_unit_pixel(I_vals_up[k,l]-I_vals[k,l], ϑ)
                weights[k,l,2] = line_integral_unit_pixel(I_vals[k,l]-I_vals_down[k,l], ϑ)
            elseif B.inter == "linear"
                weights[k,l,1] = I_vals_up[k,l]-I_vals[k,l]
                weights[k,l,2] = I_vals[k,l]-I_vals_down[k,l]
            end
            dsino[k,l,1] = (1 <= I_vals_up[k,l] <= Ns) ? S.data[Int64(I_vals_up[k,l]),i] : 0.0
            dsino[k,l,2] = (1 <= I_vals_down[k,l] <= Ns) ? S.data[Int64(I_vals_down[k,l]),i] : 0.0
        end
        nweights = abs.(weights[:,:,1]) .+ abs.(weights[:,:,2])
        nweights[iszero.(nweights)] .= 1.0
        img .+= abs.(weights[:,:,1])./nweights .* dsino[:,:,1] .+ abs.(weights[:,:,2])./nweights .* dsino[:,:,2]
    end
    #=
    for i in 1:length(θ)
        ϑ = θ[i]
        s_vals = X .* cos(ϑ) + Y .* sin(ϑ)

        # Convert s to detector index: scale [-max_s, max_s] to [1, Ns]
        s_max = maximum(abs.(s_vals))
        s_range = range(-s_max, stop=s_max, length=Ns)
        interp = LinearInterpolation(s_range, sinogram[:, i], extrapolation_bc=0.0)

        img .+= interp.(s_vals)
    end
    =#
    return img
end

using FFTW

function fftfreq(n::Int, Δs::Float64)
    return [i <= n ÷ 2 ? i : i - n for i in 0:n-1] ./ (n * Δs)
end

function filter_back(sinogram::AbstractMatrix, Δs::Float64)
    S = Sinogram(sinogram)
    Ns = S.dim_t
    Nθ = length(S.angles)
    filtered = zeros(Ns, Nθ)

    # Frequency axis
    # freqs = fftshift(-(Ns+1)÷2:(Ns+1)÷2 - 1)[1:end-1]
    # freqs = collect(range(-1, stop=1, length=Ns))
    # ramp = abs.(freqs)  # Ram-Lak filter

    ω = fftfreq(Ns, Δs)
    # ramp = (abs.(ω)).^(1/4)  # Ramp filter in frequency domain
    ramp = abs.(ω)  # Ramp filter in frequency domain
    for i in 1:Nθ
        proj = S.data[:, i]
        proj_fft = fft(proj)
        filtered_proj_fft = proj_fft .* ramp
        filtered[:, i] = real.(ifft(filtered_proj_fft))
    end
    return filtered
end
