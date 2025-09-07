module DataTransformations

using Random, Images, Luxor #, DoubleFloats, Quadmath
using Augmentor
using ImageTransformations, Interpolations, OffsetArrays, Rotations, CoordinateTransformations

# setprecision(Float64, 32)

export RandomAffineTransformation, RandomAffineTransformation3d
export MikadoNoise
export SaltNoise
export BarNoise
export ElasticNoise

struct RandomAffineTransformation
    scaling_x::Tuple{Float64, Float64}
    scaling_y::Tuple{Float64, Float64}
    rotating::Tuple{Float64, Float64}
    shearing_x::Tuple{Float64, Float64}
    shearing_y::Tuple{Float64, Float64}
    translating_x::Tuple{Int64, Int64}
    translating_y::Tuple{Int64, Int64}
    function RandomAffineTransformation(scx, scy, rot, shx, shy, tax, tay)
        if !(0 <= scx[1] <= scx[2]) 
            error("inconsistent x scaling")
        elseif !(0 <= scy[1] <= scy[2]) 
            error("inconsistent y scaling")
        elseif !(rot[1] <= rot[2]) 
            error("inconsistent rotating")
        elseif !(shx[1] <= shx[2]) 
            error("inconsistent x shearing")
        elseif !(shy[1] <= shy[2]) 
            error("inconsistent y shearing")
        elseif !(tax[1] <= tax[2]) 
            error("inconsistent x translating")
        elseif !(tay[1] <= tay[2]) 
            error("inconsistent y translating")
        end
        return new(scx, scy, rot, shx, shy, tax, tay)
    end
end

function RandomAffineTransformation(;
    scale_x::Tuple{Float64, Float64}=(1.0, 1.0),
    scale_y::Tuple{Float64, Float64}=(1.0, 1.0),
    rotate::Tuple{Float64, Float64}=(0.0, 0.0),
    shear_x::Tuple{Float64, Float64}=(0.0, 0.0),
    shear_y::Tuple{Float64, Float64}=(0.0, 0.0),
    shift_x::Tuple{Int64, Int64}=(0, 0),
    shift_y::Tuple{Int64, Int64}=(0, 0)
)
    return RandomAffineTransformation(scale_x, scale_y, rotate, shear_x, shear_y, shift_x, shift_y)
end

function (A::RandomAffineTransformation)(image::AbstractMatrix)
    I = Gray{Float64}.(image) #Gray{Float64}.(image)
    I = scaling(I, randu(A.scaling_x), randu(A.scaling_y))
    I = rotate(I, randu(A.rotating))
    I = shear(I, randu(A.shearing_x), randu(A.shearing_y))
    I = translate(I, randi(A.translating_x), randi(A.translating_y))
    return I
end

function randu(bounds::Tuple{Float64, Float64})
    (a, b) = bounds
    return (b - a) * rand() + a
end

function randi(bounds::Tuple{Int64, Int64})
    (a, b) = bounds
    return rand(a:b)
end

function scaling(I::Matrix{Gray{Float64}}, sx::Float64, sy::Float64)
    return augment(I, Zoom(sy, sx))
end

function rotate(I::Matrix{Gray{Float64}}, α::Float64)
    s = size(I)
    return augment(I, Rotate(α) |> CropSize(s))
end

function shear(I::Matrix{Gray{Float64}}, α::Float64, β::Float64)
    s = size(I)
    return augment(I, ShearX(α) |> ShearY(β) |> CropSize(s))
end

function translate(I::Matrix{Gray{Float64}}, x::Int64, y::Int64)
    return circshift(I, (-y, x))
end

abstract type AbstractNoise end

struct MikadoNoise <: AbstractNoise
    sticks::Tuple{Int64, Int64}
    length::Tuple{Float64, Float64}
    width::Tuple{Float64, Float64}
    function MikadoNoise(s, l, w)
        if !(0 <= s[1] <= s[2]) 
            error("inconsistent stick number")
        elseif !(0 <= l[1] <= l[2]) 
            error("inconsistent length")
        elseif !(0 <= w[1] <= w[2]) 
            error("inconsistent width")
        end
        return new(s, l, w)
    end
end

function MikadoNoise(;
    sticks::Tuple{Int64, Int64}=(10, 15),
    length::Tuple{Float64, Float64}=(0.125, 0.25),
    width::Tuple{Float64, Float64}=(2.0, 2.0)
)
    return MikadoNoise(sticks, length, width)
end

function (N::MikadoNoise)(image::AbstractMatrix)
    I = Gray{Float64}.(image)
    E = render_noise(N, size(I))
    return max.(I, E)
end

struct SaltNoise <: AbstractNoise
    dots::Tuple{Int64, Int64}
    width::Tuple{Float64, Float64}
    pos::Tuple{Float64, Float64}
    function SaltNoise(d, w, p)
        if !(0 <= d[1] <= d[2]) 
            error("inconsistent dot number")
        elseif !(0 <= w[1] <= w[2]) 
            error("inconsistent width")
        elseif !((0 <= p[1] <= 1) & (0 <= p[2] <= 1)) 
            error("inconsistent position")
        end
        return new(d, w, p)
    end
end

function SaltNoise(;
    dots::Tuple{Int64, Int64}=(10, 15),
    width::Tuple{Float64, Float64}=(1e-5, 1e-5),
    pos::Tuple{Float64, Float64}=(0.0, 0.0)
)
    return SaltNoise(dots, width, pos)
end

function (N::SaltNoise)(image::AbstractMatrix)
    I = Gray{Float64}.(image)
    E = render_noise(N, size(I))
    return max.(I, E)
end

struct BarNoise <: AbstractNoise
    bars::Tuple{Int64, Int64}
    width::Tuple{Float64, Float64}
    function BarNoise(b, w)
        if !(0 <= b[1] <= b[2]) 
            error("inconsistent bar number")
        elseif !(0 <= w[1] <= w[2]) 
            error("inconsistent width")
        end
        return new(b, w)
    end
end

function BarNoise(;
    bars::Tuple{Int64, Int64}=(2, 5),
    width::Tuple{Float64, Float64}=(2.0, 2.0)
)
    return BarNoise(bars, width)
end

function (N::BarNoise)(image::AbstractMatrix)
    I = Gray{Float64}.(image)
    E = 1.0 .- render_noise(N, size(I))
    return min.(I, E)
end

function render_noise(noise::AbstractNoise, size::Tuple{Integer,Integer})
    initiate_luxor_drawing(size)
    luxor_draw(noise)
    return extract_luxor_drawing()
end

function initiate_luxor_drawing(size::Tuple{Integer,Integer})
    (y, x) = size
    Drawing(x, y, :png)
    origin()
    background("black")
    sethue("white")
    Luxor.scale(x / 2, y / 2)
    return nothing
end

function luxor_draw(noise::MikadoNoise)
    for _ in 1:randi(noise.sticks) 
        luxor_draw_stick(noise)
    end
    return nothing
end

function luxor_draw_stick(noise::MikadoNoise)
    direction = rand([:vertical, :horizontal, :diagonal_up, :diagonal_down])
    length = randu(noise.length)
    setline(randu(noise.width))
    stick_path(direction, length)
    strokepath()    
end

function stick_path(direction::Symbol, length::Float64)
    (x, y) = (2 - length) .* (rand(2) .- 0.5) 
    if direction==:vertical
        move(Point(x, y + length / 2))
        line(Point(x, y - length / 2))
    elseif direction==:horizontal
        move(Point(x + length / 2, y))
        line(Point(x - length / 2, y))
    elseif direction==:diagonal_up
        move(Point(x - length / 2sqrt(2), y - length / 2sqrt(2)))
        line(Point(x + length / 2sqrt(2), y + length / 2sqrt(2)))
    elseif direction==:diagonal_down
        move(Point(x - length / 2sqrt(2), y + length / 2sqrt(2)))
        line(Point(x + length / 2sqrt(2), y - length / 2sqrt(2)))
    end   
end

function luxor_draw(noise::SaltNoise)
    for _ in 1:randi(noise.dots) 
        luxor_draw_dot(noise)
    end
    return nothing
end

function luxor_draw_dot(noise::SaltNoise)
    width = randu(noise.width)
    if max(noise.pos[1],noise.pos[2])==0
        pos=rand(2)
    else
        pos=[noise.pos[1],noise.pos[2]]
    end
    #(x, y) = (2 - width) .* (rand(2) .- 0.5)
    (x, y) = (2 - width) .* (pos .- 0.5) 
    circle(Point(x, y), width / 2; action=:fill)
end

function luxor_draw(noise::BarNoise)
    for _ in 1:randi(noise.bars) 
        luxor_draw_bar(noise)
    end
    return nothing
end

function luxor_draw_bar(noise::BarNoise)
    width = randu(noise.width)
    x = (2 - width) * (rand() - 0.5)
    setline(width)
    move(Point(x, 1.0))
    line(Point(x, -1.0))
    strokepath()
end

function extract_luxor_drawing()
    image = image_as_matrix()
    finish()
    return Gray{Float64}.(image)
end

struct ElasticNoise
    amplitude_x::Tuple{Float64, Float64}
    amplitude_y::Tuple{Float64, Float64}
    frequency_x::Tuple{Float64, Float64}
    frequency_y::Tuple{Float64, Float64}
    phase_x::Tuple{Float64, Float64}
    phase_y::Tuple{Float64, Float64}
    function ElasticNoise(ax, ay, fx, fy, px, py)
        if !(0 <= ax[1] <= ax[2]) 
            error("inconsistent x amplitude")
        elseif !(0 <= ay[1] <= ay[2]) 
            error("inconsistent y amplitude")
        elseif !(0 <= fx[1] <= fx[2]) 
            error("inconsistent x frequency")
        elseif !(0 <= fy[1] <= fy[2]) 
            error("inconsistent y frequency")
        elseif !(px[1] <= px[2]) 
            error("inconsistent x phase")
        elseif !(py[1] <= py[2]) 
            error("inconsistent y phase")
        end
        return new(ax, ay, fx, fy, px, py)
    end
end

function ElasticNoise(;
    amplitude_x::Tuple{Float64, Float64}=(0.0, 0.0),
    amplitude_y::Tuple{Float64, Float64}=(0.0, 0.0),
    frequency_x::Tuple{Float64, Float64}=(1.0, 1.0),
    frequency_y::Tuple{Float64, Float64}=(1.0, 1.0),
    phase_x::Tuple{Float64, Float64}=(0.0, 0.0),
    phase_y::Tuple{Float64, Float64}=(0.0, 0.0),
)
    return ElasticNoise(amplitude_x, amplitude_y, frequency_x, frequency_y, phase_x, phase_y)
end

function (N::ElasticNoise)(image::AbstractMatrix)
    I = Gray{Float64}.(image)
    dim_y, dim_x = size(I)
    J = zero(I)
	ax = randu(N.amplitude_x)
    ay = randu(N.amplitude_y)
	fx = randu(N.frequency_x)
	fy = randu(N.frequency_y)
    px = randu(N.phase_x)
    py = randu(N.phase_y)
    for j in axes(I, 1), k in axes(I, 2)
        x = k + ax * sin(2π * fx * j / dim_y + px)
        y = j + ay * cos(2π * fy * k / dim_x + py)
        J[j, k] = periodic_bilinear_interpolation(x, y, I) 
    end
    return J
end

function periodic_bilinear_interpolation(x::Float64, y::Float64, I::Matrix{Gray{Float64}})
    (dim_y, dim_x) = size(I)
    x₁ = floor(Int64, x)
    x₂ = x₁ + 1
    y₁ = floor(Int64, y)
    y₂ = y₁ + 1    
    z = (x₂ - x) * (y₂ - y) * I[periodic_index(y₁, dim_y), periodic_index(x₁, dim_x)]
    z += (x₂ - x) * (y - y₁) * I[periodic_index(y₂, dim_y), periodic_index(x₁, dim_x)]
    z += (x - x₁) * (y₂ - y) * I[periodic_index(y₁, dim_y), periodic_index(x₂, dim_x)]
    z += (x - x₁) * (y - y₁) * I[periodic_index(y₂, dim_y), periodic_index(x₂, dim_x)]
    return z
end

function periodic_index(j::Int64, dim::Int64)
    return mod(j - 1, dim) + 1
end

struct RandomAffineTransformation3d
    scaling_x::Tuple{Float64, Float64}
    scaling_y::Tuple{Float64, Float64}
    scaling_z::Tuple{Float64, Float64}
    rotating_x::Tuple{Float64, Float64}
    rotating_y::Tuple{Float64, Float64}
    rotating_z::Tuple{Float64, Float64}
    shearing_xy::Tuple{Float64, Float64}
    shearing_yx::Tuple{Float64, Float64}
    shearing_yz::Tuple{Float64, Float64}
    shearing_zy::Tuple{Float64, Float64}
    shearing_xz::Tuple{Float64, Float64}
    shearing_zx::Tuple{Float64, Float64}
    translating_x::Tuple{Int64, Int64}
    translating_y::Tuple{Int64, Int64}
    translating_z::Tuple{Int64, Int64}
    function RandomAffineTransformation3d(scx, scy, scz, rotx, roty, rotz, shxy, shyx, shyz, shzy, shxz, shzx, tax, tay, taz)
        if !(0 <= scx[1] <= scx[2]) 
            error("inconsistent x scaling")
        elseif !(0 <= scy[1] <= scy[2]) 
            error("inconsistent y scaling")
        elseif !(0 <= scz[1] <= scz[2]) 
            error("inconsistent z scaling")
        elseif !(rotx[1] <= rotx[2]) 
            error("inconsistent x rotating")
        elseif !(roty[1] <= roty[2]) 
            error("inconsistent y rotating")
        elseif !(rotz[1] <= rotz[2]) 
            error("inconsistent z rotating")
        elseif !(shxy[1] <= shxy[2]) 
            error("inconsistent xy shearing")
        elseif !(shyx[1] <= shyx[2]) 
            error("inconsistent yx shearing")
        elseif !(shyz[1] <= shyz[2]) 
            error("inconsistent yz shearing")
        elseif !(shzy[1] <= shzy[2]) 
            error("inconsistent zy shearing")
        elseif !(shxz[1] <= shxz[2]) 
            error("inconsistent xz shearing")
        elseif !(shzx[1] <= shzx[2]) 
            error("inconsistent zx shearing")
        elseif !(tax[1] <= tax[2]) 
            error("inconsistent x translating")
        elseif !(tay[1] <= tay[2]) 
            error("inconsistent y translating")
        elseif !(taz[1] <= taz[2]) 
            error("inconsistent z translating")
        end
        return new(scx, scy, scz, rotx, roty, rotz, shxy, shyx, shyz, shzy, shxz, shzx, tax, tay, taz)
    end
end

function RandomAffineTransformation3d(;
    scale_x::Tuple{Float64, Float64}=(1.0, 1.0),
    scale_y::Tuple{Float64, Float64}=(1.0, 1.0),
    scale_z::Tuple{Float64, Float64}=(1.0, 1.0),
    rotate_x::Tuple{Real, Real}=(0.0, 0.0),
    rotate_y::Tuple{Real, Real}=(0.0, 0.0),
    rotate_z::Tuple{Real, Real}=(0.0, 0.0),
    shear_xy::Tuple{Float64, Float64}=(0.0, 0.0),
    shear_yx::Tuple{Float64, Float64}=(0.0, 0.0),
    shear_yz::Tuple{Float64, Float64}=(0.0, 0.0),
    shear_zy::Tuple{Float64, Float64}=(0.0, 0.0),
    shear_xz::Tuple{Float64, Float64}=(0.0, 0.0),
    shear_zx::Tuple{Float64, Float64}=(0.0, 0.0),
    shift_x::Tuple{Int64, Int64}=(0, 0),
    shift_y::Tuple{Int64, Int64}=(0, 0),
    shift_z::Tuple{Int64, Int64}=(0, 0)
)
    return RandomAffineTransformation3d(scale_x, scale_y, scale_z, rotate_x, rotate_y, rotate_z, shear_xy, shear_yx, shear_yz, shear_zy, shear_xz, shear_zx, shift_x, shift_y, shift_z)
end

function (A::RandomAffineTransformation3d)(image::AbstractArray)
    I = image                 
    I = scaling3d(I, randu(A.scaling_x), randu(A.scaling_y), randu(A.scaling_z))  
    I = rotate3d(I, randu(A.rotating_x), randu(A.rotating_y), randu(A.rotating_z))
    I = shear3d(I, randu(A.shearing_xy), randu(A.shearing_yx), randu(A.shearing_yz), randu(A.shearing_zy), randu(A.shearing_xz), randu(A.shearing_zx))
    I = translate3d(I, randi(A.translating_x), randi(A.translating_y), randi(A.translating_z))
    return I
end

function scaling3d(I::AbstractArray, sx::Float64, sy::Float64, sz::Float64)
    Scale = AffineMap([1/sx 0.0 0.0; 0.0 1/sy 0.0; 0.0 0.0 1/sz], [0.0, 0.0, 0.0])
    center_offset = (-).(div.(size(I), 2));  
    I_centered = OffsetArray(I, center_offset...);
    I_scaled = warp(I_centered, Scale, axes(I_centered); fillvalue = 0.0, interp = BSpline(Linear()))
    I_centered = OffsetArray(I_scaled, (-).(center_offset)...);
    return I_centered
end

function rotate3d(I::AbstractArray, a::Real, b::Real, g::Real)
    Rx = RotX(a)
    Ry = RotY(b)
    Rz = RotZ(g)
    R = Matrix(Rx) * Matrix(Ry) * Matrix(Rz)
    Rotation = AffineMap(R, [0.0, 0.0, 0.0]);
    center_offset = (-).(div.(size(I), 2));
    I_centered = OffsetArray(I, center_offset...);
    I_rotated = warp(I_centered, Rotation, axes(I_centered); fillvalue = 0.0, interp = BSpline(Linear()))
    I_centered = OffsetArray(I_rotated, (-).(center_offset)...);
    return I_centered
end

function shear3d(I::AbstractArray, sxy::Float64, syx::Float64, syz::Float64, szy::Float64, sxz::Float64, szx::Float64)
    Translation = AffineMap([1.0 sxy sxz; syx 1.0 syz; szx szy 1.0], [0.0, 0.0, 0.0])
    center_offset = (-).(div.(size(I), 2));
    I_centered = OffsetArray(I, center_offset...);
    I_shared = warp(I_centered, Translation, axes(I_centered); fillvalue = 0.0, interp = BSpline(Linear()))
    I_centered = OffsetArray(I_shared, (-).(center_offset)...);
    return I_centered
end

function translate3d(I::AbstractArray, x::Int64, y::Int64, z::Int64)
    Translation = AffineMap([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0], [x, y, z])
    center_offset = (-).(div.(size(I), 2));
    I_centered = OffsetArray(I, center_offset...);
    I_translated = warp(I_centered, Translation, axes(I_centered); fillvalue = 0.0, interp = BSpline(Linear()))
    I_centered = OffsetArray(I_translated, (-).(center_offset)...);
    return I_centered
end

end