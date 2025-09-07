module TestImages

using Images
using Luxor

using LinearAlgebra
using StaticArrays
using ProgressMeter

export Circle, Empty, Polygon, Square, Star, Triangle
export OrbAndCross, Shield
export render
export extend_image
export generate_academic_classes, generate_ml_classes
export generate_3d_rot
export generate_phantom
export generate_academic_classes_3d

abstract type AbstractShape end
abstract type AbstractBaseShape <: AbstractShape end
abstract type AbstractComposedShape <: AbstractShape end

struct Circle <: AbstractBaseShape
    rotation::Real
    scale::Tuple{Real,Real}
end

Circle(; rotation::Real=0, scale::Tuple{Real,Real}=(1, 1)) = Circle(rotation, scale)

struct Polygon <: AbstractBaseShape
    edges::Integer
    rotation::Real
    scale::Tuple{Real,Real}
    Polygon(n, r, s) = (n <= 0) ? error("negative edges") : new(n, r, s)
end

function Polygon(edges; rotation::Real=0, scale::Tuple{Real,Real}=(1, 1))
    return Polygon(edges, rotation, scale)
end

function Triangle(; rotation::Real=0, scale::Tuple{Real,Real}=(1, 1))
    return Polygon(3, rotation + π / 2, scale)
end

function Square(; rotation::Real=0, scale::Tuple{Real,Real}=(1, 1))
    return Polygon(4, rotation + π / 4, scale)
end

struct Star <: AbstractBaseShape
    rays::Integer
    rotation::Real
    scale::Tuple{Real,Real}
    Star(n, r, s) = (n <= 0) ? error("negative rays") : new(n, r, s)
end

function Star(rays; rotation::Real=0, scale::Tuple{Real,Real}=(1, 1))
    return Star(rays, rotation, scale)
end

struct Empty <: AbstractBaseShape end

struct OrbAndCross <: AbstractComposedShape
    orb::AbstractBaseShape
    cross::AbstractBaseShape
    rotation::Real
    scale::Tuple{Real,Real}
end

function OrbAndCross(orb, cross; rotation::Real=0, scale::Tuple{Real,Real}=(1, 1))
    return OrbAndCross(orb, cross, rotation, scale)
end

struct Shield <: AbstractComposedShape
    emblem::AbstractBaseShape
    rotation::Real
    scale::Tuple{Real,Real}
end

function Shield(emblem; rotation::Real=0, scale::Tuple{Real,Real}=(1, 1))
    return Shield(emblem, rotation, scale)
end

struct Emblem <: AbstractComposedShape
    emblem::AbstractBaseShape
    rotation::Real
    scale::Tuple{Real,Real}
end

function Emblem(emblem; rotation::Real=0, scale::Tuple{Real,Real}=(1, 1))
    return Emblem(emblem, rotation, scale)
end

function render(
    shape::AbstractShape; size::Tuple{Integer,Integer}=(128, 128), width::Real=2
)
    initiate_luxor_drawing(size, width)
    luxor_draw(shape)
    return extract_luxor_drawing()
end

function initiate_luxor_drawing(size::Tuple{Integer,Integer}, width::Real)
    (x, y) = size
    Drawing(x, y, :png)
    origin()
    background("black")
    setline(width)
    Luxor.scale(x / 2, y / 2)
    return nothing
end

function luxor_draw(::Empty) end

function luxor_draw(circle::Circle)
    apply_local_transform(circle)
    luxor_draw_circle()
    annul_local_transform(circle)
    return nothing
end

function luxor_draw_circle()
    sethue("black")
    circle(Point(0, 0), 0.75; action=:fill)
    sethue("white")
    circle(Point(0, 0), 0.75; action=:stroke)
    return nothing
end

function luxor_draw(polygon::Polygon)
    apply_local_transform(polygon)
    luxor_draw_polygon(polygon.edges)
    annul_local_transform(polygon)
    return nothing
end

function luxor_draw_polygon(edges::Integer)
    sethue("black")
    ngon(Point(0, 0), 0.75, edges; action=:fill)
    sethue("white")
    ngon(Point(0, 0), 0.75, edges; action=:stroke)
    return nothing
end

function luxor_draw(star::Star)
    apply_local_transform(star)
    luxor_draw_star(star.rays)
    annul_local_transform(star)
    return nothing
end

function luxor_draw_star(rays::Integer)
    if rays > 0
        sethue("white")
        for k in 1:rays
            move(Point(0, 0))
            line(Point(0.75sin(k * 2π / rays), -0.75cos(k * 2π / rays)))
        end
        strokepath()
    end
    
    return nothing
end

function luxor_draw(orbandcross::OrbAndCross)
    apply_local_transform(orbandcross)
    luxor_draw_crosspiece()
    luxor_draw_orb(orbandcross)
    luxor_draw_cross(orbandcross)
    annul_local_transform(orbandcross)
    return nothing
end

function luxor_draw_crosspiece()
    sethue("white")
    move(Point(0, -0.5))
    line(Point(0, 0.5))
    strokepath()
    return nothing
end

function luxor_draw_orb(orbandcross::OrbAndCross)
    translate(0, 0.5)
    Luxor.scale(0.5)
    luxor_draw(orbandcross.orb)
    Luxor.scale(2)
    translate(0, -0.5)
    return nothing
end

function luxor_draw_cross(orbandcross::OrbAndCross)
    translate(0, -0.5)
    Luxor.scale(0.5)
    luxor_draw(orbandcross.cross)
    Luxor.scale(2)
    translate(0, 0.5)
    return nothing
end

function luxor_draw(shield::Shield)
    apply_local_transform(shield)
    luxor_draw_shield()
    luxor_draw_emblem(shield)
    annul_local_transform(shield)
    return nothing
end

function luxor_draw(emblem::Emblem)
    apply_local_transform(emblem)
    luxor_draw_emblem(emblem)
    annul_local_transform(emblem)
    return nothing
end

function luxor_draw_shield()
    sethue("white")
    move(Point(-0.75, -0.75))
    line(Point(-0.75, -0.5))
    curve(Point(-0.75, 0.5), Point(0, 0.75), Point(0, 0.75))
    curve(Point(0, 0.75), Point(0.75, 0.5), Point(0.75, -0.5))
    line(Point(0.75, -0.75))
    closepath()
    strokepath()
    return nothing
end

function luxor_draw_emblem(shield::Shield)
    translate(0, -0.125)
    Luxor.scale(0.5)
    luxor_draw(shield.emblem)
    Luxor.scale(2)
    translate(0, 0.125)
    return nothing
end

function luxor_draw_emblem(emblem::Emblem)
    translate(0, -0.125)
    Luxor.scale(0.5)
    luxor_draw(emblem.emblem)
    Luxor.scale(2)
    translate(0, 0.125)
    return nothing
end

function apply_local_transform(shape::AbstractShape)
    (x, y) = shape.scale
    ϕ = shape.rotation
    rotate(-ϕ)
    Luxor.scale(x, y)
    return nothing
end

function annul_local_transform(shape::AbstractShape)
    (x, y) = shape.scale
    ϕ = shape.rotation
    Luxor.scale(1 / x, 1 / y)
    rotate(ϕ)
    return nothing
end

function extract_luxor_drawing()
    image = image_as_matrix()
    finish()
    return Gray{Float64}.(image)
end

function extend_image(image::AbstractMatrix, shape::Tuple{Int64, Int64})
    (dim_y, dim_x) = size(image)
    if dim_y > shape[1] || dim_x > shape[2]
        error("dimension mismatch")
    end
    I = zeros(Gray{Float64}, shape)
    id_x = max((shape[2] - dim_x) ÷ 2, 1)
    id_y = max((shape[1] - dim_y) ÷ 2, 1)
    I[id_y:(id_y + dim_y - 1), id_x:(id_x + dim_x - 1)] .= Gray{Float64}.(image)
    return I
end

function extend_image(image::AbstractMatrix, shape::Int64)
    return extend_image(image, (shape, shape))
end


function generate_academic_classes(images::AbstractArray, label::AbstractArray; class_size::Int64=10, shuf::Int64=0)
    num = length(images)
    classes = []
    labels = []
    for k in 1:num
        for l in 1:class_size
            append!(classes, images[k,:,:])
            append!(labels, label[k])
        end
    end
    if shuf==1
        return shuffle_data(classes, labels)
    else
        return classes, labels
    end
end

function shuffle_data(classes::AbstractArray, labels::AbstractArray)
    dim = length(classes)
    p = shuffle(1:dim)
    classes = classes[p]
    labels = labels[p]
    return classes, labels
end

function generate_ml_classes(trainset, labels::AbstractArray, size_classes::Int64, thres::Float64=0.5)
    target = trainset.targets;
    pos = findall(x->x==labels[1], target)
    pos = pos[1:size_classes]
    for k in 2:size(labels)[1]
        pos1 = findall(x->x==labels[k], target)
        append!(pos, pos1[1:size_classes])
    end
    trainset.features[trainset.features.<thres] .= 0.0
    data_mnist_tens = Gray{Float64}.(permutedims(trainset[pos].features, [3,2,1]))
    dim = length(labels)*size_classes
    data_mnist = [extend_image(imresize(data_mnist_tens[i,:,:], ratio=2),128) for i in 1:dim]
    label_mnist = trainset[pos].targets
    return data_mnist, label_mnist #shuffle_data(data_mnist, label_mnist)
end

function generate_3d_rot(image::AbstractMatrix; order::Float64=2.0)
    (dim_y, dim_x) = size(image)
    A = Array{Float64}(image)
    I = zeros(dim_y, dim_y, dim_x)
    diam = Int(dim_x/2)
    for z in 1:dim_y
        for i in 1:dim_y
            for j in 1:dim_x
                for l in 0:diam
                    if Int(floor((abs(i - diam)^order + abs(j - diam)^order)^(1/order)))==l
                        I[z,i,j] = A[z,diam+1-l]
                    end
                end
            end
        end
    end
    return I
end

"""
#    ellipsoid_value(x, y, z, c, a, rot, intensity)
# Returns the value at point (x, y, z) due to an ellipsoid centered at `c`,
# with semi-axes `a`, rotation `rot`, and given `intensity`.
"""
function ellipsoid_value(x, y, z, c::SVector{3,Float64},a::SVector{3,Float64}, rot::SMatrix{3,3,Float64},intensity::Float64)
    p = SVector{3,Float64}((x, y, z)) - c
    p_rot = rot * p
    val = sum((p_rot ./ a).^2)
    return val ≤ 1.0 ? intensity : 0.0
end

"""
#    generate_phantom(N)
# Generates a 3D Shepp–Logan phantom on an NxNxN grid over [-1, 1]^3.
"""
function generate_phantom(N::Int=128)
    xs = range(-1.0, 1.0; length=N)
    ys = xs
    zs = xs
    phantom = zeros(Float64, N, N, N)
    I3 = rotz(0.0)

    # Define ellipsoids: [intensity, center, axes, rotation]
    ellipsoids = [
        (+1.0,  SVector{3,Float64}((0.0, 0.0, 0.0)),     SVector{3,Float64}((0.69, 0.92, 0.9)),   I3),
        (-0.8,  SVector{3,Float64}((0.0, 0.0, 0.0)),     SVector{3,Float64}((0.6624, 0.8740, 0.88)), I3),
        (-0.2,  SVector{3,Float64}((0.22, 0.0, -0.25)),  SVector{3,Float64}((0.41, 0.16, 0.21)),  rotz(72)),
        (-0.2,  SVector{3,Float64}((-0.22, 0.0, -0.25)), SVector{3,Float64}((0.41, 0.16, 0.21)),  rotz(-72)),
        (+0.1,  SVector{3,Float64}((0.0, 0.35, -0.25)),  SVector{3,Float64}((0.25, 0.17, 0.21)),  I3),
        (+0.1,  SVector{3,Float64}((0.0, -0.35, -0.25)), SVector{3,Float64}((0.25, 0.17, 0.21)),  I3),
        (+0.2,  SVector{3,Float64}((0.0, 0.0, 0.25)),    SVector{3,Float64}((0.2, 0.3, 0.3)),     I3),
        (+0.2,  SVector{3,Float64}((0.25, 0.0, 0.25)),   SVector{3,Float64}((0.1, 0.1, 0.1)),     I3),
        (+0.2,  SVector{3,Float64}((-0.25, 0.0, 0.25)),  SVector{3,Float64}((0.1, 0.1, 0.1)),     I3),
        (+0.2,  SVector{3,Float64}((0.0, 0.0, 0.5)),     SVector{3,Float64}((0.1, 0.1, 0.1)),     I3),
    ]

    @showprogress 1 "Generating phantom..." for i in 1:N, j in 1:N, k in 1:N
        val = 0.0
        x, y, z = xs[i], ys[j], zs[k]
        for (intensity, c, a, rot) in ellipsoids
            val += ellipsoid_value(x, y, z, c, a, rot, intensity)
        end
        phantom[i, j, k] = val
    end
    return phantom
end

"""
#    rotz(theta_deg)
# Returns a 3x3 rotation matrix around z-axis by theta_deg degrees.
"""
function rotz(theta_deg::Real)
    θ = deg2rad(theta_deg)
    SMatrix{3,3,Float64}((cos(θ), -sin(θ), 0.0, sin(θ), cos(θ), 0.0, 0.0, 0.0, 1.0))
end

function generate_academic_classes_3d(images::AbstractArray, label::AbstractArray; class_size::Int64=10)
    num = length(images)
    classes = []
    labels = []
    for k in 1:num
        for l in 1:class_size
            append!(classes, images[k,:,:,:])
            append!(labels, label[k])
        end
    end
    return classes, labels
end

end