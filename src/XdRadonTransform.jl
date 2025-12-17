module XdRadonTransform

using Interpolations: LinearInterpolation
using Statistics

export TestImages
export DataTransformations

export RadonTransform
export RadonTransform3d

export Phantom, Sinogram

export scale

export uniform_radii, uniform_circle, fibonacci_sphere, random_spherical_directions

include("TestImages.jl")
include("DataTransformations.jl")

include("RadonTransform.jl")
# include("RadonTransform3d.jl")

end # module XdRadonTransform
