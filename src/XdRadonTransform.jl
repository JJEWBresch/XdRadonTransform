module XdRadonTransform

using Interpolations: LinearInterpolation
using Sobol
using Statistics
using StatsFuns

export TestImages
export DataTransformations

export RadonTransform, RadonTransformRay
export Phantom, Sinogram

export RestrictedJohnTransform
export PhantomBox, RestrictedSinogram

export scale, tangent_grid

export uniform_radii, uniform_circle, fibonacci_spherical_directions, random_spherical_directions, sobol_spherical_directions, sw_distance

include("TestImages.jl")
include("DataTransformations.jl")

include("RadonTransform.jl")
include("RadonTransformFast.jl")   # R(P), Rθ(P) (fast; R(P; legacy=true): original), backproject
include("JohnTransform.jl")

end # module XdRadonTransform
