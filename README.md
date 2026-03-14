# XdRadonTransform

This repository contains the code and experiments for the papers:

1. Robert Beinert, Jonas Bresch, Michael Quellmalz
   '[A Discrete Radon Transform Based on the Area of Cube-Plane Intersection](https://doi.org/10.1007/978-3-031-92366-1_19)',
   arxiv:xxxx.xxxxx,
   2026.

Please cite the papers if you use the code.

## Usage

To initialize the environment for the experiments,
start a Julia REPL in the main directory
and use the following commands:

```julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Examples

The basic usage of the code is shown
in the following Pluto notebooks:

- `Pluto/iterface.jl`:
  Simple example for setting the Radon transform. 
  Example usage on a simple object.

To run Pluto notebooks,
open a Julia REPL in the main directory
and start Pluto via

```julia
using Pluto; Pluto.run()
```

## Experiments

The simulations in [1] can be reproduced
using the following Pluto notebooks:

- `IEEE/Tab2_3_shape_matching_nonaff.jl`:
  Nearest neighbour classification for ModelNet10 dataset
  using the closed formula of the Radon transform for shape matching via trace-transform (Daras et al.).
  This experiment corresponds to
  Table 2 and 3 and Figure 6.
- `IEEE/Tab3_4_shape_recognition_nonaff.jl`:
  Nearest neighbour classification for for ModelNet10 dataset
  using the closed formula of the Radon transform for object recognition vi mNR-CDT (Beckmann et al.).
  This experiment corresponds to
  Table 4 and 5 and Figure 7.
- `IEEE/SecIII.E_shape_matching_aff.jl`:
  Setting as in (a) on affine transformed object classes.
- `IEEE/SecIII.E_shape_recognition_aff.jl`:
  Setting as in (b) on affine transformed object classes.
- `IEEE/Fig8_9_clustering_2d_sliced_wasserstein.jl`:
  Best fitting histograms for 2D.
  This experiment corresponds to
  Figure 8 and 9 (top).
- `IEEE/Fig8_9_clustering_3d_sliced_wasserstein.jl`:
  Best fitting histograms for 3D.
  This experiment corresponds to
  Figure 9 (bottom).
- `IEEE/Fig10_11_sliced_Wasserstein_barycenter.jl`:
  Sliced Wasserstein barycenter calculation.
  This experiment corresponds to
  Figure 10 and 11.
- `IEEE/Fig12_Monte_Carlo_4d.jl`:
  Monte Carlo integration - a comparison for approximating the Radon transform in 4D.
  This experiment corresponds to
  Figure 12.

## License

This code is licensed under a MIT License.
