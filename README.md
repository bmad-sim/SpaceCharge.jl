# SpaceCharge.jl

SpaceCharge.jl is a high-performance Julia package for 3D space-charge calculations, supporting both CPU and GPU acceleration. It is designed for beam physics and accelerator modeling applications.

## Algorithm and Citation

This package implements a parallel fast Fourier transform (FFT) based 3D space charge algorithm using integrated Green functions, as described in:

> Mayes, Christopher and Ryne, Robert and Sagan, David. "3D Space Charge in Bmad." 9th International Particle Accelerator Conference, 2018. [doi:10.18429/JACoW-IPAC2018-THPAK085](https://doi.org/10.18429/JACoW-IPAC2018-THPAK085)

The original Fortran implementation is available at: [https://github.com/RobertRyne/OpenSpaceCharge](https://github.com/RobertRyne/OpenSpaceCharge)

The package provides high-level routines to:
- Deposit weighted charged particles on a 3D rectangular grid.
- Calculate the space charge fields on this grid.
- Interpolate the field to an arbitrary point within its domain.

Convolutions of the Green functions and the charge density are performed efficiently with FFTs.

## Installation

To install SpaceCharge.jl, open the Julia REPL and run:

```julia
using Pkg
Pkg.add("SpaceCharge")
```

## Getting Started

- **Examples:** See `examples/` for scripts demonstrating basic usage, GPU acceleration, cathode boundary conditions, and analytical comparisons.
- **Benchmarks:** The `benchmark/` directory contains scripts to evaluate performance on different hardware and problem sizes.
- **Tests:** The `test/` directory provides a comprehensive test suite including an analytical comparison against an isotropic Gaussian bunch.

## Repeated field calculations

Reuse a mesh and particle field outputs between steps:

```julia
Ex, Ey, Ez = similar(x), similar(y), similar(z)
deposit!(mesh, x, y, z, q)
solve!(mesh)
interpolate_field!(Ex, Ey, Ez, mesh, x, y, z)
```

The allocating `interpolate_field(mesh, x, y, z)` remains supported. The solver
uses real FFTs with reusable scratch buffers. CPU plans use `FFTW.MEASURE` at
the first solve; use `solve!(mesh; fft_flags=FFTW.ESTIMATE)` to reduce setup time
for short runs. GPU plans use their backend defaults. FFT setup is amortized by
repeated solves, so measure both initialization and warmed tracking throughput.

CPU solves prune FFT lines that have zero input or discarded output by default.
The charge transform starts on a compact padded-x buffer; inverse transforms
retain the existing `n:2n-1` physical slices in each direction, including for
shifted cathode images. Green-kernel transforms stay full. This preserves the
direct field operator and adds no symmetry transform or physics approximation.
Use `solve!(mesh; pruned_fft=false)` to select full transforms when benchmarking
other grid/thread configurations. GPU continues to use full real transforms.
Changing this option rebuilds CPU plans and caches. Pruning adds two compact
real buffers (8 MiB at 64^3 Float64) and extra initial planning work; speedup
depends on shape and thread count.

For free space, the integrated field kernel is evaluated in one octant and
reflected with exact odd/even symmetry. Shifted cathode kernels retain the full
evaluation. The discretization, CIC weights, gamma factors, and physical field
crop are unchanged; floating-point rounding can differ.

`solve!` caches the three transformed Green kernels by default. Changes to exact
grid spacing, gamma, or cathode offset invalidate the relevant cache; changing
charge values does not. Replacing mesh arrays rebuilds the workspace. Each mesh
retains at most one free-space and one image cache. Each cache costs three
half-complex spectra (about 49 MiB for a 64³ Float64 mesh). Use
`solve!(mesh; cache_green=false)` for a changing mesh or to avoid allocating the
cache; previously allocated caches remain available for later reuse. Free them
and the FFT workspace with `mesh._workspace = nothing` if necessary. Translation
of a free-space mesh alone does not invalidate its kernel; cathode translations
change the image offset and do invalidate that kernel.

Use one mesh per independently concurrent solve; its scratch and cache are mutable.

Large CPU deposition batches use private grids and a fixed-order reduction,
avoiding atomic writes. This scratch is reused and limited to 64 MiB and eight
logical chunks per mesh; small batches retain the serial path. Start Julia with
`--threads=N` to enable CPU parallelism. GPU deposition continues to use the
existing atomic CIC kernel. Cathode fields accumulate directly from the FFT
scratch, without allocating an image mesh or image field each step.

## Contributing

Contributions are welcome! Please open issues or pull requests for bug reports, feature requests, or improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
