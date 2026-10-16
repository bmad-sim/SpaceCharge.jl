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

The `dev` API prepares a field solution from particles separately from evaluating it:

```julia
using SpaceCharge

x = [-0.001, 0.001]; y = [0.0, 0.0]; z = [0.0, 0.0] # metres, lab frame
charge = [0.5e-9, 0.5e-9] # signed macroparticle coulombs
sc = FFTIGF((32, 32, 32))
Ex, Ey, Ez = similar(x), similar(y), similar(z)
solve!(sc, x, y, z, charge; gamma=1) # returns sc
field!(Ex, Ey, Ez, sc, x, y, z)     # overwrites and returns (Ex, Ey, Ez), V/m
```

Repeated `field!` calls reuse the frozen field at any evaluation points inside the
FFT grid. Ordinary column views work without packing. Callers supply compatible
Float32/Float64 arrays on one backend and call `solve!` before `field!`. Basic input
checks are included; call-order tracking is intentionally omitted.

Both operations accept `active=nothing` or a Boolean mask. Inactive particle
coordinates and charges are never read; inactive evaluation points receive zero.
Empty/all-inactive particle batches are rejected; active zero-charge particles are valid.

Each `solve!(sc,x,y,z,charge)` fits bounds to the active particles. To override them
for one call, use `solve!(sc,x,y,z,charge; bounds=(lo,hi))`. The next call without
`bounds` fits again. Bounds are not a constructor option. Both grid faces are
included. Collapsed adaptive axes expand to nominal 1 micrometre cells. Use nonnegative z bounds and particle positions with `at_cathode=true`.
Supply lab z and signed macroparticle coulombs; the solver applies the gamma
transformation once and returns lab E in V/m. Gamma defaults to 1 on every solve.

For staged work, first establish geometry with a particle-based `solve!`. Then
`deposit!(sc,x,y,z,charge; active,clear=true)` uses that current geometry;
`clear=false` accumulates. Follow deposition with
`solve!(sc;gamma=...)`, then `field!`. Empty/all-inactive additions with
`clear=false` preserve existing charge; with `clear=true` they are rejected.

Plans, Green spectra and deposition scratch are reused after warm-up. Configure
FFTW threads before planning. `cache_green=false` bypasses spectrum caching;
`pruned_fft=false` selects full CPU transforms. GPU transforms are always full.
For CUDA, load CUDA and KernelAbstractions and pass `backend=get_backend(x)`.
Adaptive fitting uses a fused backend reduction and transfers only the final
six-number extent to the host on GPU. The backend selects the launch and owns
its temporary reduction storage. An explicit bounds override skips geometry
reduction, though a supplied mask still needs an `any` reduction to reject
all-inactive input. Synchronize GPU work for host
access or timing.

This replaces the mesh-based public API; no compatibility aliases are provided.

- **Examples:** See `examples/` for scripts demonstrating basic usage, GPU acceleration, cathode boundary conditions, and analytical comparisons.
- **Benchmarks:** The `benchmark/` directory contains scripts to evaluate performance on different hardware and problem sizes.
- **Tests:** The `test/` directory provides a comprehensive test suite including an analytical comparison against an isotropic Gaussian bunch.

## Contributing

Contributions are welcome! Please open issues or pull requests for bug reports, feature requests, or improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
