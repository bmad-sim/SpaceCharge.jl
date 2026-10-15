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

The `dev` API prepares a source separately from evaluating its electric field:

```julia
using SpaceCharge

x = [-0.001, 0.001]; y = [0.0, 0.0]; z = [0.0, 0.0] # metres, lab frame
charge = [0.5e-9, 0.5e-9] # signed macroparticle coulombs
sc = FFTIGF((32, 32, 32); bounds=((-0.01,-0.01,-0.01), (0.01,0.01,0.01)))
Ex, Ey, Ez = similar(x), similar(y), similar(z)
solve!(sc, x, y, z, charge; gamma=1) # returns sc
field!(Ex, Ey, Ez, sc, x, y, z)     # overwrites and returns (Ex, Ey, Ez), V/m
```

Targets may be a different length from the source, and repeated `field!` calls
reuse the frozen source even if the original source arrays have changed. Ordinary
column views work without packing. Arrays must use the solver's Float32/Float64
precision, one-based vector axes, and backend/device. Outputs must not overlap
one another, targets, or solver storage. There is no implicit device transfer.

Both operations accept `active=nothing` or a Boolean mask on the same backend.
Inactive source coordinates and charges are never read; inactive target outputs
are zero. Tracking state and conversion to signed coulombs remain caller-owned.
Empty/all-inactive sources produce a valid zero solution. Active data must be
finite, including coordinates of zero-charge particles.

Omit `bounds` to fit active source positions on each combined solve. Explicit
`bounds=(lo,hi)` are useful for fixed geometry, reproducible resolution, and probes
outside the source extent. The domain includes both faces: the upper face uses
the last cell. Active points outside the box are rejected. Collapsed adaptive axes
expand symmetrically to nominal 1 micrometre cells, enlarged if needed for floating
point resolution. Bounds that cannot represent distinct grid nodes are rejected.
An empty adaptive source has no geometric restriction on finite targets.
Adaptive GPU preparation reads a bounded extrema summary back to the host to
establish geometry. Fixed bounds avoid this reduction and synchronization.

`gamma` is finite and at least one, and defaults to **1 on every solve**. Supply
physical lab z; the solver stretches longitudinal separations and boosts transverse
E once. Outputs are lab E, without target charge, magnetic cancellation, or tracking
kicks. `at_cathode=true` retains the image-charge E convention at z=0 and requires
nonnegative z bounds/sources.

For selective recomputation, use `deposit!(sc,x,y,z,charge; active,clear=true)` on
fixed or previously fitted geometry, followed by `solve!(sc;gamma=...)`. Deposition
invalidates the field; `clear=false` accumulates into a valid deposited source.
An unprepared solver or failed refresh cannot be evaluated.

Plans, spectra, and bounded deposition scratch are reusable and created on first
preparation. Warm up with the expected batch size before timing. FFTW's thread
setting is global: configure it before planning and do not change it concurrently
with solves. `cache_green=false` bypasses spectrum caching; `pruned_fft=false`
selects full CPU transforms. GPU transforms are always full. CUDA is optional;
construct with `backend=get_backend(x)` after loading CUDA and KernelAbstractions.
GPU work follows the calling stream; synchronize at host access/timing boundaries
or before independent-stream use. Device data errors may surface at synchronization.
Concurrent `field!` calls require disjoint outputs; never solve and evaluate the
same instance concurrently.

This replaces the mesh-based public API; no compatibility aliases are provided.

- **Examples:** See `examples/` for scripts demonstrating basic usage, GPU acceleration, cathode boundary conditions, and analytical comparisons.
- **Benchmarks:** The `benchmark/` directory contains scripts to evaluate performance on different hardware and problem sizes.
- **Tests:** The `test/` directory provides a comprehensive test suite including an analytical comparison against an isotropic Gaussian bunch.

## Contributing

Contributions are welcome! Please open issues or pull requests for bug reports, feature requests, or improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
