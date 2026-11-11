# SpaceCharge.jl

SpaceCharge.jl calculates space charge fields for beam physics and
accelerator modeling. It currently provides an integrated Green function (IGF)
solver with CPU and GPU support.

## Installation

Requires Julia 1.10 or later. To install the `dev` branch:

```julia
using Pkg
Pkg.add(url="https://github.com/bmad-sim/SpaceCharge.jl.git", rev="dev")
```

## Getting started

Create a solver, prepare the field from particles, then evaluate it at the
particle positions. Positions are lab-frame coordinates in metres, charges are
signed macroparticle charges in coulombs, and the resulting electric field is
in V/m.

```julia
using SpaceCharge

x = [-0.001, 0.001]
y = [-0.0005, 0.0005]
z = [-0.002, 0.002]
charge = [0.5e-9, 0.5e-9]

sc = FFTIGF((32, 32, 32))
solve!(sc, x, y, z, charge, 1)

Ex, Ey, Ez = similar(x), similar(y), similar(z)
field!(Ex, Ey, Ez, sc, x, y, z)
```

## Public API

The shared interface separates preparing a field from evaluating it:

- `solve!(solver, x, y, z, charge, gamma; mask=nothing)` prepares the field
  from particle coordinates and charges. The Lorentz factor `gamma` must be finite
  and at least 1.
- `field!(Ex, Ey, Ez, solver, xt, yt, zt; mask=nothing)` evaluates the prepared
  field at the supplied points. It overwrites your output vectors and returns
  `(Ex, Ey, Ez)`. Call `solve!` first, then call `field!` to evaluate the field
  without computing the field again.

Supply equal-length coordinate and charge vectors for `solve!`. For `field!`,
allocate output vectors matching the evaluation coordinates. Arrays must use
the solver's precision and device. Evaluation points must be finite and
inside the prepared domain. An optional Boolean `mask` includes entries marked
`true` and skips those marked `false`. Skipped evaluation points receive zero field.

See the [examples](examples/) for CPU, GPU, cathode, and analytical comparisons.

## Prepared execution in a tracking loop

Keep the ordinary solve! and field! interface for standalone calls. A tracking
application that validates its inputs at setup can prepare resources once and
use the explicit unchecked execution interface:

    sc = FFTIGF((32, 32, 32))
    prepared = prepare!(sc, x, y, z, charge; mask)
    # Inside the loop, after updating valid particle data:
    solve_prepared!(prepared, x, y, z, charge, gamma; mask)
    field_prepared!(Ex, Ey, Ez, prepared, x, y, z; mask)

Preparation checks array axes, precision, backend, and the initial fitted
geometry, then allocates FFT plans, Green-spectrum storage, and CPU deposition
scratch. It allows empty/all-inactive batches at setup and does not perform a
physical solve or change the current field. The caller validates active charges
and the field-output layout before entering its loop.

The prepared calls do not repeat input validation or select/rebuild workspaces.
The caller must keep the prepared particle count/layout and solver resources
unchanged, supply at least one active particle, finite active data and a finite
representable gamma >= 1, fitted bounds with finite positive representable
spacings, and evaluation points inside the current bounds.
Call prepare! again after changing array layout/count, grid, backend, precision,
solver arrays/options, or FFT thread settings. Handles share mutable solver
state and are not safe for concurrent use.

Particle coordinates, charges, inclusion mask, and gamma may change each step.
Bounds are fitted each solve unless explicitly supplied, and Green spectra are
refreshed when the geometry or gamma changes, including cathode image offsets.
Numerical deposition, FFT convolution, and interpolation kernels are shared
with the ordinary interface.

## Citation

Mayes, Christopher, Robert Ryne, and David Sagan. "3D Space Charge in Bmad."
IPAC 2018. [doi:10.18429/JACoW-IPAC2018-THPAK085](https://doi.org/10.18429/JACoW-IPAC2018-THPAK085).
The original Fortran implementation is available in
[OpenSpaceCharge](https://github.com/RobertRyne/OpenSpaceCharge).

## Contributing

Bug reports and pull requests are welcome.

See the [solver interface guide](docs/solver-interface.md) for implementing a solver.

## License

SpaceCharge.jl is licensed under the [MIT License](LICENSE).
