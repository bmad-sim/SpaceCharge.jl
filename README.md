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
