# Implementing a solver

Space charge solvers calculate the electric field from charged particles
and evaluate the field at caller-selected points. Each solver has a concrete
type that owns its scratch space and implements `solve!` and `field!` (see
[`src/interface.jl`](../src/interface.jl)). Constructors, physical models, and
supported devices are specific to each solver.

## Shared methods

```julia
solve!(solver, x, y, z, charge, gamma; mask=nothing)
field!(Ex, Ey, Ez, solver, xt, yt, zt; mask=nothing)
```

`solve!` computes the space charge field from particle positions and charge.
`gamma` is the beam's Lorentz factor. An optional Boolean `mask` includes entries marked
`true` and skips those marked `false`. Skipped evaluation points receive zero field.

`field!` evaluates the prepared solution, overwrites the caller's three output
vectors, and returns `(Ex, Ey, Ez)`. It can evaluate different point sets without
repeating preparation. The caller controls when to refresh the field.

## Arrays and physical conventions

- Particle inputs are one-dimensional `x`, `y`, `z`, and `charge` arrays with
  matching axes. Evaluation coordinates and outputs form a separate group with
  matching axes; its length may differ from the particle count.
- Positions are simultaneous lab-frame Cartesian coordinates in meters, with
  z along the beam's longitudinal direction.
- `charge[i]` is signed macroparticle charge in coulombs, including its weight.
  Do not renormalize surviving particles or apply another total-charge factor.
- Outputs are physical lab-frame electric-field components in V/m.
- Arrays must have compatible precision and backend placement. Document the
  types and devices the solver supports.

## Masks and input requirements

`mask=nothing` includes every entry. Otherwise, `mask` is a Boolean vector
with the same axes and compatible backend as its particle or evaluation group.
`true` includes an entry and `false` skips it.
Skip inactive particles before reading their coordinates or charges. Their
values may be nonfinite. For inactive evaluation points, return zero fields
without reading their coordinates. An empty evaluation batch succeeds.

Zero charge does not make a particle inactive. Net charge zero does not
imply a zero field. Document the solver's behavior for empty or all-inactive
particle batches, along with its physical domain restrictions.

## Ownership and reuse

The solver owns the state needed to evaluate its prepared field. Once preparation
has completed, changing the original particle arrays must not change that field.
Reuse solver storage across calls where practical. Evaluation writes into the
provided outputs and does not implicitly prepare a new solution.

Callers must call `solve!` successfully before evaluation. Calling `field!` before
a successful `solve!` is undefined behavior.

Device work may continue after a method returns. Keep inputs alive and unchanged
until queued reads complete, order preparation before evaluation, and synchronize
before host access when required. Do not solve or mutate a solver concurrently
with its evaluation.

## Implementing the methods

The following method sketch assumes a concrete `MySolver` and private
`_prepare!` and `_evaluate!` routines that handle its numerical work, input checks,
and masks. No common base type or internal storage layout is required.

```julia
import SpaceCharge: solve!, field!

function solve!(solver::MySolver, x, y, z, charge, gamma; mask=nothing)
    _prepare!(solver, x, y, z, charge, gamma; mask)
    return nothing
end

function field!(Ex, Ey, Ez, solver::MySolver, xt, yt, zt; mask=nothing)
    _evaluate!(Ex, Ey, Ez, solver, xt, yt, zt; mask)
    return Ex, Ey, Ez
end
```

Keep algorithm-specific options and extra operations on the concrete solver.
They do not become requirements for other implementations.

## Validation

- Check physical reference cases, units, signed-charge behavior, and gamma
  dependence within the model's assumptions.
- Check return values, complete output overwriting, different evaluation counts,
  repeated evaluation, and independence from particle-input mutation after
  preparation completes.
- Check masks with unreadable or nonfinite inactive entries, zero-charge
  particles, empty evaluation arrays, and documented particle-batch policies.
- Check supported precision, views, and devices, including actual GPU
  execution when GPU support is claimed. Check structural input errors.
- Measure preparation and evaluation separately after compilation and setup.
  Check storage reuse and unintended copying or allocation; synchronize GPU
  timings at measurement boundaries.
