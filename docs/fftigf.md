# FFTIGF

`FFTIGF` computes a three-dimensional electric field by depositing particle
charges on a uniform Cartesian grid and convolving them with an integrated
Green function using FFTs. It supports free space and an optional image-charge
contribution from a cathode at `z = 0`.

This guide covers the FFTIGF-specific API, numerical method, and practical
choices. For the shared solver contract, see the [solver interface](solver-interface.md).

## Constructor

```julia
using SpaceCharge, KernelAbstractions, FFTW

sc = FFTIGF((32, 32, 64);
    T=Float64,
    backend=CPU(),
    fft_flags=FFTW.MEASURE,
    cache_green=true,
    pruned_fft=true,
    at_cathode=false,
    field_method=:direct,
)
```

The tuple gives the number of **grid nodes** in `(x, y, z)`, including both
endpoints. Each dimension must be an `Int` of at least 2. Powers of two are
preferred for performance but not required. The constructor sets the grid
dimensions and allocates charge and field arrays.

| Keyword | Default | Meaning |
| --- | --- | --- |
| `T` | `Float64` | Array and calculation precision: `Float32` or `Float64`. |
| `backend` | `CPU()` | KernelAbstractions backend for allocation and kernels. GPU use also requires a compatible FFT implementation; see the CUDA example below. |
| `fft_flags` | `FFTW.MEASURE` | CPU FFTW planning effort. `FFTW.ESTIMATE` reduces planning work for short runs. GPU plans use their backend defaults. |
| `cache_green` | `true` | Retain Green-function spectra for reuse when the geometry and `gamma` permit it: three field spectra for `:direct`, one potential spectrum for `:potential_fd`. |
| `pruned_fft` | `true` | On CPU, skip transforms over known zeros and inverse-transform regions that will be discarded. GPU transforms are always full. |
| `at_cathode` | `false` | Add the electric field of opposite-sign charges reflected across `z = 0`. |
| `field_method` | `:direct` | Integrate the electric-field kernel directly; `:potential_fd` instead integrates the potential kernel and uses finite differences as described below. These are the only supported values. |

Select the potential method with
`FFTIGF((32, 32, 64); field_method=:potential_fd)`. Both methods use the same
`solve!`, `deposit!`, and `field!` API, charge grid, and public field layout.

Create a separate solver to change grid dimensions, precision, backend, or
constructor options. FFT plans and their working arrays are allocated lazily
on the first solve.

## Particle workflow and units

```julia
using SpaceCharge

x = [-0.001, 0.001]
y = [-0.0005, 0.0005]
z = [-0.002, 0.002]
charge = [0.5e-9, 0.5e-9]
gamma = 2.0

sc = FFTIGF((32, 32, 32))
solve!(sc, x, y, z, charge, gamma)

Ex, Ey, Ez = similar(x), similar(y), similar(z)
field!(Ex, Ey, Ez, sc, x, y, z)

# Evaluate the same field at different points within the domain.
xt, yt, zt = [0.0], [0.0], [0.0]
Ext, Eyt, Ezt = similar(xt), similar(yt), similar(zt)
field!(Ext, Eyt, Ezt, sc, xt, yt, zt)
```

`x`, `y`, and `z` are simultaneous lab-frame Cartesian coordinates in metres;
`z` is the longitudinal direction. `charge` contains signed macroparticle
charges in coulombs, including the number of physical particles represented by
each macroparticle. It is not a particle count or a charge density. All three
returned electric components are lab-frame fields in V/m.

`gamma` is one Lorentz factor for the entire distribution. Supply it as a
required positional argument to **every** `solve!` call, including the
grid-only form. It must be a finite real value at least 1, representable in the
solver's precision. Use `1` for a stationary electrostatic calculation. Supply
lab-frame coordinates directly; the kernel accounts for longitudinal
stretching internally.

This is an electrostatic rest-frame model with one common `gamma`. The solver
does not advance particles, produce a magnetic field, or apply a magnetic-force
cancellation factor. A tracking application is responsible for constructing
the force appropriate to its model.

### Arrays and grid storage

Use one-dimensional, one-based arrays or views. Coordinate and charge vectors
in a particle call must have matching axes. Each `field!` call requires
matching axes for its evaluation coordinates and three output vectors; these
can have a different length from the particle arrays used to prepare the field.
All floating-point arrays must have exactly the solver's element type and use
its backend and device. There is no automatic precision conversion or transfer
between CPU and GPU. Ordinary column views and strided vector views are
supported. Use separate output buffers for the three components.

The solver exposes the following useful grid data after a solve:

| Property | Layout and units |
| --- | --- |
| `sc.grid_size` | `(nx, ny, nz)` node counts. |
| `sc.min_bounds`, `sc.max_bounds` | Three-tuples of lab-frame endpoint coordinates in metres. |
| `sc.delta` | Three-tuple of lab-frame node spacings in metres. |
| `sc.gamma` | Lorentz factor used by the latest grid solve. |
| `sc.rho` | `(nx, ny, nz)` array of **deposited charge per node in coulombs**, despite the name `rho`. |
| `sc.efield` | `(nx, ny, nz, 3)` array of electric field in V/m; the last index selects `Ex`, `Ey`, or `Ez`. |

Node `(i, j, k)` is at
`sc.min_bounds .+ (i-1, j-1, k-1) .* sc.delta`. Deposition conserves the sum of
included particle charges up to floating-point roundoff. There is no extra
cell-volume factor to apply to `sc.rho` or `sc.efield`.

## API and field reuse

### Prepare a field from particles

```julia
solve!(sc, x, y, z, charge, gamma; mask=nothing, bounds=nothing)
```

This call selects the bounds, clears and deposits the charge grid, and computes
all three grid-field components. It returns `nothing`. At least one particle
must be included, even when explicit bounds are supplied. Included particles
may have zero charge, and a distribution with zero net signed charge is valid.

Included coordinates and charges must be finite. With supplied bounds, every
included particle must lie within the closed domain. Treat these as caller
requirements: the numerical kernels do not comprehensively validate individual
coordinates and charges. Invalid data can produce invalid fields or indexing
errors.

### Evaluate an existing field

```julia
field!(Ex, Ey, Ez, sc, xt, yt, zt; mask=nothing)
```

This call trilinearly interpolates the grid field, overwrites the output
vectors, and returns `(Ex, Ey, Ez)`. It leaves the charge grid and field grid
unchanged. Included evaluation coordinates must be finite and lie inside the
closed bounds. Both lower and upper faces are supported; an upper-face point
uses the upper node of the last cell. There is no supported extrapolation
outside the domain.

Call `solve!` before evaluating a field. Once computed, that field is
independent of subsequent changes to the input particle arrays. Repeated
`field!` calls can use different positions, lengths, and masks without running
the FFTs again. An empty evaluation batch returns the empty output vectors.

### Masks

Each particle or evaluation call accepts an optional Boolean vector `mask`.
`true` includes an entry; `false` skips it before its coordinates or charge are
read. The mask must have the same axes as that call's vectors and reside on
the same backend. CPU Boolean vectors and `BitVector`s are supported; on CUDA,
use a device Boolean vector.

The deposition mask and evaluation mask are independent. Skipped particles
neither contribute charge nor affect automatically fitted bounds. Skipped
evaluation points receive zero in all three output components. For example:

```julia
using SpaceCharge

x = [-0.001, 0.001, NaN]
y = [-0.0005, 0.0005, NaN]
z = [-0.002, 0.002, NaN]
charge = [0.5e-9, 0.5e-9, NaN]
mask = Bool[true, true, false]

sc = FFTIGF((16, 16, 16))
solve!(sc, x, y, z, charge, 1; mask)
Ex, Ey, Ez = similar(x), similar(y), similar(z)
field!(Ex, Ey, Ez, sc, x, y, z; mask)
@assert Ex[3] == Ey[3] == Ez[3] == 0
```

### Separate deposition and grid solving

```julia
deposit!(sc, x, y, z, charge; mask=nothing, clear=true)
solve!(sc, gamma)
```

Use these stages after a particle-based solve has established the desired
geometry. `deposit!` uses the existing bounds and spacings; it does not refit
the grid or update `sc.efield`. With `clear=true`, it replaces the deposited
charge. With `clear=false`, it adds to the current charge grid. It returns
`sc`. Included particles must satisfy the same finite and in-bounds requirements
as a particle solve with supplied bounds.

An empty or all-false-mask deposition is an error with `clear=true`; with
`clear=false`, it leaves the charge unchanged. Call `solve!(sc, gamma)` after
finishing deposition to update the field. This grid-only solve returns
`nothing` and uses the current charge grid without changing its geometry or
redepositing particles.

For example, accumulate two particle batches on one fixed grid:

```julia
using SpaceCharge

x1, y1, z1, q1 = [-0.001], [0.0005], [-0.002], [0.5e-9]
x2, y2, z2, q2 = [0.001], [-0.0005], [0.002], [0.5e-9]
bounds = ((-0.004, -0.004, -0.004), (0.004, 0.004, 0.004))
gamma = 2.0
sc = FFTIGF((32, 32, 32))

# Establish the domain and compute the first field.
solve!(sc, x1, y1, z1, q1, gamma; bounds)

# Replace the deposited charges, then add another batch.
deposit!(sc, x1, y1, z1, q1)
deposit!(sc, x2, y2, z2, q2; clear=false)
solve!(sc, gamma)

xt, yt, zt = [0.0], [0.0], [0.0]
Ex, Ey, Ez = similar(xt), similar(yt), similar(zt)
field!(Ex, Ey, Ez, sc, xt, yt, zt)

# Reuse these same lab-frame charges and geometry with another gamma.
solve!(sc, 3.0)
field!(Ex, Ey, Ez, sc, xt, yt, zt)
```

Changing `gamma` in the last step changes the field kernel; it does not move
or rescale the deposited lab-frame distribution. Serialize calls that modify
one solver, since its charge, field, and FFT workspace are shared mutable
storage. For independent simultaneous solves, use separate solver instances.

## Bounds and spatial resolution

For each axis, node spacing is

$$
\Delta_k = \frac{h_k-l_k}{n_k-1},
$$

where `lo = (lx, ly, lz)` and `hi = (hx, hy, hz)` are the lower and upper bounds.
Set them for a particle solve with `bounds=(lo, hi)`. The endpoints must be
finite and strictly ordered in every axis after conversion to `T`; the
spacing and its inverse must also be finite and positive.

Without `bounds`, each particle solve fits the coordinate extrema of the
included particles. For a nonzero span `s` along an axis, it adds
`1e-6 * s / (n-1)` to each side. This small endpoint padding can round away in
the selected precision and does not provide a substantial surrounding
evaluation region. When all included coordinates on an axis equal `a`, it
constructs a centered interval with nominal spacing
`max(T(1e-6), T(4) * eps(a))`. This fallback prevents a collapsed grid; choose
explicit bounds when the physical width of a point-like or flat distribution
matters.

An override applies only to that particle `solve!` call. A later particle solve
without `bounds` fits again. `deposit!` and `solve!(sc, gamma)` retain the
geometry already established. To retain an automatically fitted domain for
later particle solves, save and pass `(sc.min_bounds, sc.max_bounds)`.

Choose bounds to contain both the particles and every point where you intend
to evaluate the field. Increasing the domain at fixed node counts makes the
spacing coarser. Increasing node counts at fixed bounds resolves smaller
features but uses more memory and FFT work. Automatically fitted bounds can
change the resolution from one particle distribution to the next; fixed
bounds are useful for convergence studies and consistent repeated solves.

The bounds specify the computational sampling region. In free-space mode
they do not represent conducting walls or periodic physical boundaries.

## Numerical method

The calculation combines cloud-in-cell deposition, a cell-averaged Green
function selected by `field_method`, FFT convolution, and trilinear interpolation.

1. **Deposit charge.** A particle in a grid cell contributes to its eight
   surrounding nodes. If its fractional cell coordinates are `(fx, fy, fz)`,
   the weights are products of `fx` or `1-fx`, `fy` or `1-fy`, and `fz` or
   `1-fz`. The eight weights sum to one. The resulting nodal charges are
   denoted by `Qj` below.
2. **Integrate the kernel.** Both methods work with the longitudinally
   stretched displacement and cell dimensions

   $$
   \mathbf R'=(R_x,R_y,\gamma R_z),\qquad
   \boldsymbol\Delta'=(\Delta_x,\Delta_y,\gamma\Delta_z),\qquad
   V'=\Delta_x\Delta_y\gamma\Delta_z.
   $$

   Each nodal charge is represented by a uniform rectangular cell centered on
   that node. Boundary-node cells extend half a spacing beyond the bounds;
   the kernel does not clip them at the domain boundary. For `:direct`, the
   kernel for component `c` is the average of the Coulomb electric-field kernel
   over this cell in the stretched coordinates:

   $$
   G_c(\mathbf R)=\frac{s_c}{V'}
   \int_{-\Delta'_x/2}^{\Delta'_x/2}
   \int_{-\Delta'_y/2}^{\Delta'_y/2}
   \int_{-\Delta'_z/2}^{\Delta'_z/2}
   \frac{R'_c-u_c}{|\mathbf R'-\mathbf u|^3}\,du_z\,du_y\,du_x,
   \qquad (s_x,s_y,s_z)=(\gamma,\gamma,1).
   $$

   The integrals are evaluated analytically using an antiderivative at eight
   cell corners and an alternating corner sum. This method integrates electric
   components directly, without differentiating a potential grid. For free
   space, each component is odd in its own displacement coordinate and even
   in the others. Reflection constructs the full kernel from an octant and
   sets its own-component central plane to zero.

   `:potential_fd` instead uses the exact analytic cell integral of
   `1/|R' - u|`, with the same `1/V'` normalization, to obtain the rest-frame
   potential kernel. It uses no near/far approximation. Its free-space kernel
   is even in all three displacement coordinates.
3. **Convolve on the grid.** For `:direct`, the discrete field at node `i` is

   $$
   E_c(\mathbf r_i)=\frac{1}{4\pi\epsilon_0}
   \sum_j Q_j\,G_c(\mathbf r_i-\mathbf r_j).
   $$

   Both methods embed charge and kernel arrays on a grid doubled in every
   dimension: FFT lengths are `(2nx, 2ny, 2nz)`. Charge is zero padded. Real
   FFTs turn convolution into multiplication of spectra, followed by normalized
   inverse transforms cropped to the physical grid. This embedding implements
   free-space linear convolution in the retained region.

   `:direct` computes the field components. `:potential_fd` computes the
   rest-frame potential `Phi` in volts on the `(nx, ny, nz)` physical grid,
   then takes its negative derivatives. Along each axis, it uses second-order
   centered differences at interior nodes and second-order three-point
   one-sided differences at the two boundary faces. If an axis has only two
   nodes, both endpoints use the same two-point difference. With `Dk` denoting
   these differences using lab-frame spacings, the field is

   $$
   E_x=-\gamma D_x\Phi,\qquad E_y=-\gamma D_y\Phi,\qquad
   E_z=-\frac{1}{\gamma}D_z\Phi.
   $$

   The longitudinal `1/gamma` converts to a derivative in the stretched coordinate.
4. **Interpolate.** `field!` applies the same trilinear weights to the eight
   neighboring grid-field values at each evaluation point.

Thus `Ex = gamma * Ex_rest`, `Ey = gamma * Ey_rest`, and `Ez = Ez_rest`, where
the rest-frame calculation uses the stretched coordinates and unchanged
macroparticle charges. The factor `1/V'` is part of the kernel normalization;
deposition stores charge rather than density. The Coulomb factor uses the SI
permittivity, so no additional charge, volume, or relativistic factor is
needed to obtain the documented electric field.

## Cathode image field

With `at_cathode=true`, the solver adds an equal and opposite image of the
deposited distribution reflected across the fixed plane `z = 0`. It reverses
the nodal charges in `z`, negates them, and performs a second convolution with
a longitudinally shifted kernel. The shift is `zmin + zmax` in lab coordinates;
the kernel applies the same `gamma` stretching as for the real distribution.
The real deposited charge array is retained. Both field methods support this
construction. With `:potential_fd`, real and image potentials are accumulated
on the physical grid before taking finite differences.

```julia
using SpaceCharge

x, y, z, charge = [0.0], [0.0], [0.01], [1e-9]
sc = FFTIGF((32, 32, 32); at_cathode=true)
bounds = ((-0.02, -0.02, 0.0), (0.02, 0.02, 0.04))
solve!(sc, x, y, z, charge, 1; bounds)
Ex, Ey, Ez = similar(x), similar(y), similar(z)
field!(Ex, Ey, Ez, sc, x, y, z)
```

For a beam on the positive side of the plane, choose coordinates and bounds
appropriate to that region. Cathode mode does not move the plane, clip
particles at it, or alter the supplied or automatically fitted bounds. To
evaluate the field on the cathode, include `z = 0` in the domain. At the plane,
the ideal image construction cancels tangential electric field and doubles
the normal component of the free-space field, up to numerical error. It
models one infinite plane; finite electrodes and other boundary shapes are
outside this model.

In a tested `(7, 8, 9)` cathode case, `:potential_fd` with `Float32` left
tangential residuals of a few parts per thousand of the free-space field scale
at the plane. Both centered and one-sided stencils cancel tangential fields
in exact arithmetic; this residual comes from cancellation in the eight-corner
potential sum and roundoff amplified by differentiation. Use `Float64` when
stronger cancellation is needed. The observed residual is not a general error bound.

## CPU and GPU use

CPU arrays and `Float64` are the defaults. CPU deposition uses Julia threads
when the particle count and available scratch budget justify separate charge
grids; smaller jobs use a serial path. Start Julia with the desired thread
count, for example `julia --threads=4`. FFTW's thread count is configured
separately with `FFTW.set_num_threads(4)` before solving. Both settings affect
resource use, so measure them for your workload.

For CUDA, install and load CUDA.jl in your environment and use device arrays
throughout:

```julia
using SpaceCharge, CUDA, KernelAbstractions

@assert CUDA.functional()
CUDA.allowscalar(false)
x = CuArray(Float32[-0.001, 0.001])
y = CuArray(Float32[-0.0005, 0.0005])
z = CuArray(Float32[-0.002, 0.002])
charge = CuArray(Float32[0.5e-9, 0.5e-9])
sc = FFTIGF((32, 32, 32); T=Float32, backend=get_backend(x))
bounds = ((-0.004, -0.004, -0.004), (0.004, 0.004, 0.004))

solve!(sc, x, y, z, charge, 2; bounds)
Ex, Ey, Ez = similar(x), similar(y), similar(z)
field!(Ex, Ey, Ez, sc, x, y, z)
CUDA.synchronize()
host_fields = Array.((Ex, Ey, Ez))
```

CUDA is an optional dependency. GPU kernels use atomic additions for
deposition, and the FFTs use the GPU library loaded with the array backend.
Passing an arbitrary KernelAbstractions GPU backend alone does not ensure
that compatible real FFT plans are available. GPU work may be asynchronous;
synchronize at timing boundaries or when coordinating with host work. Keep
data on the device across repeated solves to avoid repeated transfers.

## Reuse, accuracy, and performance

FFT plans and buffers are specific to the selected field method and reused
for a solver's grid shape and precision. Both methods use the same FFT sizes
and physical crop; only `:potential_fd` allocates an additional physical-grid
potential buffer.
Changing the CPU FFTW thread count causes its FFT workspace to be rebuilt on
the next solve. Default `FFTW.MEASURE` planning makes the first solve different
from later calls; include a warm-up before measuring repeated-solve cost.

With `cache_green=true`, each solver retains one free-space spectrum set and,
when needed, one image set. Each set contains three field spectra for
`:direct` or one potential spectrum for `:potential_fd`. Reuse requires exactly
matching spacings, `gamma`, and kernel offset. Changing particle charges alone
permits reuse. Automatic bounds can change the spacing and therefore require the
spectra to be recomputed. Cathode spectra also depend on the domain's position
relative to the plane. The cache keeps the current geometry, not a history of
all geometries. `cache_green=false` trades repeated kernel transforms for
lower retained spectrum storage.

CPU pruning skips work whose results are unnecessary for the retained
convolution; it does not introduce another physical approximation. Use
`pruned_fft=false` to select full CPU transforms for comparison. Green-function
transforms remain full in both modes. Roundoff can differ between transform
paths, thread counts, and CPU/GPU deposition orders.

For `N` particles and `M = nx*ny*nz` physical nodes, deposition and evaluation
cost scale with their respective particle counts, while the FFT work scales
as `O(M log M)` with the doubled grid. Actual memory exceeds the physical
charge and three-component field arrays: the doubled real arrays, complex
spectra, optional caches, and CPU scratch also occupy space. Cathode mode adds
an image convolution. Benchmark representative grids, particle counts,
precision, and reuse patterns rather than assuming that a particular backend
or option is always faster.

Per real or image contribution, `:potential_fd` needs one kernel spectrum and
one inverse convolution transform instead of three. This reduces the number
of transforms and cached spectra, but adds a potential buffer and a
finite-difference pass. Actual speedup and memory use depend on the backend,
caching, and workload.

Accuracy depends on more than FFT precision:

- Cloud-in-cell deposition, uniform-cell averaging, and trilinear gathering
  smooth structure at the scale of the grid. The result is not a direct
  singular point-charge evaluation at very short distances. Check convergence
  by refining node counts at fixed physical bounds and by resolving the
  distribution with enough macroparticles.
- `:potential_fd` adds derivative truncation error: second order for smooth
  potentials along axes with at least three nodes, and first order at the
  endpoints of a two-node axis. `:direct` avoids this derivative approximation.
  The methods therefore need not agree on a fixed grid, especially near
  concentrated charges or poorly resolved structure. Compare accuracy at
  fixed bounds under grid refinement when choosing between them.
- The kernel uses analytic expressions with logarithms and arctangents, then
  subtracts corner values. Very unequal cell dimensions, especially after
  longitudinal stretching by a large `gamma`, can suffer cancellation or
  nonfinite values. `Float32` is more sensitive; compare against `Float64` and
  check all three components when evaluating demanding geometries.
- Shifted cathode kernels in `:direct` use the full antiderivative expression.
  Some alignments of shifted cell corners with coordinate planes can encounter
  singular intermediate expressions. Check finiteness and convergence when
  choosing cathode bounds and grid spacing, particularly close to the plane.
- Floating-point summation and GPU atomic ordering can change low-order bits.
  Numerical comparisons should use tolerances appropriate to the precision
  and field scale.

Runnable examples are in [examples/](../examples/), including CPU, CUDA,
cathode, and analytical comparisons. The repository tests also compare the
grid field with an independent integrated-kernel implementation and check the
deposition, interpolation, relativistic scaling, and cathode conventions.
