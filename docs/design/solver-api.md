# Space-charge solver API: initial migration

Status: design proposal for local `dev`; no solver refactor or new algorithm is implemented by this document.

Baseline: SpaceCharge v1.3.0, `ae30463305729aec2f04d4687ffd547b94750cb2`.

## Purpose and scope

Define the shared solver API and adapt the existing FFT integrated-Green-function solver to it. This is the entirety of Ningdong's initial implementation scope. Performance is the primary design criterion. Callers supply ordinary arrays; each concrete solver owns its reusable state. Tracking code decides when to prepare a source, where to evaluate it, and how to apply the result.

This is the initial architecture move, not a general simulation framework. It introduces no tracking owner, Bunch dependency, dimensional solver hierarchy, or universal deposition pipeline. John can later add Bassetti–Erskine (BE) and Barnes–Hut (BH) as concrete solvers implementing the shared contract. Their internals, numerical choices, constructors and tests are outside this initial work; no speculative scaffolding or decisions about them are needed to start.

## Accepted decisions and proposals

Accepted decisions:

- Public operation names are `solve!` and `field!`.
- Initial common output is electric field only.
- Inputs are ordinary x, y, z, charge arrays; outputs are preallocated Ex, Ey, Ez arrays. No input Bunch or ParticleSource struct is required.
- Support suitable views and array backends without forced copies or implicit CPU/GPU adaptation.
- Each physical solver has a concrete struct and reusable internal storage.
- Tracking owns scheduling and integration. No public `update!`, `electric_field!`, `field_view`, or `field_at` is introduced.
- Initial implementation is the shared API and existing FFT solver only; BE/BH are future contributions by John.
- Backward compatibility is not required. The new public API may break existing callers; compatibility wrappers, deprecated aliases and old return-value behavior are not requirements.
- Preserve useful staged deposition, solving and gathering for performance, with signatures designed for the new solver API.

Proposals specified below for implementation review:

- FFT solver constructor, exact return values and optional masks.
- Physical beam-aligned lab positions and lab E, with one common gamma per solve, matching existing FFT semantics.
- Separate the prepared-source lifetime from allocation/cache lifetime, with explicit failure and concurrency rules.
- Begin with the existing FFT behavior and retain its default discretization. Future solver contributions use the shared contract without being part of this migration.

## Common interface

```julia
# Construction allocates reusable storage; does not produce a valid field.
sc = FFTIGF((32, 32, 32); T=Float64, backend=CPU(),
            bounds=nothing, fft_flags=FFTW.MEASURE,
            cache_green=true, pruned_fft=true)

solve!(sc, x, y, z, charge; gamma=1, active=nothing)
field!(Ex, Ey, Ez, sc, xt, yt, zt; active=nothing)
```

`solve!` refreshes all source-dependent state needed by that algorithm and returns `sc`. FFTIGF deposits and convolves; future solver methods perform their own preparation. It does not allocate or return per-particle output fields.

`field!` evaluates the last successfully prepared solution at the supplied targets, overwrites the three outputs, and returns `(Ex, Ey, Ez)`. It never implicitly refits, redeposits, rebuilds, or calls `solve!`. A call before a successful solve, or after a failed solve, throws a clear error.

Require finite gamma >= 1, representable in the solver precision. Validate it before publishing a prepared solution. Keep `gamma` on each solve rather than an implicit reference momentum in the source arrays. Calling solve without the keyword means gamma = 1, not reuse of a previous gamma.

Constructor settings affecting physical discretization, bounds and evaluation remain algorithm-specific. Do not accept arbitrary unused keywords to make constructors appear uniform. Initial FFT boundaries are free space and the existing cathode image convention; `at_cathode=false` may be retained as a solver option. Unsupported boundary combinations must fail. Future solvers document the physical models they implement.

## Array layout and source normalization

All source arrays are one-dimensional and length N. Index i refers to the same source particle in x, y, z and charge. All target arrays and outputs are one-dimensional and length M, which may differ from N.

- x, y, z and xt, yt, zt are physical Cartesian positions in metres.
- charge[i] is signed macroparticle charge in coulombs, including any statistical weight. It is not species charge alone, current, probability, or a density.
- The solver must not renormalize surviving particles or multiply by a second total-charge parameter.
- Outputs are Ex, Ey, Ez in V/m, evaluated in target order. They are not force, kick, or rigidity-normalized fields.
- Initially require compatible Float32 or Float64 coordinate/charge/output storage and matching solver precision. Reject unsupported types or conversions explicitly rather than silently converting or copying arrays.
- Require compatible device/backend placement, equal axes within each array group, and the initial ordinary one-based indexing convention. Different groups may have different lengths.
- Dense vectors and ordinary column views are primary fast paths. Other strided views may be supported when the backend permits them; support must not imply equal performance.
- Output arrays may not overlap one another, target inputs, or prepared solver storage. Aliasing is a documented precondition; the API does not promise expensive complete overlap detection on every call.

For equal macroparticle charges, a reusable charge vector is sufficient initially. Scalar-charge overloads or lazy constant arrays are optional later optimizations, not reasons to allocate a new vector each solve.

BeamTracking stores an N×6 matrix in SOA form with columns [x, px, y, py, z, pz]. Dense Julia matrices have unit particle stride within a column and a column stride of N. Reusing its component-array convention does not require storing six coordinates here. Its canonical longitudinal coordinate must be converted to the physical source position by the caller; blindly passing columns 1, 3, 5 is not generally valid. Its `q` field is spin quaternions and its `weight` field is not universally defined as signed coulombs.

```julia
# positions is an N×3 matrix of physical positions, already in the agreed chart.
x = view(positions, :, 1)
y = view(positions, :, 2)
z = view(positions, :, 3)

# Existing physical coordinate arrays can be passed directly as well.
solve!(sc, x, y, z, charge; gamma=gamma)

# Output column views avoid packing/unpacking copies.
Ex = view(fields, :, 1)
Ey = view(fields, :, 2)
Ez = view(fields, :, 3)
field!(Ex, Ey, Ez, sc, x, y, z)
```

View wrappers should be constructed/reused outside hot loops where useful. Do not require an N×3 or N×6 materialization when separate components already exist.

## Inactive sources and targets

`active=nothing` means all entries are active. Otherwise it is a same-backend length-N Boolean source mask for solve, or length-M Boolean target mask for field. This is independent of any tracking package's state constants. A caller with UInt8 particle states may populate a reusable Boolean mask or use an explicitly supported specialized overload later.

Inactive sources contribute nothing. Their positions may be nonfinite and must not be used for extrema, moments, deposition indices, or tree building. Active zero-charge particles contribute no source field but remain valid targets. Empty/all-inactive/all-zero-charge sources produce a valid zero-field solution; signed net charge zero alone does not justify this shortcut.

Inactive targets receive zero output and are not geometrically evaluated. Empty target batches succeed without a kernel launch. Active source/target data must meet the numerical model's finiteness/domain requirements.

## Frame and E-only output

The initial proposed common model is co-moving electrostatic space charge with one common beam velocity. Coordinates describe a simultaneous lab snapshot in a Cartesian chart whose z axis is aligned with beam motion. FFTIGF returns physical lab E expressed in that chart; this defines the output convention for future implementations of the shared interface.

The existing FFT stretches longitudinal separations by gamma inside its kernel, computes the rest-frame electrostatic interaction, and boosts transverse E. Preserve this once-only transformation: callers provide lab z, not already stretched rest z. Conceptually, E_x = gamma E'_x, E_y = gamma E'_y, E_z = E'_z, with rest-frame longitudinal separations stretched by gamma. The prepared solution retains the gamma used for its solve.

For this model, external code can obtain B from known common velocity:

```text
B = beta × E / c
```

This is a cross product, not a scalar rescaling. For motion along +z, Bx = -beta_z Ey/c, By = beta_z Ex/c, Bz = 0. Gamma alone does not specify the signed direction. This relation applies to the common-velocity free-space model; retaining the existing cathode image E convention does not establish a general magnetic boundary model. SpaceCharge does not apply the target species charge or magnetic cancellation to its returned E.

Future contributions must document how their native result satisfies these units and frame conventions. Their physical models and internal algorithms are not specified here.

## Concrete solver storage

The following are schematic implementation shapes, not public field-layout guarantees:

```julia
mutable struct FFTIGF{M,O}
    mesh::M             # concrete Mesh3D; existing lazy FFT/deposition storage
    options::O          # concrete settings
    prepared::Bool
end

```

No separate algorithm/config/workspace hierarchy is required. Concrete named tuples or small option structs are sufficient. The FFT constructor can allocate its known mesh shape with temporary internal geometry and mark it unprepared until a solve establishes physical geometry; avoid an abstract `mesh` field merely to defer bounds.

Host solver objects need not themselves be GPU kernel arguments. Implementation-private immutable kernel payloads can contain only the required device arrays and numeric parameters. No public prepared-view API is necessary.

## Preparation, reuse and lifetime

FFTIGF's solve chooses or updates geometry, deposits charge and convolves the integrated field kernels. Its field! gathers prepared grid E. Other solver types can provide their own preparation and evaluation methods without sharing these internal stages.

The solver borrows caller inputs only for the duration of solve. After successful preparation, field evaluation must not depend on later mutation of those input arrays. FFTIGF retains its grid solution. Future implementations must likewise retain whatever prepared state is required to satisfy this contract; the interface does not mandate how they do so.

A new solve replaces the previous solution. Set validity false before modifying source-dependent state, and publish validity only after successful preparation has been scheduled/completed according to backend ordering. A failed refresh must not expose a partially overwritten state as valid. With asynchronous GPU work, later execution failures may surface at synchronization; no rollback is promised.

Repeated field calls may use different targets without preparing again. This evaluates a frozen source at each target set. A caller may alternatively hold previously computed per-particle field arrays while particles move; that is the caller's separate numerical choice.

Allocation reuse does not imply physical-state reuse:

- Plans/buffers require compatible backend, precision, shape/layout and planning options.
- Green spectra depend on kernel/discretization, spacing, gamma and image displacement; array shape/backend/precision also determine their owning workspace.
- Deposited charges depend on positions, charge, source mask, geometry and deposition rule.

Do not infer unchanged values from array identity. Preserve exact kernel-cache invalidation initially. Geometry snapping or approximate kernel reuse is a separate explicit numerical option with accuracy tests.

## Algorithm-specific expert workflows

Expose useful stages for selective recomputation under the new solver API. Existing numerical routines may supply their implementation, but their old public signatures do not need to remain available.

For FFTIGF, proposed staged operations expose source reuse without reaching into private workspace fields:

```julia
deposit!(sc::FFTIGF, x, y, z, charge; active=nothing, clear=true)
solve!(sc::FFTIGF; gamma=gamma)  # solve the existing deposited source
field!(Ex, Ey, Ez, sc, xt, yt, zt)
```

Depositing through the solver invalidates its prepared field. The staged solve requires initialized geometry and deposited source; it does not silently move the grid. Changing gamma can reuse deposited lab charges but invalidates the appropriate kernel/solution. Geometry changes require redeposition. The combined source-array solve owns adaptive geometry selection and uses these same numerical kernels.

Future solvers may expose their own useful expert stages alongside the shared solve!/field! methods. No common deposition stage or additional preparation interface is imposed on them. This initial work implements staged reuse only for FFTIGF.

The staged operations are supported for numerical reuse, not as compatibility shims. Direct mutation of a solver's private storage/cache fields is outside the new validity contract.

## Backend, allocation and concurrency contract

- CPU Float32/Float64 are initial targets. CUDA support is claimed per solver only after actual execution tests; KernelAbstractions alone does not establish support for every GPU backend.
- Do not claim Dual/TPSA or general AD support. Fixed-source analytical differentiation can be a separately tested later extension.
- Setup/first use may allocate and compile/plan. Stable-size warm solves must reuse particle/grid storage; field evaluation must not allocate output, copy whole inputs or repeat source preparation.
- Storage-size changes belong to explicit setup or preparation. Reserve the FFT grid and expected deposition scratch before measured hot loops; do not hide storage growth in field evaluation.
- Keep orchestration concretely specialized and dispatch outside particle loops. Preserve the existing FFT function barriers before attempting unrelated cleanup.
- Do not mandate an additional source pass or host reduction solely because of the common API. Fuse validation/reductions where appropriate; validate fixed layout/backend properties at setup.
- Prepared data is read-only during field evaluation. Concurrent calls require disjoint outputs and independent evaluation scratch where needed.
- Do not run solve concurrently with evaluation on the same solver. Use separate solver instances for independent concurrent preparations.
- Establish backend stream/event ordering between preparation and evaluation. Avoid unconditional host synchronization at every stage; callers synchronize before host access or independent-stream use as required.
- FFTW planning/thread configuration must be handled deliberately because existing thread settings are global. One solver per task does not itself make global planning configuration independent.

The baseline currently records small host allocations in warm FFT calls; do not claim the entire baseline pipeline is already zero-allocation. The initial API must add no particle/grid-sized allocation and no material overhead. Measure and account for small host allocations separately.

## API transition and module organization

The new API may replace or remove Mesh3D constructors and mesh-based public operations, including clear_mesh!, solve!(mesh), interpolate_field and interpolate_field!. No compatibility wrappers, deprecation period or preservation of their return values is required. Callers will adopt the new interface. solve!(solver, arrays...) consistently returns its solver; field! returns the three output arrays.

Reuse the existing CIC, Green-function, FFT, gather and cache machinery where it remains useful. Their internal types, ownership and file placement may change to fit concrete solver state. Preserve the intended FFT discretization and physical normalization during this architecture move; use the released implementation as a numerical/performance baseline rather than an API compatibility target.

The canonical release does not contain the potential–FD solver. Its separate feature branch is an extension example, not a prerequisite for this migration. When integrated, it can share FFT infrastructure and deposition/gather while retaining a distinct physical algorithm/discretization selection and separate scalar/vector caches.

Proposed minimal source organization:

```text
src/
  SpaceCharge.jl          # includes and exports
  interface.jl            # shared documented entry points, cheap validation
  mesh.jl                 # reusable mesh storage; no old public API guarantee
  deposition.jl           # reusable CIC implementation
  interpolation.jl        # reusable gather implementation
  green_functions.jl      # existing integrated kernels
  fft/transforms.jl       # transform infrastructure, including pruning
  solvers/free_space.jl   # existing FFT numerics; consolidate if useful
  solvers/fftigf.jl       # new concrete solver and array entry points
```

This tree is a starting point, not a requirement to retain existing file boundaries. Implement the common interface and concrete FFT solver using the useful numerical internals; no transitional public wrapper is required. Avoid unrelated numerical rewrites. Do not add dependencies on BeamTracking, Beamlines or SciBmad to core SpaceCharge.

## Caller and implementer examples

```julia
# Once: reuse ordinary arrays and solver storage.
N = length(x)
sc = FFTIGF((32,32,32); T=eltype(x), backend=get_backend(x))
Ex, Ey, Ez = similar(x), similar(x), similar(x)

# Whenever the caller chooses to refresh collective physics:
solve!(sc, x, y, z, charge; gamma=gamma)
field!(Ex, Ey, Ez, sc, x, y, z)

# Evaluate the same prepared source at a different set of points:
field!(probe_Ex, probe_Ey, probe_Ez, sc, probe_x, probe_y, probe_z)
```

Implementer sketch for the FFT adaptation (comments describe existing internal work, not additional public stages):

```julia
function solve!(sc::FFTIGF, x, y, z, charge;
                gamma=1, active=nothing)
    sc.prepared = false
    # Validate source/gamma, establish mesh geometry and deposit charge.
    # Reuse FFT plans and correctly keyed kernels; prepare grid E.
    sc.prepared = true
    return sc
end

function field!(Ex, Ey, Ez, sc::FFTIGF, xt, yt, zt;
                active=nothing)
    # Require valid state; validate target/output structure.
    # Gather the prepared grid field into caller-owned output arrays.
    # Write all active outputs and zero inactive outputs.
    return Ex, Ey, Ez
end
```

Future solver authors implement these same two entry points on their own concrete types. No shared implementation must allocate a universal intermediate representation, and this initial work adds no placeholder BE/BH types or files.

## Initial migration and contribution boundary

1. Finalize the shared array/field contract and FFT constructor; add focused interface tests.
2. Ningdong implements the concrete FFTIGF solver and combined solve/field plus staged FFT reuse, adapting useful existing kernels and storage. Remove or replace obsolete public entry points as appropriate. Validate numerical results and performance against v1.3.0; preserving its calling signatures is not a gate.
3. Validate the supported FFT CPU/backend/precision paths and steady-state performance. The separate potential-FD worktree is not modified or required for this initial move.

Ningdong owns this shared API and FFT adaptation. Once that contract is ready, John can branch from dev and contribute BE/BH implementations separately. Their internals, numerical choices and algorithm-specific validation belong to those future contributions and do not block this work. Caller/tracker integration remains independent of this solver-focused change.

## Acceptance tests and performance gates

Contract tests:

- Combined solve/field agrees with the new explicit preparation/evaluation stages. For unchanged FFT discretization and identical geometry/source, compare numerical fields with v1.3.0 reference outputs and independent physics references; old public calls need not remain executable in the new package.
- Tests target the new API. Remove or rewrite tests whose sole purpose is preserving old constructors, exports, signatures or return values; retain their numerical coverage where useful.
- Units, signed-charge linearity, common-gamma scaling and translation invariance are correct within each model's assumptions.
- Repeated field calls at different targets do not repeat preparation.
- Mutation of original source arrays after solve does not change the frozen prepared result.
- Source/output shape/backend/precision errors, unprepared use, failed solve, empty data and masks have defined behavior.
- Inactive nonfinite coordinates are ignored safely; active nonfinite data is rejected. Zero net charge is not mistaken for zero source.
- FFT target points must occupy valid interpolation cells, with a defined upper-face/out-of-domain policy. Do not carry unsafe unchecked indexing into a new public convenience path.
- Kernel/cache invalidation covers gamma, geometry, boundaries, array replacement and planning settings. Existing charge reuse and output overwriting remain correct.
- Separate solver instances are independent; concurrent evaluation is tested only for the scratch/stream arrangements actually supported.

Performance gates:

- Compare the new FFT route against unchanged v1.3.0 on identical workloads: stable geometry/cache hits and changing geometry/gamma.
- Measure setup separately from warm solve and field; do not include compilation/planning in steady-state comparisons.
- No new O(N) or O(grid) allocations in stable-size warm FFT paths, no implicit device transfers, and no new preparation work during repeated field evaluation.
- Preserve concrete specialization; inspect representative inference/kernel code when a new abstraction crosses a hot boundary.
- Use synchronized GPU timings at benchmark boundaries, actual device execution and Float32/Float64 checks for each supported solver.
- Set material-regression thresholds from observed noise on target hardware. Compare numerical algorithms at matched accuracy as well as equal storage size.

This design does not authorize a new benchmark campaign by itself. Reuse existing baseline cases when implementation begins and broaden only for new functionality or unresolved regressions.

## FFT details to settle during implementation

Specify the safe bounds/domain policy for the new FFT entry point, including upper faces, out-of-domain targets, degenerate extents and Float32 representability. Unsafe or inconsistent old behavior need not be retained. These choices concern the existing FFT adaptation and its tests; no BE/BH internal design is part of this work.

## Local source basis

- `src/SpaceCharge.jl:19`: released exports.
- `src/mesh.jl:27,65`: Mesh3D ownership, workspace and invalidation.
- `src/deposition.jl:61,290`: nodal charge deposition and public array signature.
- `src/green_functions.jl:73,124`: longitudinal stretch and transverse lab-field scaling.
- `src/interpolation.jl:114`: preallocated three-component gather.
- `src/solvers/free_space.jl:20`: released solver and return behavior.
- `examples/basic_cpu_usage.jl:8`: released four-vector particle workflow.
- BeamTracking `src/types.jl:1,24,67` and `src/kernel.jl:249` at `1a416c058674c705b302f1e4e3c0fd3cc7100315`: coordinate columns, state/weight meanings and SOA convention.
