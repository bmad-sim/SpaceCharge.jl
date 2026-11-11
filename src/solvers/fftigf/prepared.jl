# Holds the resources selected at the checked setup boundary. This is an
# execution handle, not another solver or a copy of the physical state.
struct PreparedFFTIGF{S,W,D}
    solver::S
    workspace::W
    deposit_workspace::D
    deposit_chunks::Int
end

"""
    prepare!(solver::FFTIGF, x, y, z, charge; mask=nothing)

Validate particle array axes, precision, backend, and the initial fitted
geometry, then allocate FFT plans, Green-spectrum storage, and CPU deposition
scratch for this particle count. Return a handle for solve_prepared! and
field_prepared!. Empty/all-inactive batches are allowed at setup. Inactive
coordinates are not read. No physical solve is performed or field changed.

Initial active coordinates must be finite and their fitted bounds/spacings
representable. The caller is responsible for finite active charges and for
preserving the physical input contracts during later prepared execution.

The handle shares its solver's mutable state and must not be used concurrently
with that solver or another handle. Keep the solver's grid, arrays, options,
backend, precision, and FFT thread settings unchanged while using it. Prepare
again when any of those or the particle-array layout/count changes.
"""
function prepare!(sc::FFTIGF, x, y, z, charge; mask=nothing)
    _check_arrays(sc.rho, (x, y, z, charge), mask)
    if !isempty(x) && (mask === nothing || any(mask))
        bounds = _grid_bounds_from_particles(sc, x, y, z, mask)
        _checked_grid_geometry(sc, bounds)
    end
    w = _get_workspace(sc; fft_flags=sc.options.fft_flags,
                       pruned_fft=sc.options.pruned_fft)
    if sc.options.cache_green
        if w.free_cache.spectra === nothing
            w.free_cache.spectra = map(_ -> similar(w.product), w.components)
        end
        if sc.options.at_cathode && w.image_cache.spectra === nothing
            w.image_cache.spectra = map(_ -> similar(w.product), w.components)
        end
    end
    # Zero chunks denotes GPU deposition; CPU uses serial or private-grid
    # deposition exactly as the ordinary solve! path.
    chunks = if get_backend(sc.rho) isa CPU
        sc.rho isa Array && isbitstype(eltype(sc.rho)) ?
            _deposit_chunk_count(length(x), length(sc.rho), sizeof(eltype(sc.rho)),
                                 Threads.nthreads()) : 1
    else
        0
    end
    scratch = chunks > 1 ? _get_deposit_workspace!(sc, chunks) : nothing
    return PreparedFFTIGF(sc, w, scratch, chunks)
end

"""
    solve_prepared!(prepared, x, y, z, charge, gamma; mask=nothing, bounds=nothing)

Execute a particle solve using resources from prepare!, without input
validation or plan/workspace selection. Refit the active-particle bounds on
every call unless bounds=(lo, hi) is supplied. Gamma and the Green spectra
remain dynamic; preparation does not freeze the physical solution.

The caller must preserve the prepared array axes, precision, backend, and
particle count, and supply finite active coordinates/charges, at least one
active particle, and finite representable gamma >= 1. Fitted bounds must remain
finite and their spacings finite, positive, and representable. Explicit bounds must
have three finite lower/upper coordinates, positive representable spacings,
and contain every active particle. Masked coordinates and charges are not read.
Use ordinary solve! when the caller cannot maintain these contracts.
Returns nothing.
"""
function solve_prepared!(prepared::PreparedFFTIGF, x, y, z, charge, gamma;
                         mask=nothing, bounds=nothing)
    sc = prepared.solver
    T = eltype(sc.rho)
    lo, hi = if bounds === nothing
        particle_bounds = _particle_bounds(x, y, z, mask)
        _padded_grid_bounds(sc, particle_bounds.lo, particle_bounds.hi)
    else
        lower, upper = bounds
        Tuple(T.(lower)), Tuple(T.(upper))
    end
    sc.min_bounds, sc.max_bounds = lo, hi
    sc.delta = (hi .- lo) ./ (sc.grid_size .- 1)
    fill!(sc.rho, zero(T))
    if prepared.deposit_chunks == 0
        deposit_gpu!(sc, x, y, z, charge, mask)
    elseif prepared.deposit_chunks == 1
        deposit_vectorized!(sc, x, y, z, charge, mask)
    else
        _deposit_threaded!(sc, x, y, z, charge, prepared.deposit_workspace,
                           prepared.deposit_chunks, mask)
    end
    _solve_mesh!(sc, prepared.workspace; gamma=T(gamma))
    return nothing
end

"""
    field_prepared!(Ex, Ey, Ez, prepared, x, y, z; mask=nothing)

Gather the last field without input validation, using the same interpolation
kernel as field!. The coordinate/output vectors and optional Boolean mask
must have matching axes and the prepared solver's precision and backend.
Active coordinates must be finite and inside the current closed solver bounds.
Inactive outputs are zeroed without reading their coordinates.
Returns (Ex, Ey, Ez).
"""
function field_prepared!(Ex, Ey, Ez, prepared::PreparedFFTIGF, x, y, z;
                         mask=nothing)
    return _field!(Ex, Ey, Ez, prepared.solver, x, y, z, mask)
end
