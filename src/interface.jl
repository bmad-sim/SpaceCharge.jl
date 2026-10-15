"""
    solve!(solver, x, y, z, charge; gamma=1, active=nothing)

Prepare a frozen source and return `solver`. Positions are simultaneous lab-frame
Cartesian coordinates in metres, with z along the common beam velocity. `charge`
contains signed macroparticle charges in coulombs. `gamma` must be finite and >= 1.
Inputs are borrowed only during preparation. A failed solve invalidates the field.
Omitting `gamma` means 1 on every call. Empty, all-inactive and all-zero-charge
sources prepare a valid zero field; zero signed net charge alone is not empty.

All arrays must be one-based vectors of the solver's Float32/Float64 precision on
its backend. Optional `active` is a Boolean vector on that backend; inactive
entries are skipped before their coordinates or charges are read.
"""
function solve! end

"""
    field!(Ex, Ey, Ez, solver, xt, yt, zt; active=nothing)

Evaluate the last successfully prepared source, overwriting and returning
`(Ex, Ey, Ez)` in V/m (lab E). Targets may differ from the sources. Inactive targets
receive zero without reading their coordinates. No source preparation is performed.
Outputs must not overlap one another, targets, or solver storage. Calls before a
successful solve or after a failed refresh must throw. Source and target batches
may have different lengths; arrays within each batch must have equal axes.

Concurrent evaluations require disjoint outputs. Do not prepare the same solver
concurrently with evaluation. GPU work is ordered on the calling stream; synchronize
before host access or cross-stream use. Device data errors can surface at synchronization.
"""
function field! end

@inline _active(::Nothing, i) = true
@inline _active(active, i) = @inbounds active[i]

# Array wrappers are unwrapped only for placement checks, never for computation.
_storage(a) = parent(a) === a ? a : _storage(parent(a))
_array_backend(a) = get_backend(a)
_array_backend(::BitArray) = CPU()
_same_device(a, b) = true # CUDA specializes this without becoming a core dependency.

@inline function _check_array(a, owner, ::Type{T}, ax) where T
    a isa AbstractVector || throw(ArgumentError("expected a one-dimensional array"))
    eltype(a) === T || throw(ArgumentError("array element type must match $T"))
    axes(a) == ax || throw(DimensionMismatch("array axes must match"))
    Base.require_one_based_indexing(a)
    storage = _storage(a)
    typeof(_array_backend(storage)) === typeof(get_backend(owner)) &&
        _same_device(storage, owner) ||
        throw(ArgumentError("arrays must reside on the solver's backend and device"))
    return nothing
end

@inline function _check_arrays(owner, arrays::Tuple, active)
    first(arrays) isa AbstractVector || throw(ArgumentError("expected a one-dimensional array"))
    ax = axes(first(arrays))
    for a in arrays
        _check_array(a, owner, eltype(owner), ax)
    end
    active === nothing || _check_array(active, owner, Bool, ax)
    return nothing
end

function _gamma(::Type{T}, gamma) where T
    gamma isa Real && isfinite(gamma) && gamma >= 1 ||
        throw(ArgumentError("gamma must be finite and >= 1"))
    g = T(gamma)
    isfinite(g) && g >= one(T) || throw(ArgumentError("gamma is not representable in $T"))
    return g
end
