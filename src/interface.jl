"""
    solve!(solver, x, y, z, charge; gamma=1, active=nothing)

Prepare a frozen field solution from particles and return `solver`. Positions are
simultaneous lab-frame Cartesian coordinates in metres, with z along the longitudinal
direction. `charge` contains signed macroparticle charges in coulombs. `gamma` must be finite and >= 1.
Optional `active` is a Boolean vector; inactive entries are skipped 
before their coordinates or charges are read.
"""
function solve! end

"""
    field!(Ex, Ey, Ez, solver, xt, yt, zt; active=nothing)

Evaluate the last successfully prepared field, overwriting and returning
`(Ex, Ey, Ez)` in V/m. Inactive evaluation points receive zero without reading
their coordinates.
"""
function field! end

@inline _active(::Nothing, i) = true
@inline _active(active, i) = @inbounds active[i]

function _check_arrays(owner, arrays::Tuple, active)
    backend = get_backend(owner)
    for a in arrays
        a isa AbstractVector || throw(ArgumentError("expected a vector"))
        axes(a) == axes(first(arrays)) || throw(DimensionMismatch("array axes must match"))
        eltype(a) === eltype(owner) || throw(ArgumentError("array precision must match the solver"))
        typeof(get_backend(a)) === typeof(backend) || throw(ArgumentError("array backend must match the solver"))
    end
    if active !== nothing
        active isa AbstractVector{Bool} || throw(ArgumentError("active must be a Boolean vector"))
        axes(active) == axes(first(arrays)) || throw(DimensionMismatch("mask axes must match"))
        # KernelAbstractions handles ordinary views, but has no BitArray backend.
        mask_backend = active isa BitArray || parent(active) isa BitArray ? CPU() : get_backend(active)
        typeof(mask_backend) === typeof(backend) || throw(ArgumentError("mask backend must match the solver"))
    end
    return nothing
end

function _gamma(::Type{T}, gamma) where T
    gamma isa Real && isfinite(gamma) && gamma >= 1 ||
        throw(ArgumentError("gamma must be finite and >= 1"))
    g = T(gamma)
    isfinite(g) || throw(ArgumentError("gamma is not representable in $T"))
    return g
end
