"""
    solve!(solver, x, y, z, charge, gamma; mask=nothing)

Prepare a frozen field solution from particles. Positions are
simultaneous lab-frame Cartesian coordinates in metres, with z along the longitudinal
direction. `charge` contains signed macroparticle charges in coulombs. The required
positional argument `gamma` must be finite and >= 1; pass `1` for stationary calculations.
Optional `mask` is a Boolean vector: true includes an entry, false skips it
before its coordinates or charge are read. Returns `nothing`.
"""
function solve! end

"""
    field!(Ex, Ey, Ez, solver, xt, yt, zt; mask=nothing)

Evaluate the last successfully prepared field, overwriting and returning
`(Ex, Ey, Ez)` in V/m. Inactive evaluation points receive zero without reading
their coordinates.
Active evaluation points must be finite and inside the solver's closed bounds.
"""
function field! end

@inline _is_included(::Nothing, i) = true
@inline _is_included(mask, i) = @inbounds mask[i]

function _check_arrays(owner, arrays::Tuple, mask)
    backend = get_backend(owner)
    for a in arrays
        a isa AbstractVector || throw(ArgumentError("expected a vector"))
        axes(a) == axes(first(arrays)) || throw(DimensionMismatch("array axes must match"))
        eltype(a) === eltype(owner) || throw(ArgumentError("array precision must match the solver"))
        typeof(get_backend(a)) === typeof(backend) || throw(ArgumentError("array backend must match the solver"))
    end
    if mask !== nothing
        mask isa AbstractVector{Bool} || throw(ArgumentError("mask must be a Boolean vector"))
        axes(mask) == axes(first(arrays)) || throw(DimensionMismatch("mask axes must match"))
        # KernelAbstractions handles ordinary views, but has no BitArray backend.
        mask_backend = mask isa BitArray || parent(mask) isa BitArray ? CPU() : get_backend(mask)
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
