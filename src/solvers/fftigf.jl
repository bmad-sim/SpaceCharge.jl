# Small reusable reduction storage for adaptive GPU geometry. No particle-sized
# packing, masks, or temporary arrays are constructed by preparation.
struct SourceExtent{T}
    lo::NTuple{3,T}
    hi::NTuple{3,T}
    valid::Bool
end
_empty_extent(::Type{T}) where T = SourceExtent(ntuple(_ -> T(Inf), 3), ntuple(_ -> T(-Inf), 3), true)
@inline function _merge_extent(a::SourceExtent, b::SourceExtent)
    SourceExtent(min.(a.lo, b.lo), max.(a.hi, b.hi), a.valid & b.valid)
end
@inline function _source_extent(x, y, z, q, active, i)
    T = eltype(x)
    if _active(active, i)
        @inbounds p = (x[i], y[i], z[i])
        @inbounds valid = all(isfinite, p) && isfinite(q[i])
        return SourceExtent(p, p, valid)
    end
    return _empty_extent(T)
end

@kernel function _extent_kernel!(out, x, y, z, q, active, stride)
    tid = @index(Local, Linear)
    group = @index(Group, Linear)
    i = @index(Global, Linear)
    partial = _empty_extent(eltype(x))
    for j in i:stride:length(x)
        partial = _merge_extent(partial, _source_extent(x, y, z, q, active, j))
    end
    shared = @localmem eltype(out) (256,)
    shared[tid] = partial
    @synchronize
    for offset in (128, 64, 32, 16, 8, 4, 2, 1)
        if tid <= offset
            shared[tid] = _merge_extent(shared[tid], shared[tid + offset])
        end
        @synchronize
    end
    if tid == 1
        out[group] = shared[1]
    end
end

"""
    FFTIGF(grid_size; T=Float64, backend=CPU(), bounds=nothing,
           fft_flags=FFTW.MEASURE, cache_green=true, pruned_fft=true,
           at_cathode=false)

Reusable integrated-field Green-function FFT solver. `bounds=(lo, hi)` specifies
a fixed closed box of grid nodes in metres; otherwise each combined `solve!` fits
the active sources. All dimensions need at least two nodes. Upper faces use the
last interpolation cell; active out-of-box sources and targets are errors.

Adaptive nondegenerate extents retain the released 1e-6-cell padding (at least one
representable step). A collapsed axis is expanded symmetrically to a nominal
1 micrometre cell, or enough to resolve nodes at that coordinate's precision.
Unrepresentable spacings are rejected. Use explicit bounds for controlled physical
resolution and for probes outside the source extent. Adaptive GPU preparation
reads a bounded extrema summary back to the host; fixed bounds avoid that reduction
and synchronization.

`at_cathode=true` retains the image-charge E convention at z=0 and requires
nonnegative z bounds/sources. FFT plans, spectra and CPU deposition scratch are
allocated lazily during preparation; warm up with the expected batch size before
timing. The package does not change FFTW's global thread count: configure it before
planning, and do not change it concurrently with solves. GPU always uses full FFTs.
"""
mutable struct FFTIGF{M,O,R,H}
    mesh::M
    options::O
    extent_device::R
    extent_host::H
    geometry::Bool
    deposited::Bool
    prepared::Bool
    empty_source::Bool
end

function Base.show(io::IO, sc::FFTIGF)
    print(io, "FFTIGF{", eltype(sc.mesh.rho), "}(", sc.mesh.grid_size,
          ", ", get_backend(sc.mesh.rho), ", prepared=", sc.prepared, ")")
end

function _geometry(n, lo, hi, ::Type{T}) where T
    length(lo) == length(hi) == 3 || throw(ArgumentError("bounds need three coordinates"))
    l, h = Tuple(T.(lo)), Tuple(T.(hi))
    d = (h .- l) ./ (n .- 1)
    for k in 1:3
        isfinite(l[k]) && isfinite(h[k]) && h[k] > l[k] &&
            isfinite(d[k]) && isfinite(inv(d[k])) && d[k] > 0 &&
            d[k] >= max(eps(l[k]), eps(h[k])) ||
            throw(ArgumentError("bounds must give finite, distinct grid nodes in $T"))
    end
    return l, h, d
end

function FFTIGF(n::NTuple{3,Int}; T=Float64, backend=CPU(), bounds=nothing,
                fft_flags=FFTW.MEASURE, cache_green::Bool=true,
                pruned_fft::Bool=true, at_cathode::Bool=false)
    T in (Float32, Float64) || throw(ArgumentError("FFTIGF supports Float32 and Float64"))
    all(k -> k >= 2, n) || throw(ArgumentError("grid dimensions must be at least 2"))
    backend isa Union{CPU,GPU} || throw(ArgumentError("unsupported FFT backend"))
    lo, hi, delta = bounds === nothing ? _geometry(n, (0,0,0), (1,1,1), T) :
        _geometry(n, bounds[1], bounds[2], T)
    at_cathode && lo[3] < 0 && throw(ArgumentError("cathode bounds require z >= 0"))
    mesh = Mesh3D(n, lo, hi, backend)
    options = (;adaptive=bounds === nothing, fft_flags, cache_green, pruned_fft, at_cathode)
    device = backend isa CPU ? nothing : KernelAbstractions.allocate(backend, SourceExtent{T}, 128)
    host = backend isa CPU ? nothing : Vector{SourceExtent{T}}(undef, 128)
    return FFTIGF(mesh, options, device, host, bounds !== nothing, false, false, false)
end

function _extent(sc, x, y, z, q, active)
    result = _empty_extent(eltype(x))
    isempty(x) && return result
    if get_backend(sc.mesh.rho) isa CPU
        @inbounds for i in eachindex(x)
            result = _merge_extent(result, _source_extent(x, y, z, q, active, i))
        end
    else
        groups = min(128, cld(length(x), 1024))
        _extent_kernel!(get_backend(sc.mesh.rho), 256)(sc.extent_device, x, y, z, q,
            active, 256groups; ndrange=256groups)
        # Only the bounded summary crosses to the host, which must establish
        # physical geometry before depositing. Fixed bounds avoid this reduction.
        copyto!(sc.extent_host, 1, sc.extent_device, 1, groups)
        for i in 1:groups
            result = _merge_extent(result, sc.extent_host[i])
        end
    end
    result.valid || throw(ArgumentError("active source coordinates and charges must be finite"))
    return result
end

function _fit_geometry!(sc, ext::SourceExtent{T}) where T
    n = sc.mesh.grid_size
    lo, hi = ext.lo, ext.hi
    sc.options.at_cathode && lo[3] < 0 && throw(ArgumentError("cathode sources require z >= 0"))
    pairs = ntuple(3) do k
        l, h = lo[k], hi[k]
        if l == h
            # Enough precision to represent every node, even far from the origin.
            step = max(T(1e-6), T(4) * eps(l))
            halfspan = step * T(n[k] - 1) / 2
            (l - halfspan, h + halfspan)
        else
            # Evaluate the released padding in Float64, then ensure it survives
            # rounding to Float32. No snapping or approximate cache keys.
            pad = 1e-6 * (Float64(h) - Float64(l)) / (n[k] - 1)
            (min(T(l - pad), prevfloat(l)), max(T(h + pad), nextfloat(h)))
        end
    end
    l, h = first.(pairs), last.(pairs)
    if sc.options.at_cathode && l[3] < 0
        l = (l[1], l[2], zero(T))
    end
    l, h, d = _geometry(n, l, h, T)
    sc.mesh.min_bounds, sc.mesh.max_bounds, sc.mesh.delta = l, h, d
    sc.geometry = true
    return nothing
end

function solve!(sc::FFTIGF, x, y, z, charge; gamma=1, active=nothing)
    sc.prepared = false
    sc.deposited = false
    g = _gamma(eltype(sc.mesh.rho), gamma)
    _check_arrays(sc.mesh.rho, (x, y, z, charge), active)
    sc.empty_source = false
    if sc.options.adaptive
        ext = _extent(sc, x, y, z, charge, active)
        if ext.lo[1] == Inf
            sc.geometry = false
            sc.empty_source = true
        else
            _fit_geometry!(sc, ext)
        end
    end
    if sc.empty_source
        fill!(sc.mesh.rho, zero(eltype(sc.mesh.rho)))
    else
        _deposit!(sc.mesh, x, y, z, charge, active)
    end
    sc.deposited = true
    return solve!(sc; gamma=g)
end

"""
    deposit!(solver::FFTIGF, x, y, z, charge; active=nothing, clear=true)

Deposit lab charges on the existing geometry and invalidate the prepared field.
Requires explicit constructor bounds or geometry from a previous combined solve.
`clear=false` accumulates charge. Follow with `solve!(solver; gamma=1)`.
"""

function deposit!(sc::FFTIGF, x, y, z, charge; active=nothing, clear::Bool=true)
    sc.prepared = false
    was_deposited = sc.deposited
    sc.deposited = false
    sc.geometry || throw(ArgumentError("deposit! requires fixed bounds or a previous source geometry"))
    (clear || was_deposited) || throw(ArgumentError("clear=false requires a valid deposited source"))
    _check_arrays(sc.mesh.rho, (x, y, z, charge), active)
    _deposit!(sc.mesh, x, y, z, charge, active; clear)
    sc.empty_source = false
    sc.deposited = true
    return sc
end

"""
    solve!(solver::FFTIGF; gamma=1)

Convolve the existing deposited source without refitting geometry or redepositing.
Returns `solver`. Omitting `gamma` resets it to 1, including after a relativistic solve.
"""
function solve!(sc::FFTIGF; gamma=1)
    sc.prepared = false
    g = _gamma(eltype(sc.mesh.rho), gamma)
    sc.deposited || throw(ArgumentError("solve! requires a deposited source"))
    sc.mesh.gamma = g
    if sc.empty_source
        fill!(sc.mesh.efield, zero(eltype(sc.mesh.rho)))
    else
        o = sc.options
        _solve_mesh!(sc.mesh; o.at_cathode, o.fft_flags, o.cache_green, o.pruned_fft)
    end
    sc.prepared = true
    return sc
end

function field!(Ex, Ey, Ez, sc::FFTIGF, x, y, z; active=nothing)
    sc.prepared || throw(ArgumentError("field! requires a successful solve!"))
    _check_arrays(sc.mesh.rho, (x, y, z, Ex, Ey, Ez), active)
    isempty(x) && return (Ex, Ey, Ez)
    m = sc.mesh
    interpolate_kernel!(get_backend(m.rho))(Ex, Ey, Ez, m.efield, x, y, z,
        m.min_bounds, inv.(m.delta), m.max_bounds, active, Val(sc.empty_source); ndrange=length(x))
    return Ex, Ey, Ez
end
