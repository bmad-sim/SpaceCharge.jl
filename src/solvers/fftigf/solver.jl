"""
    FFTIGF(grid_size; T=Float64, backend=CPU(),
           fft_flags=FFTW.MEASURE, cache_green=true, pruned_fft=true,
           at_cathode=false)

Reusable integrated-field Green-function FFT solver. Each particle-based `solve!`
fits the active particles unless given `bounds=(lo, hi)` for that call.
Active charges must be finite; with supplied bounds, active positions must also
be finite and inside the closed bounds.
`at_cathode=true` adds the image-charge field at z=0. FFT plans, Green spectra,
and deposition scratch are reused. GPU transforms are always full.
"""
mutable struct FFTIGF{T <: AbstractFloat,A <: AbstractArray{T},B <: AbstractArray{T},O}
    grid_size::NTuple{3,Int}
    min_bounds::NTuple{3,T}
    max_bounds::NTuple{3,T}
    delta::NTuple{3,T}
    gamma::T
    rho::A
    efield::B
    _workspace::Union{Nothing,NamedTuple}
    _deposit_workspace::Union{Nothing,Vector{Array{T,3}}}
    options::O
end

function Base.show(io::IO, sc::FFTIGF)
    print(io, "FFTIGF{", eltype(sc.rho), "}(", sc.grid_size,
          ", ", get_backend(sc.rho), ")")
end

function FFTIGF(n::NTuple{3,Int}; T=Float64, backend=CPU(),
                fft_flags=FFTW.MEASURE, cache_green::Bool=true,
                pruned_fft::Bool=true, at_cathode::Bool=false)
    T in (Float32, Float64) || throw(ArgumentError("FFTIGF supports Float32 and Float64"))
    all(k -> k >= 2, n) || throw(ArgumentError("grid dimensions must be at least 2"))
    backend isa Union{CPU,GPU} || throw(ArgumentError("unsupported FFT backend"))
    # Placeholder geometry; a particle-based solve establishes physical bounds.
    lo, hi = ntuple(_ -> zero(T), 3), ntuple(_ -> one(T), 3)
    delta = (hi .- lo) ./ (n .- 1)
    rho = KernelAbstractions.zeros(backend, T, n...)
    efield = KernelAbstractions.zeros(backend, T, n..., 3)
    options = (;fft_flags, cache_green, pruned_fft, at_cathode)
    return FFTIGF(n, lo, hi, delta, one(T), rho, efield, nothing, nothing,
                  options)
end

function _particle_bounds(x, y, z, active=nothing)
    T = eltype(x)
    init = (lo=ntuple(_ -> T(Inf), 3), hi=ntuple(_ -> T(-Inf), 3))
    isempty(x) && return init
    @inline combine(a, b) = (lo=min.(a.lo, b.lo), hi=max.(a.hi, b.hi))
    @inline point(a, b, c) = (lo=(a, b, c), hi=(a, b, c))
    @inline function masked_point(keep, i, x, y, z)
        keep || return init
        p = @inbounds (x[i], y[i], z[i])
        (lo=p, hi=p)
    end
    points = active === nothing ? Base.Broadcast.broadcasted(point, x, y, z) :
        Base.Broadcast.broadcasted(masked_point, active, eachindex(x), Ref(x), Ref(y), Ref(z))
    return mapreduce(identity, combine, points; init)
end

function _set_grid_bounds!(sc::FFTIGF{T}, bounds) where T
    lo, hi = bounds
    length(lo) == length(hi) == 3 || throw(ArgumentError("bounds need three coordinates"))
    lo, hi = Tuple(T.(lo)), Tuple(T.(hi))
    all(isfinite, lo) && all(isfinite, hi) && all(hi .> lo) ||
        throw(ArgumentError("bounds must be finite with hi > lo"))
    delta = (hi .- lo) ./ (sc.grid_size .- 1)
    all(d -> isfinite(d) && d > 0 && isfinite(inv(d)), delta) ||
        throw(ArgumentError("grid spacing must be finite and positive"))
    sc.min_bounds, sc.max_bounds, sc.delta = lo, hi, delta
    return nothing
end

# Compute padded grid bounds without changing the solver.
function _grid_bounds_from_particles(sc::FFTIGF{T}, x, y, z, active) where T
    n = sc.grid_size
    particle_bounds = _particle_bounds(x, y, z, active)
    lo, hi = particle_bounds.lo, particle_bounds.hi
    all(isfinite, lo) && all(isfinite, hi) || throw(ArgumentError("active particle coordinates must be finite"))
    pairs = ntuple(3) do k
        lower = lo[k]
        upper = hi[k]
        if lower == upper
            # Enough precision to represent every node, even far from the origin.
            spacing = max(T(1e-6), T(4) * eps(lower))
            half_span = spacing * T(n[k] - 1) / 2
            lower_bound = lower - half_span
            upper_bound = upper + half_span
        else
            padding = T(1e-6) * (upper - lower) / (n[k] - 1)
            lower_bound = lower - padding
            upper_bound = upper + padding
        end
        return lower_bound, upper_bound
    end
    return first.(pairs), last.(pairs)
end

function _check_active_particles(x, active)
    (isempty(x) || (active !== nothing && !any(active))) &&
        throw(ArgumentError("at least one active particle is required"))
    return nothing
end

function solve!(sc::FFTIGF, x, y, z, charge; gamma=1, active=nothing, bounds=nothing)
    _check_arrays(sc.rho, (x, y, z, charge), active)
    g = _gamma(eltype(sc.rho), gamma)
    _check_active_particles(x, active)
    if bounds === nothing
        bounds = _grid_bounds_from_particles(sc, x, y, z, active)
    end
    _set_grid_bounds!(sc, bounds)
    _deposit!(sc, x, y, z, charge, active)
    _solve_mesh!(sc; gamma=g)
    return nothing
end

"""
    deposit!(solver::FFTIGF, x, y, z, charge; active=nothing, clear=true)

Deposit on geometry established by the last particle-based `solve!`.
Active positions must be finite and inside its closed bounds; active charges must be finite.
`clear=false` accumulates charge.
Empty/all-inactive input is an error with `clear=true` and adds nothing with
`clear=false`. Follow with `solve!(solver; gamma=1)`.
"""
function deposit!(sc::FFTIGF, x, y, z, charge; active=nothing, clear::Bool=true)
    _check_arrays(sc.rho, (x, y, z, charge), active)
    clear && _check_active_particles(x, active)
    _deposit!(sc, x, y, z, charge, active; clear)
    return sc
end

"""
    solve!(solver::FFTIGF; gamma=1)

Convolve the existing deposited charge without refitting geometry or redepositing.
Returns `nothing`. Omitting `gamma` resets it to 1, including after a relativistic solve.
"""
function solve!(sc::FFTIGF; gamma=1)
    _solve_mesh!(sc; gamma=_gamma(eltype(sc.rho), gamma))
    return nothing
end

function field!(Ex, Ey, Ez, sc::FFTIGF, x, y, z; active=nothing)
    _check_arrays(sc.rho, (x, y, z, Ex, Ey, Ez), active)
    isempty(x) && return (Ex, Ey, Ez)
    interpolate_kernel!(get_backend(sc.rho))(Ex, Ey, Ez, sc.efield, x, y, z,
        sc.min_bounds, inv.(sc.delta), active; ndrange=length(x))
    return Ex, Ey, Ez
end
