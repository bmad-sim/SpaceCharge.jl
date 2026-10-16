"""
    FFTIGF(grid_size; T=Float64, backend=CPU(),
           fft_flags=FFTW.MEASURE, cache_green=true, pruned_fft=true,
           at_cathode=false, field_method=:direct)

Reusable integrated Green-function FFT solver. Each particle-based `solve!`
fits the active particles unless given `bounds=(lo, hi)` for that call.
Active charges must be finite; with supplied bounds, active positions must also
be finite and inside the closed bounds.
`at_cathode=true` adds the image-charge field at z=0. FFT plans, Green spectra,
and deposition scratch are reused. GPU transforms are always full.
`field_method=:direct` integrates the electric field. `:potential_fd` integrates
the potential, with centered second-order differences inside and three-point
one-sided differences at the faces. Two-node axes use a two-point difference.
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
                pruned_fft::Bool=true, at_cathode::Bool=false,
                field_method::Symbol=:direct)
    T in (Float32, Float64) || throw(ArgumentError("FFTIGF supports Float32 and Float64"))
    all(k -> k >= 2, n) || throw(ArgumentError("grid dimensions must be at least 2"))
    backend isa Union{CPU,GPU} || throw(ArgumentError("unsupported FFT backend"))
    field_method in (:direct, :potential_fd) ||
        throw(ArgumentError("field_method must be :direct or :potential_fd"))
    # Placeholder geometry; a particle-based solve establishes physical bounds.
    lo, hi = ntuple(_ -> zero(T), 3), ntuple(_ -> one(T), 3)
    delta = (hi .- lo) ./ (n .- 1)
    rho = KernelAbstractions.zeros(backend, T, n...)
    efield = KernelAbstractions.zeros(backend, T, n..., 3)
    options = (;fft_flags, cache_green, pruned_fft, at_cathode, field_method)
    return FFTIGF(n, lo, hi, delta, one(T), rho, efield, nothing, nothing,
                  options)
end

function _particle_bounds(x, y, z, mask=nothing)
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
    points = mask === nothing ? Base.Broadcast.broadcasted(point, x, y, z) :
        Base.Broadcast.broadcasted(masked_point, mask, eachindex(x), Ref(x), Ref(y), Ref(z))
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
function _grid_bounds_from_particles(sc::FFTIGF{T}, x, y, z, mask) where T
    n = sc.grid_size
    particle_bounds = _particle_bounds(x, y, z, mask)
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

function _check_particle_presence(x, mask)
    (isempty(x) || (mask !== nothing && !any(mask))) &&
        throw(ArgumentError("at least one active particle is required"))
    return nothing
end

function solve!(sc::FFTIGF, x, y, z, charge, gamma; mask=nothing, bounds=nothing)
    _check_arrays(sc.rho, (x, y, z, charge), mask)
    g = _gamma(eltype(sc.rho), gamma)
    _check_particle_presence(x, mask)
    if bounds === nothing
        bounds = _grid_bounds_from_particles(sc, x, y, z, mask)
    end
    _set_grid_bounds!(sc, bounds)
    _deposit!(sc, x, y, z, charge, mask)
    _solve_mesh!(sc; gamma=g)
    return nothing
end

"""
    deposit!(solver::FFTIGF, x, y, z, charge; mask=nothing, clear=true)

Deposit on geometry established by the last particle-based `solve!`.
Active positions must be finite and inside its closed bounds; active charges must be finite.
`clear=false` accumulates charge.
Empty/all-inactive input is an error with `clear=true` and adds nothing with
`clear=false`. Follow with `solve!(solver, gamma)` using the desired Lorentz factor.
"""
function deposit!(sc::FFTIGF, x, y, z, charge; mask=nothing, clear::Bool=true)
    _check_arrays(sc.rho, (x, y, z, charge), mask)
    clear && _check_particle_presence(x, mask)
    _deposit!(sc, x, y, z, charge, mask; clear)
    return sc
end

"""
    solve!(solver::FFTIGF, gamma)

Convolve the existing deposited charge without refitting geometry or redepositing.
Returns `nothing`. Supply positional `gamma` on every call; pass `1` for
stationary calculations.
"""
function solve!(sc::FFTIGF, gamma)
    _solve_mesh!(sc; gamma=_gamma(eltype(sc.rho), gamma))
    return nothing
end

function field!(Ex, Ey, Ez, sc::FFTIGF, x, y, z; mask=nothing)
    _check_arrays(sc.rho, (x, y, z, Ex, Ey, Ez), mask)
    isempty(x) && return (Ex, Ey, Ez)
    interpolate_kernel!(get_backend(sc.rho))(Ex, Ey, Ez, sc.efield, x, y, z,
        sc.min_bounds, inv.(sc.delta), mask; ndrange=length(x))
    return Ex, Ey, Ez
end
