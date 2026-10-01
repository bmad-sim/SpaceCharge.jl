using AbstractFFTs

"""
    Mesh3D{T, A, B}

A struct to represent a 3D Cartesian mesh.

# Fields
- `grid_size::NTuple{3, Int}`:   Size of the grid in each dimension (x, y, z).
- `min_bounds::NTuple{3, T}`:    Minimum bounds of the physical domain in each dimension.
- `max_bounds::NTuple{3, T}`:    Maximum bounds of the physical domain in each dimension.
- `delta::NTuple{3, T}`:         Grid spacing in each dimension.
- `gamma::T`:                    Lorentz factor for relativistic calculations.
- `total_charge::T`:             Total charge in the system.
- `rho::A`:                      Charge density array.
- `efield::B`:                   Electric field array with the last dimension for components (x, y, z).
- `_workspace::Union{Nothing, NamedTuple}`: Internal workspace for solver optimization (pre-allocated arrays and FFT plans).
"""
mutable struct Mesh3D{T <: AbstractFloat, A <: AbstractArray{T}, B <: AbstractArray{T}}
    # Grid dimensions
    grid_size::NTuple{3, Int}
    # Physical domain
    min_bounds::NTuple{3, T}
    max_bounds::NTuple{3, T}
    delta::NTuple{3, T}
    # Physics parameters
    gamma::T
    total_charge::T
    # Data arrays (CPU or GPU)
    rho::A
    efield::B
    # Optimization workspace (lazy initialization)
    _workspace::Union{Nothing, NamedTuple}
end

"""
    _get_workspace(mesh; fft_flags=FFTW.MEASURE)

Lazily allocate real FFT buffers and plans. CPU planning effort is configurable;
GPU plans use the backend default. Replacing mesh arrays rebuilds the workspace.
"""
function _get_workspace(mesh::Mesh3D{T}; fft_flags=FFTW.MEASURE) where T
    n = mesh.grid_size
    size(mesh.rho) == n || throw(DimensionMismatch("rho does not match grid_size"))
    size(mesh.efield) == (n..., 3) || throw(DimensionMismatch("efield does not match grid_size"))
    w = mesh._workspace
    if w === nothing || w.owner !== mesh.rho || w.field_owner !== mesh.efield ||
       size(w.pad) != 2 .* n || w.flags != fft_flags
        dims = 2 .* n
        pad = similar(mesh.rho, dims)
        result = similar(pad)
        rhohat = similar(mesh.rho, Complex{T}, (n[1]+1, dims[2], dims[3]))
        product = similar(rhohat)
        corners = similar(mesh.rho, n .+ 1)
        octant = similar(mesh.rho)
        if get_backend(mesh.rho) isa CPU
            forward = plan_rfft(pad; flags=fft_flags)
            backward = plan_brfft(product, dims[1]; flags=fft_flags)
        else
            forward = plan_rfft(pad)
            backward = plan_brfft(product, dims[1])
        end
        mesh._workspace = (;owner=mesh.rho, field_owner=mesh.efield,
                            flags=fft_flags, pad, result, rhohat, product, forward, backward,
                            corners, octant)
    end
    return mesh._workspace
end
"""
    Mesh3D(grid_size, particles_x, particles_y, particles_z; kwargs...)

Construct a `Mesh3D` object with bounds automatically determined from particle positions.
This is the recommended constructor as it eliminates the need for bounds checking during deposition.

# Arguments
- `grid_size::NTuple{3, Int}`: Number of grid points in each dimension (nx, ny, nz).
- `particles_x`: Array of particle x-coordinates.
- `particles_y`: Array of particle y-coordinates.  
- `particles_z`: Array of particle z-coordinates.

# Keyword Arguments
- `T::Type{<:AbstractFloat} = Float64`: The floating-point type for the mesh data.
- `backend = get_backend(particles_x)`: The KernelAbstractions backend to use. Defaults to the backend of the particle arrays.
- `gamma::Real = 1.0`: Relativistic gamma factor of the beam.
- `total_charge::Real = 0.0`: Total charge of the particle bunch.

# Returns
- A `Mesh3D` object with bounds automatically set to contain all particles with padding.
"""
function Mesh3D(
    grid_size::NTuple{3, Int},
    particles_x,
    particles_y,
    particles_z;
    T::Type{<:AbstractFloat}=Float64,
    backend=get_backend(particles_x),
    gamma::Real=1.0,
    total_charge::Real=0.0
)
    # --- Validation ---
    if any(grid_size .<= 1)
        error("All elements of grid_size must be at least 2.")
    end

    if isempty(particles_x) || isempty(particles_y) || isempty(particles_z)
        error("Particle arrays cannot be empty.")
    end

    if !(length(particles_x) == length(particles_y) == length(particles_z))
        error("Particle coordinate arrays must have the same length.")
    end

    # --- Automatic Bounds Calculation ---
    # Find min/max of particle positions
    x_min, x_max = extrema(particles_x)
    y_min, y_max = extrema(particles_y)
    z_min, z_max = extrema(particles_z)
    
    # Compute initial bounds and grid spacing
    min_bounds = (x_min, y_min, z_min)
    max_bounds = (x_max, y_max, z_max)
    delta = ((max_bounds[1] - min_bounds[1]) / (grid_size[1] - 1),
             (max_bounds[2] - min_bounds[2]) / (grid_size[2] - 1),
             (max_bounds[3] - min_bounds[3]) / (grid_size[3] - 1))
    
    # Small padding to protect against indexing errors (Fortran logic)
    min_bounds = (min_bounds[1] - 1e-6 * delta[1],
                  min_bounds[2] - 1e-6 * delta[2],
                  min_bounds[3] - 1e-6 * delta[3])
    max_bounds = (max_bounds[1] + 1e-6 * delta[1],
                  max_bounds[2] + 1e-6 * delta[2],
                  max_bounds[3] + 1e-6 * delta[3])
    
    # Recompute delta after padding
    delta = ((max_bounds[1] - min_bounds[1]) / (grid_size[1] - 1),
             (max_bounds[2] - min_bounds[2]) / (grid_size[2] - 1),
             (max_bounds[3] - min_bounds[3]) / (grid_size[3] - 1))

    # Patch: If any delta is zero, set it to a small value (1e-6)
    delta = (
        delta[1] == 0 ? 1e-6 : delta[1],
        delta[2] == 0 ? 1e-6 : delta[2],
        delta[3] == 0 ? 1e-6 : delta[3]
    )
    delta = T.(delta)

    # --- Type Conversion ---
    cv_min_bounds = T.(min_bounds)
    cv_max_bounds = T.(max_bounds)
    cv_gamma = T(gamma)
    cv_total_charge = T(total_charge)

    # --- Array Allocation ---
    rho = KernelAbstractions.zeros(backend, T, grid_size...)
    efield = KernelAbstractions.zeros(backend, T, grid_size..., 3)

    # --- Struct Instantiation ---
    return Mesh3D{T, typeof(rho), typeof(efield)}(
        grid_size,
        cv_min_bounds,
        cv_max_bounds,
        delta,
        cv_gamma,
        cv_total_charge,
        rho,
        efield,
        nothing,
    )
end

"""
    Mesh3D(grid_size, min_bounds, max_bounds; kwargs...)

Construct a `Mesh3D` object with manually specified bounds.
Use the particle-based constructor instead for automatic bounds determination.

# Arguments
- `grid_size::NTuple{3, Int}`: Number of grid points in each dimension (nx, ny, nz).
- `min_bounds::NTuple{3, Real}`: Physical coordinates of the lower-left-front corner of the mesh (x_min, y_min, z_min).
- `max_bounds::NTuple{3, Real}`: Physical coordinates of the upper-right-back corner of the mesh (x_max, y_max, z_max).

# Keyword Arguments
- `T::Type{<:AbstractFloat} = Float64`: The floating-point type for the mesh data.
- `backend = CPU()`: The KernelAbstractions backend to use (e.g., `CPU()` for CPU, `CUDABackend()` for GPU).
- `gamma::Real = 1.0`: Relativistic gamma factor of the beam.
- `total_charge::Real = 0.0`: Total charge of the particle bunch.

# Returns
- A `Mesh3D` object with initialized fields.
"""
function Mesh3D(
    grid_size::NTuple{3, Int},
    min_bounds::NTuple{3, Real},
    max_bounds::NTuple{3, Real};
    T::Type{<:AbstractFloat}=Float64,
    backend=CPU(),
    gamma::Real=1.0,
    total_charge::Real=0.0
)
    # --- Validation ---
    if any(grid_size .<= 1)
        error("All elements of grid_size must be at least 2.")
    end
    if any(max_bounds .<= min_bounds)
        error("max_bounds must be strictly greater than min_bounds for all dimensions.")
    end

    # --- Type Conversion ---
    cv_min_bounds = T.(min_bounds)
    cv_max_bounds = T.(max_bounds)
    cv_gamma = T(gamma)
    cv_total_charge = T(total_charge)

    # --- Derived Properties ---
    delta = (cv_max_bounds .- cv_min_bounds) ./ (grid_size .- 1)

    # --- Array Allocation ---
    rho = KernelAbstractions.zeros(backend, T, grid_size...)
    efield = KernelAbstractions.zeros(backend, T, grid_size..., 3)

    # --- Struct Instantiation ---
    return Mesh3D{T, typeof(rho), typeof(efield)}(
        grid_size,
        cv_min_bounds,
        cv_max_bounds,
        delta,
        cv_gamma,
        cv_total_charge,
        rho,
        efield,
        nothing,
    )
end

function Base.show(io::IO, mesh::Mesh3D{T}) where {T}
    nx, ny, nz = mesh.grid_size
    lo = mesh.min_bounds
    hi = mesh.max_bounds
    arr_type = nameof(typeof(mesh.rho).name.wrapper)
    print(io, "Mesh3D{$T, $arr_type} ($(nx)x$(ny)x$(nz)) bounds=[($(lo[1]),$(lo[2]),$(lo[3])), ($(hi[1]),$(hi[2]),$(hi[3]))] gamma=$(mesh.gamma)")
end
