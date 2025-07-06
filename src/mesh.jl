using Adapt
using AbstractFFTs

"""
    Mesh3D{T, A}

A struct to represent a 3D Cartesian mesh.

# Fields
- `grid_size::NTuple{3, Int}`:   Size of the grid in each dimension (x, y, z).
- `min_bounds::NTuple{3, T}`:    Minimum bounds of the physical domain in each dimension.
- `max_bounds::NTuple{3, T}`:    Maximum bounds of the physical domain in each dimension.
- `delta::NTuple{3, T}`:         Grid spacing in each dimension.
- `gamma::T`:                    Lorentz factor for relativistic calculations.
- `total_charge::T`:             Total charge in the system.
- `rho::A`:                      Charge density array.
- `phi::A`:                      Electric potential array.
- `efield::AbstractArray{T, 4}`: Electric field array with the last dimension for components (x, y, z).
- `bfield::AbstractArray{T, 4}`: Magnetic field array with the last dimension for components (x, y, z).
- `_workspace::Union{Nothing, NamedTuple}`: Internal workspace for solver optimization (pre-allocated arrays and FFT plans).
"""
mutable struct Mesh3D{T <: AbstractFloat, A <: AbstractArray{T}}
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
    phi::A
    efield::AbstractArray{T, 4} # Last dimension for component (x,y,z)
    bfield::AbstractArray{T, 4}
    # Optimization workspace (lazy initialization)
    _workspace::Union{Nothing, NamedTuple}
end

"""
    _get_workspace(mesh::Mesh3D{T, A}) where {T<:AbstractFloat, A<:AbstractArray}

Get or create the workspace for the mesh. This includes pre-allocated arrays 
and cached in-place FFT plans for maximum performance of the free space solver.

The workspace contains:
- `crho`: Complex charge density array (doubled size for FFT padding)
- `cgrn`: Complex Green's function array 
- `temp_result`: Complex temporary result array
- `fft_plan_inplace`: Cached in-place forward FFT plan
- `ifft_plan_inplace`: Cached in-place inverse FFT plan
"""
function _get_workspace(mesh::Mesh3D{T, A}) where {T<:AbstractFloat, A<:AbstractArray}
    if mesh._workspace === nothing
        nx, ny, nz = mesh.grid_size
        nx2, ny2, nz2 = 2nx, 2ny, 2nz
        
        # Pre-allocate workspace arrays
        crho = similar(mesh.rho, Complex{T}, nx2, ny2, nz2)
        cgrn = similar(mesh.rho, Complex{T}, nx2, ny2, nz2)
        temp_result = similar(mesh.rho, Complex{T}, nx2, ny2, nz2)
        
        # Create in-place FFT plans (optimized for memory efficiency)
        fft_plan_inplace = plan_fft!(crho)
        ifft_plan_inplace = plan_ifft!(temp_result)
        
        mesh._workspace = (
            crho = crho,
            cgrn = cgrn,
            temp_result = temp_result,
            fft_plan_inplace = fft_plan_inplace,
            ifft_plan_inplace = ifft_plan_inplace,
        )
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
- `array_type::Type{<:AbstractArray} = Array`: The array type to use for data storage (e.g., `Array` for CPU, `CuArray` for GPU).
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
    array_type::Type{<:AbstractArray}=Array,
    gamma::Real=1.0,
    total_charge::Real=0.0
)
    # --- Validation ---
    if any(grid_size .<= 0)
        error("All elements of grid_size must be positive.")
    end
    
    if isempty(particles_x) || isempty(particles_y) || isempty(particles_z)
        error("Particle arrays cannot be empty.")
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
    # Create arrays on the CPU first
    rho_cpu = zeros(T, grid_size)
    phi_cpu = zeros(T, grid_size)
    efield_cpu = zeros(T, (grid_size..., 3))
    bfield_cpu = zeros(T, (grid_size..., 3))

    # Move arrays to the target device (e.g., GPU)
    rho = adapt(array_type, rho_cpu)
    phi = adapt(array_type, phi_cpu)
    efield = adapt(array_type, efield_cpu)
    bfield = adapt(array_type, bfield_cpu)

    # --- Struct Instantiation ---
    A = typeof(rho) # Get the concrete array type
    return Mesh3D{T, A}(
        grid_size,
        cv_min_bounds,
        cv_max_bounds,
        delta,
        cv_gamma,
        cv_total_charge,
        rho,
        phi,
        efield,
        bfield,
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
- `array_type::Type{<:AbstractArray} = Array`: The array type to use for data storage (e.g., `Array` for CPU, `CuArray` for GPU).
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
    array_type::Type{<:AbstractArray}=Array,
    gamma::Real=1.0,
    total_charge::Real=0.0
)
    # --- Validation ---
    if any(grid_size .<= 0)
        error("All elements of grid_size must be positive.")
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
    # Create arrays on the CPU first
    rho_cpu = zeros(T, grid_size)
    phi_cpu = zeros(T, grid_size)
    efield_cpu = zeros(T, (grid_size..., 3))
    bfield_cpu = zeros(T, (grid_size..., 3))

    # Move arrays to the target device (e.g., GPU)
    rho = adapt(array_type, rho_cpu)
    phi = adapt(array_type, phi_cpu)
    efield = adapt(array_type, efield_cpu)
    bfield = adapt(array_type, bfield_cpu)

    # --- Struct Instantiation ---
    A = typeof(rho) # Get the concrete array type
    return Mesh3D{T, A}(
        grid_size,
        cv_min_bounds,
        cv_max_bounds,
        delta,
        cv_gamma,
        cv_total_charge,
        rho,
        phi,
        efield,
        bfield,
        nothing,
    )
end
