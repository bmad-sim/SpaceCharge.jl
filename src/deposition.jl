using KernelAbstractions
using Atomix

"""
    clear_mesh!(mesh)

Clear the charge density array in the mesh to zero.
This should be called before deposition to ensure no accumulation from previous runs.
"""
function clear_mesh!(mesh::Mesh3D)
    fill!(mesh.rho, 0)
end

"""
    deposit_particle!(rho, particle_x, particle_y, particle_z, particle_q, min_bounds, delta; use_atomic=false)

Core deposition algorithm for a single particle using Cloud-in-Cell (CIC) method.
This is the shared logic between CPU and GPU implementations.

# Arguments
- `rho`: 3D charge density array to modify
- `particle_x, particle_y, particle_z`: Single particle coordinates
- `particle_q`: Single particle charge
- `min_bounds`: Grid minimum bounds (x_min, y_min, z_min)
- `delta`: Grid spacing (dx, dy, dz)
- `use_atomic`: Whether to use atomic operations for thread safety
"""
function deposit_particle!(
    rho,
    particle_x,
    particle_y,
    particle_z,
    particle_q,
    min_bounds,
    delta;
    use_atomic::Bool=false
)
    # --- 1. Find the particle's normalized coordinates ---
    norm_x = (particle_x - min_bounds[1]) / delta[1]
    norm_y = (particle_y - min_bounds[2]) / delta[2]
    norm_z = (particle_z - min_bounds[3]) / delta[3]

    # --- 2. Find the index of the lower-left-front grid cell ---
    ix = floor(Int, norm_x)
    iy = floor(Int, norm_y)
    iz = floor(Int, norm_z)

    # --- 3. Calculate the particle's position within its cell ---
    dx = norm_x - ix
    dy = norm_y - iy
    dz = norm_z - iz

    # --- 4. Pre-calculate weight components ---
    w_x0 = 1 - dx
    w_x1 = dx
    w_y0 = 1 - dy
    w_y1 = dy
    w_z0 = 1 - dz
    w_z1 = dz
    
    charge = particle_q

    # --- 5. Deposit charge to all 8 neighboring grid points ---
    # No bounds checking needed since mesh is sized to contain all particles
    if use_atomic
        # Use atomic operations for thread safety (GPU)
        Atomix.@atomic rho[ix + 1, iy + 1, iz + 1] += charge * w_x0 * w_y0 * w_z0
        Atomix.@atomic rho[ix + 2, iy + 1, iz + 1] += charge * w_x1 * w_y0 * w_z0
        Atomix.@atomic rho[ix + 1, iy + 2, iz + 1] += charge * w_x0 * w_y1 * w_z0
        Atomix.@atomic rho[ix + 2, iy + 2, iz + 1] += charge * w_x1 * w_y1 * w_z0
        Atomix.@atomic rho[ix + 1, iy + 1, iz + 2] += charge * w_x0 * w_y0 * w_z1
        Atomix.@atomic rho[ix + 2, iy + 1, iz + 2] += charge * w_x1 * w_y0 * w_z1
        Atomix.@atomic rho[ix + 1, iy + 2, iz + 2] += charge * w_x0 * w_y1 * w_z1
        Atomix.@atomic rho[ix + 2, iy + 2, iz + 2] += charge * w_x1 * w_y1 * w_z1
    else
        # Direct assignment for single-threaded CPU
        rho[ix + 1, iy + 1, iz + 1] += charge * w_x0 * w_y0 * w_z0
        rho[ix + 2, iy + 1, iz + 1] += charge * w_x1 * w_y0 * w_z0
        rho[ix + 1, iy + 2, iz + 1] += charge * w_x0 * w_y1 * w_z0
        rho[ix + 2, iy + 2, iz + 1] += charge * w_x1 * w_y1 * w_z0
        rho[ix + 1, iy + 1, iz + 2] += charge * w_x0 * w_y0 * w_z1
        rho[ix + 2, iy + 1, iz + 2] += charge * w_x1 * w_y0 * w_z1
        rho[ix + 1, iy + 2, iz + 2] += charge * w_x0 * w_y1 * w_z1
        rho[ix + 2, iy + 2, iz + 2] += charge * w_x1 * w_y1 * w_z1
    end
end

"""
    deposit_kernel!(rho, particles_x, particles_y, particles_z, particles_q, min_bounds, delta, grid_size)

A kernel for depositing particle charges onto a 3D grid using the Cloud-in-Cell (CIC) method.

This kernel is designed to be run on either a CPU or a GPU using `KernelAbstractions.jl`.
Bounds checking is not needed as the mesh is sized to contain all particles.

# Arguments
- `rho`: A 3D array representing the charge density on the grid. This array is modified in place.
- `particles_x`: A 1D array of particle x-coordinates.
- `particles_y`: A 1D array of particle y-coordinates.
- `particles_z`: A 1D array of particle z-coordinates.
- `particles_q`: A 1D array of particle charges.
- `min_bounds`: A tuple or vector containing the minimum coordinates of the grid boundaries (x_min, y_min, z_min).
- `delta`: A tuple or vector containing the grid spacing in each dimension (dx, dy, dz).
- `grid_size`: A tuple or vector containing the number of grid points in each dimension (nx, ny, nz).
"""
@kernel function deposit_kernel!(
    rho,
    particles_x,
    particles_y,
    particles_z,
    particles_q,
    min_bounds,
    delta,
    grid_size,
)
    i = @index(Global, Linear)
    
    # Use the shared core algorithm with atomic operations for thread safety
    deposit_particle!(
        rho,
        particles_x[i],
        particles_y[i],
        particles_z[i],
        particles_q[i],
        min_bounds,
        delta;
        use_atomic=true
    )
end

"""
    deposit_gpu!(mesh, particles_x, particles_y, particles_z, particles_q)

GPU-optimized deposit function using KernelAbstractions.
"""
function deposit_gpu!(
    mesh::Mesh3D,
    particles_x,
    particles_y,
    particles_z,
    particles_q
)
    # Run GPU kernel
    backend = get_backend(mesh.rho)
    kernel! = deposit_kernel!(backend)
    kernel!(
        mesh.rho,
        particles_x,
        particles_y,
        particles_z,
        particles_q,
        mesh.min_bounds,
        mesh.delta,
        mesh.grid_size,
        ndrange=length(particles_x),
        workgroupsize=256,
    )
end

"""
    deposit_vectorized!(mesh, particles_x, particles_y, particles_z, particles_q)

Serial CPU deposition, with a SIMD hint where the compiler can safely vectorize.
Particles can share grid nodes, so scatter updates retain their dependencies.
No bounds checking is needed as the mesh is sized to contain all particles.
"""
function deposit_vectorized!(
    mesh::Mesh3D,
    particles_x,
    particles_y, 
    particles_z,
    particles_q,
)
    n_particles = length(particles_x)
    
    # Process particles in chunks for better cache performance and vectorization
    chunk_size = 64  # Process 64 particles at a time for better cache locality
    
    for chunk_start in 1:chunk_size:n_particles
        chunk_end = min(chunk_start + chunk_size - 1, n_particles)
        
        # Process each particle in the chunk with vectorized operations where possible
        @inbounds @simd for i in chunk_start:chunk_end
            # Use the shared core algorithm without atomic operations for CPU
            deposit_particle!(
                mesh.rho,
                particles_x[i],
                particles_y[i],
                particles_z[i],
                particles_q[i],
                mesh.min_bounds,
                mesh.delta;
                use_atomic=false
            )
        end
    end
end

# Private grids avoid atomic contention for CPU deposition. Keep the additional
# storage bounded and use enough particles per chunk to amortize clearing and
# reducing a full grid. These conservative cutoffs do not affect the result.
const _DEPOSIT_MAX_SCRATCH_BYTES = 64 * 1024 * 1024
const _DEPOSIT_MAX_CHUNKS = 8
const _DEPOSIT_MIN_PARTICLES_PER_CHUNK = 16_384

function _deposit_chunk_count(nparticles, ncells, element_bytes, nthreads)
    bytes_per_grid = ncells * element_bytes
    memory_chunks = div(_DEPOSIT_MAX_SCRATCH_BYTES, max(1, bytes_per_grid))
    particles_per_chunk = max(_DEPOSIT_MIN_PARTICLES_PER_CHUNK, cld(ncells, 4))
    return max(1, min(nthreads, _DEPOSIT_MAX_CHUNKS, memory_chunks,
                      div(nparticles, particles_per_chunk)))
end

function _get_deposit_workspace!(mesh::Mesh3D{T}, nchunks) where {T}
    scratch = mesh._deposit_workspace
    if scratch === nothing || length(scratch) < nchunks || size(first(scratch)) != size(mesh.rho)
        scratch = [Array{T,3}(undef, size(mesh.rho)) for _ in 1:nchunks]
        mesh._deposit_workspace = scratch
    end
    return scratch
end

function _deposit_threaded!(mesh, particles_x, particles_y, particles_z, particles_q, scratch, nchunks)
    n = length(particles_x)
    bounds, delta = mesh.min_bounds, mesh.delta
    # Scratch belongs to the logical chunk, not the worker thread. Task
    # migration and calls from a non-primary Julia thread are therefore safe.
    Threads.@threads for chunk in 1:nchunks
        grid = scratch[chunk]
        fill!(grid, zero(eltype(grid)))
        first_particle = div(n * (chunk - 1), nchunks) + 1
        last_particle = div(n * chunk, nchunks)
        @inbounds for i in first_particle:last_particle
            deposit_particle!(grid, particles_x[i], particles_y[i], particles_z[i],
                              particles_q[i], bounds, delta; use_atomic=false)
        end
    end
    rho = mesh.rho
    # All contributions for each cell are reduced in a fixed chunk order.
    # The existing mesh is retained when deposit!(...; clear=false) is used.
    Threads.@threads for block in 1:nchunks
        first_cell = div(length(rho) * (block - 1), nchunks) + 1
        last_cell = div(length(rho) * block, nchunks)
        @inbounds @simd for cell in first_cell:last_cell
            charge = zero(eltype(rho))
            for chunk in 1:nchunks
                charge += scratch[chunk][cell]
            end
            rho[cell] += charge
        end
    end
    return nothing
end

function deposit_cpu!(mesh, particles_x, particles_y, particles_z, particles_q)
    # Retain the serial generic-array path; private scratch is ordinary dense
    # CPU storage, as are the normal Mesh3D CPU arrays.
    nchunks = mesh.rho isa Array && isbitstype(eltype(mesh.rho)) ? _deposit_chunk_count(length(particles_x),
        length(mesh.rho), sizeof(eltype(mesh.rho)), Threads.nthreads()) : 1
    if nchunks == 1
        deposit_vectorized!(mesh, particles_x, particles_y, particles_z, particles_q)
    else
        scratch = _get_deposit_workspace!(mesh, nchunks)
        _deposit_threaded!(mesh, particles_x, particles_y, particles_z, particles_q,
                           scratch, nchunks)
    end
    return nothing
end

"""
    deposit!(mesh, particles_x, particles_y, particles_z, particles_q; clear=true)

Deposit particle charges onto the grid in a `Mesh3D` object.

This function selects deposition based on the backend and workload:
- For CPU arrays: Uses private-grid threading for sufficiently large particle
  batches, with a serial fallback. Additional scratch is bounded to 64 MiB.
- For GPU arrays: Uses optimized GPU kernel

# Arguments
- `mesh`: A `Mesh3D` object.
- `particles_x`: A 1D array of particle x-coordinates.
- `particles_y`: A 1D array of particle y-coordinates.
- `particles_z`: A 1D array of particle z-coordinates.
- `particles_q`: A 1D array of particle charges.

# Keyword Arguments
- `clear::Bool=true`: Whether to clear the mesh before deposition. Set to false if you want to accumulate charges.
"""
function deposit!(
    mesh::Mesh3D,
    particles_x,
    particles_y,
    particles_z,
    particles_q;
    clear::Bool=true
)
    if !(length(particles_x) == length(particles_y) == length(particles_z) == length(particles_q))
        error("Particle coordinate and charge arrays must have the same length.")
    end

    if clear
        clear_mesh!(mesh)
    end
    
    backend = get_backend(mesh.rho)

    # Determine the appropriate deposition method based on array type
    if backend isa CPU
        deposit_cpu!(mesh, particles_x, particles_y, particles_z, particles_q)
    elseif backend isa GPU
        # Use GPU kernel for GPU arrays
        deposit_gpu!(mesh, particles_x, particles_y, particles_z, particles_q)
    else
        error("Unsupported backend: $backend")
    end
end
