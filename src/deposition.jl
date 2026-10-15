using KernelAbstractions
using Atomix

"""
    deposit_particle!(rho, particle_x, particle_y, particle_z, particle_q, min_bounds, inv_delta; use_atomic=false)

Core deposition algorithm for a single particle using Cloud-in-Cell (CIC) method.
This is the shared logic between CPU and GPU implementations.

# Arguments
- `rho`: 3D charge density array to modify
- `particle_x, particle_y, particle_z`: Single particle coordinates
- `particle_q`: Single particle charge
- `min_bounds`: Grid minimum bounds (x_min, y_min, z_min)
- `inv_delta`: Reciprocal grid spacing, computed outside the particle loop
- `use_atomic`: Whether to use atomic operations for thread safety
"""
@inline function deposit_particle!(
    rho,
    particle_x,
    particle_y,
    particle_z,
    particle_q,
    min_bounds,
    inv_delta;
    use_atomic::Bool=false
)
    ix, dx = _cell(particle_x, min_bounds[1], inv_delta[1], size(rho, 1))
    iy, dy = _cell(particle_y, min_bounds[2], inv_delta[2], size(rho, 2))
    iz, dz = _cell(particle_z, min_bounds[3], inv_delta[3], size(rho, 3))

    # --- 4. Pre-calculate weight components ---
    w_x0 = 1 - dx
    w_x1 = dx
    w_y0 = 1 - dy
    w_y1 = dy
    w_z0 = 1 - dz
    w_z1 = dz

    charge = particle_q

    # --- 5. Deposit charge to all 8 neighboring grid points ---
    # The public scatter pass has already validated the physical domain.
    @inbounds begin
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
end

# Validate inside the scatter pass. Inactive values are never loaded.
@inline function _deposit_one!(rho, x, y, z, q, i, lo, hi, inv_delta, active, atomic)
    if _active(active, i)
        @inbounds xi, yi, zi, qi = x[i], y[i], z[i], q[i]
        _inside(xi, yi, zi, lo, hi) && isfinite(qi) ||
            error("active source must be finite and inside the FFT bounds")
        deposit_particle!(rho, xi, yi, zi, qi, lo, inv_delta; use_atomic=atomic)
    end
    return nothing
end

@kernel function deposit_kernel!(rho, x, y, z, q, lo, hi, inv_delta, active)
    i = @index(Global, Linear)
    _deposit_one!(rho, x, y, z, q, i, lo, hi, inv_delta, active, true)
end

function deposit_gpu!(mesh, x, y, z, q, active)
    isempty(x) && return nothing
    deposit_kernel!(get_backend(mesh.rho))(mesh.rho, x, y, z, q,
        mesh.min_bounds, mesh.max_bounds, inv.(mesh.delta), active;
        ndrange=length(x), workgroupsize=256)
    return nothing
end

function deposit_vectorized!(mesh, x, y, z, q, active=nothing)
    inv_delta = inv.(mesh.delta)
    @inbounds for i in eachindex(x)
        _deposit_one!(mesh.rho, x, y, z, q, i, mesh.min_bounds, mesh.max_bounds,
                      inv_delta, active, false)
    end
    return nothing
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

function _deposit_threaded!(mesh, particles_x, particles_y, particles_z, particles_q, scratch, nchunks, active=nothing)
    n = length(particles_x)
    bounds, inv_delta = mesh.min_bounds, inv.(mesh.delta)
    # Scratch belongs to the logical chunk, not the worker thread. Task
    # migration and calls from a non-primary Julia thread are therefore safe.
    Threads.@threads for chunk in 1:nchunks
        grid = scratch[chunk]
        fill!(grid, zero(eltype(grid)))
        first_particle = div(n * (chunk - 1), nchunks) + 1
        last_particle = div(n * chunk, nchunks)
        @inbounds for i in first_particle:last_particle
            _deposit_one!(grid, particles_x, particles_y, particles_z, particles_q,
                          i, bounds, mesh.max_bounds, inv_delta, active, false)
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

function deposit_cpu!(mesh, particles_x, particles_y, particles_z, particles_q, active)
    # Retain the serial generic-array path; private scratch is ordinary dense
    # CPU storage, as are the normal Mesh3D CPU arrays.
    nchunks = mesh.rho isa Array && isbitstype(eltype(mesh.rho)) ? _deposit_chunk_count(length(particles_x),
        length(mesh.rho), sizeof(eltype(mesh.rho)), Threads.nthreads()) : 1
    if nchunks == 1
        deposit_vectorized!(mesh, particles_x, particles_y, particles_z, particles_q, active)
    else
        scratch = _get_deposit_workspace!(mesh, nchunks)
        _deposit_threaded!(mesh, particles_x, particles_y, particles_z, particles_q,
                           scratch, nchunks, active)
    end
    return nothing
end

function _deposit!(mesh, x, y, z, q, active; clear=true)
    clear && fill!(mesh.rho, zero(eltype(mesh.rho)))
    if get_backend(mesh.rho) isa CPU
        deposit_cpu!(mesh, x, y, z, q, active)
    else
        deposit_gpu!(mesh, x, y, z, q, active)
    end
    return nothing
end
