using KernelAbstractions
using StaticArrays

"""
    deposit_kernel!(rho, particles_x, particles_y, particles_z, particles_q, min_bounds, delta, grid_size)

A kernel for depositing particle charges onto a 3D grid using the Cloud-in-Cell (CIC) method.

This kernel is designed to be run on either a CPU or a GPU using `KernelAbstractions.jl`.

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

    # --- 1. Find the particle's normalized coordinates ---
    # The coordinates are normalized to the grid spacing.
    norm_x = (particles_x[i] - min_bounds[1]) / delta[1]
    norm_y = (particles_y[i] - min_bounds[2]) / delta[2]
    norm_z = (particles_z[i] - min_bounds[3]) / delta[3]

    # --- 2. Find the index of the lower-left-front grid cell ---
    # These are the integer indices of the cell that the particle is in.
    ix = floor(Int, norm_x)
    iy = floor(Int, norm_y)
    iz = floor(Int, norm_z)

    # --- 3. Calculate the particle's position within its cell ---
    # These are the fractional distances from the lower-left-front corner of the cell.
    dx = norm_x - ix
    dy = norm_y - iy
    dz = norm_z - iz

    # --- 4. Calculate the trilinear weighting factors ---
    # These factors determine how much charge is deposited on each of the 8 surrounding grid points.
    w_000 = (1 - dx) * (1 - dy) * (1 - dz)
    w_100 = dx * (1 - dy) * (1 - dz)
    w_010 = (1 - dx) * dy * (1 - dz)
    w_110 = dx * dy * (1 - dz)
    w_001 = (1 - dx) * (1 - dy) * dz
    w_101 = dx * (1 - dy) * dz
    w_011 = (1 - dx) * dy * dz
    w_111 = dx * dy * dz

    # --- 5. Deposit the charge onto the 8 neighboring grid points ---
    # The charge is distributed according to the weighting factors.
    # We need to check if the grid point is within the grid boundaries before depositing charge.
    if 1 <= ix + 1 <= grid_size[1] && 1 <= iy + 1 <= grid_size[2] && 1 <= iz + 1 <= grid_size[3]
        rho[ix + 1, iy + 1, iz + 1] += particles_q[i] * w_000
    end
    if 1 <= ix + 2 <= grid_size[1] && 1 <= iy + 1 <= grid_size[2] && 1 <= iz + 1 <= grid_size[3]
        rho[ix + 2, iy + 1, iz + 1] += particles_q[i] * w_100
    end
    if 1 <= ix + 1 <= grid_size[1] && 1 <= iy + 2 <= grid_size[2] && 1 <= iz + 1 <= grid_size[3]
        rho[ix + 1, iy + 2, iz + 1] += particles_q[i] * w_010
    end
    if 1 <= ix + 2 <= grid_size[1] && 1 <= iy + 2 <= grid_size[2] && 1 <= iz + 1 <= grid_size[3]
        rho[ix + 2, iy + 2, iz + 1] += particles_q[i] * w_110
    end
    if 1 <= ix + 1 <= grid_size[1] && 1 <= iy + 1 <= grid_size[2] && 1 <= iz + 2 <= grid_size[3]
        rho[ix + 1, iy + 1, iz + 2] += particles_q[i] * w_001
    end
    if 1 <= ix + 2 <= grid_size[1] && 1 <= iy + 1 <= grid_size[2] && 1 <= iz + 2 <= grid_size[3]
        rho[ix + 2, iy + 1, iz + 2] += particles_q[i] * w_101
    end
    if 1 <= ix + 1 <= grid_size[1] && 1 <= iy + 2 <= grid_size[2] && 1 <= iz + 2 <= grid_size[3]
        rho[ix + 1, iy + 2, iz + 2] += particles_q[i] * w_011
    end
    if 1 <= ix + 2 <= grid_size[1] && 1 <= iy + 2 <= grid_size[2] && 1 <= iz + 2 <= grid_size[3]
        rho[ix + 2, iy + 2, iz + 2] += particles_q[i] * w_111
    end
end

"""
    deposit!(mesh, particles_x, particles_y, particles_z, particles_q)

Deposit particle charges onto the grid in a `Mesh3D` object.

This function is a high-level wrapper that calls the `deposit_kernel!`.
It automatically handles CPU and GPU arrays.

# Arguments
- `mesh`: A `Mesh3D` object.
- `particles_x`: A 1D array of particle x-coordinates.
- `particles_y`: A 1D array of particle y-coordinates.
- `particles_z`: A 1D array of particle z-coordinates.
- `particles_q`: A 1D array of particle charges.
"""
function deposit!(
    mesh::Mesh3D,
    particles_x,
    particles_y,
    particles_z,
    particles_q,
)
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
    )
end
