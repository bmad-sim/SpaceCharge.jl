using KernelAbstractions

"""
    interpolate_kernel!(Ex, Ey, Ez, Bx, By, Bz, efield, bfield, particles_x, particles_y, particles_z, min_bounds, delta)

A kernel for interpolating the electric and magnetic fields from the grid to particle positions.

This kernel is designed to be run on either a CPU or a GPU using `KernelAbstractions.jl`.

# Arguments
- `Ex`, `Ey`, `Ez`: 1D arrays to store the interpolated electric field components.
- `Bx`, `By`, `Bz`: 1D arrays to store the interpolated magnetic field components.
- `efield`: A 4D array representing the electric field on the grid.
- `bfield`: A 4D array representing the magnetic field on the grid.
- `particles_x`, `particles_y`, `particles_z`: 1D arrays of particle coordinates.
- `min_bounds`: A tuple or vector containing the minimum coordinates of the grid boundaries.
- `delta`: A tuple or vector containing the grid spacing in each dimension.
"""
@kernel function interpolate_kernel!(
    Ex,
    Ey,
    Ez,
    Bx,
    By,
    Bz,
    efield,
    bfield,
    particles_x,
    particles_y,
    particles_z,
    min_bounds,
    delta,
)
    i = @index(Global, Linear)

    # --- 1. Find the particle's normalized coordinates ---
    norm_x = (particles_x[i] - min_bounds[1]) / delta[1]
    norm_y = (particles_y[i] - min_bounds[2]) / delta[2]
    norm_z = (particles_z[i] - min_bounds[3]) / delta[3]

    # --- 2. Find the index of the lower-left-front grid cell ---
    ix = floor(Int, norm_x)
    iy = floor(Int, norm_y)
    iz = floor(Int, norm_z)

    # --- 3. Calculate the particle's position within its cell ---
    dx = norm_x - ix
    dy = norm_y - iy
    dz = norm_z - iz

    # --- 4. Calculate the trilinear weighting factors ---
    w_000 = (1 - dx) * (1 - dy) * (1 - dz)
    w_100 = dx * (1 - dy) * (1 - dz)
    w_010 = (1 - dx) * dy * (1 - dz)
    w_110 = dx * dy * (1 - dz)
    w_001 = (1 - dx) * (1 - dy) * dz
    w_101 = dx * (1 - dy) * dz
    w_011 = (1 - dx) * dy * dz
    w_111 = dx * dy * dz

    # --- 5. Interpolate the fields ---
    Ex[i] = (
        efield[ix + 1, iy + 1, iz + 1, 1] * w_000 +
        efield[ix + 2, iy + 1, iz + 1, 1] * w_100 +
        efield[ix + 1, iy + 2, iz + 1, 1] * w_010 +
        efield[ix + 2, iy + 2, iz + 1, 1] * w_110 +
        efield[ix + 1, iy + 1, iz + 2, 1] * w_001 +
        efield[ix + 2, iy + 1, iz + 2, 1] * w_101 +
        efield[ix + 1, iy + 2, iz + 2, 1] * w_011 +
        efield[ix + 2, iy + 2, iz + 2, 1] * w_111
    )
    Ey[i] = (
        efield[ix + 1, iy + 1, iz + 1, 2] * w_000 +
        efield[ix + 2, iy + 1, iz + 1, 2] * w_100 +
        efield[ix + 1, iy + 2, iz + 1, 2] * w_010 +
        efield[ix + 2, iy + 2, iz + 1, 2] * w_110 +
        efield[ix + 1, iy + 1, iz + 2, 2] * w_001 +
        efield[ix + 2, iy + 1, iz + 2, 2] * w_101 +
        efield[ix + 1, iy + 2, iz + 2, 2] * w_011 +
        efield[ix + 2, iy + 2, iz + 2, 2] * w_111
    )
    Ez[i] = (
        efield[ix + 1, iy + 1, iz + 1, 3] * w_000 +
        efield[ix + 2, iy + 1, iz + 1, 3] * w_100 +
        efield[ix + 1, iy + 2, iz + 1, 3] * w_010 +
        efield[ix + 2, iy + 2, iz + 1, 3] * w_110 +
        efield[ix + 1, iy + 1, iz + 2, 3] * w_001 +
        efield[ix + 2, iy + 1, iz + 2, 3] * w_101 +
        efield[ix + 1, iy + 2, iz + 2, 3] * w_011 +
        efield[ix + 2, iy + 2, iz + 2, 3] * w_111
    )

    Bx[i] = (
        bfield[ix + 1, iy + 1, iz + 1, 1] * w_000 +
        bfield[ix + 2, iy + 1, iz + 1, 1] * w_100 +
        bfield[ix + 1, iy + 2, iz + 1, 1] * w_010 +
        bfield[ix + 2, iy + 2, iz + 1, 1] * w_110 +
        bfield[ix + 1, iy + 1, iz + 2, 1] * w_001 +
        bfield[ix + 2, iy + 1, iz + 2, 1] * w_101 +
        bfield[ix + 1, iy + 2, iz + 2, 1] * w_011 +
        bfield[ix + 2, iy + 2, iz + 2, 1] * w_111
    )
    By[i] = (
        bfield[ix + 1, iy + 1, iz + 1, 2] * w_000 +
        bfield[ix + 2, iy + 1, iz + 1, 2] * w_100 +
        bfield[ix + 1, iy + 2, iz + 1, 2] * w_010 +
        bfield[ix + 2, iy + 2, iz + 1, 2] * w_110 +
        bfield[ix + 1, iy + 1, iz + 2, 2] * w_001 +
        bfield[ix + 2, iy + 1, iz + 2, 2] * w_101 +
        bfield[ix + 1, iy + 2, iz + 2, 2] * w_011 +
        bfield[ix + 2, iy + 2, iz + 2, 2] * w_111
    )
    Bz[i] = (
        bfield[ix + 1, iy + 1, iz + 1, 3] * w_000 +
        bfield[ix + 2, iy + 1, iz + 1, 3] * w_100 +
        bfield[ix + 1, iy + 2, iz + 1, 3] * w_010 +
        bfield[ix + 2, iy + 2, iz + 1, 3] * w_110 +
        bfield[ix + 1, iy + 1, iz + 2, 3] * w_001 +
        bfield[ix + 2, iy + 1, iz + 2, 3] * w_101 +
        bfield[ix + 1, iy + 2, iz + 2, 3] * w_011 +
        bfield[ix + 2, iy + 2, iz + 2, 3] * w_111
    )
end

"""
    interpolate_field(mesh, particles_x, particles_y, particles_z)

Interpolate the electric and magnetic fields from the grid to particle positions.

# Arguments
- `mesh`: A `Mesh3D` object.
- `particles_x`, `particles_y`, `particles_z`: 1D arrays of particle coordinates.

# Returns
- A tuple containing six 1D arrays: `(Ex, Ey, Ez, Bx, By, Bz)`.
"""
function interpolate_field(
    mesh::Mesh3D,
    particles_x,
    particles_y,
    particles_z,
)
    backend = get_backend(mesh.rho)
    kernel! = interpolate_kernel!(backend)

    num_particles = length(particles_x)
    Ex = similar(particles_x)
    Ey = similar(particles_x)
    Ez = similar(particles_x)
    Bx = similar(particles_x)
    By = similar(particles_x)
    Bz = similar(particles_x)

    kernel!(
        Ex,
        Ey,
        Ez,
        Bx,
        By,
        Bz,
        mesh.efield,
        mesh.bfield,
        particles_x,
        particles_y,
        particles_z,
        mesh.min_bounds,
        mesh.delta,
        ndrange=num_particles,
    )

    return Ex, Ey, Ez, Bx, By, Bz
end
