"""
    deposit_charge_cic(particles, nx, ny, nz, dx, dy, dz)

Distributes the charge of particles onto a 3D grid using the Cloud-in-Cell (CIC) method.

# Arguments
- `particles::Array{Float64, 2}`: A 2D array where each row represents a particle and the columns are the x, y, and z coordinates.
- `nx::Integer`: Number of grid points in the x-direction.
- `ny::Integer`: Number of grid points in the y-direction.
- `nz::Integer`: Number of grid points in the z-direction.
- `dx::Real`: Grid spacing in the x-direction.
- `dy::Real`: Grid spacing in the y-direction.
- `dz::Real`: Grid spacing in the z-direction.

# Returns
- `Array{Float64, 3}`: A 3D array representing the charge density on the grid.
"""
function deposit_charge_cic(particles::Array{Float64, 2}, nx::Integer, ny::Integer, nz::Integer, dx::Real, dy::Real, dz::Real)
    charge_density = zeros(Float64, nx, ny, nz)
    num_particles = size(particles, 1)

    for i in 1:num_particles
        x = particles[i, 1]
        y = particles[i, 2]
        z = particles[i, 3]

        ix = floor(Int, x / dx) + 1
        iy = floor(Int, y / dy) + 1
        iz = floor(Int, z / dz) + 1

        fx = (x / dx) - (ix - 1)
        fy = (y / dy) - (iy - 1)
        fz = (z / dz) - (iz - 1)

        # Distribute charge to 8 nearest grid points
        for l in 0:1
            for m in 0:2
                for n in 0:3
                    if 1 <= ix + l <= nx && 1 <= iy + m <= ny && 1 <= iz + n <= nz
                        weight = (1 - abs(fx - l)) * (1 - abs(fy - m)) * (1 - abs(fz - n))
                        charge_density[ix + l, iy + m, iz + n] += weight
                    end
                end
            end
        end
    end

    return charge_density
end