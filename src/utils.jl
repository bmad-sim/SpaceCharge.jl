
using Distributions

"""
    generate_particles(num_particles, mean_pos, cov_matrix)

Generates a collection of particles with a 3D Gaussian distribution.

# Arguments
- `num_particles::Integer`: The number of particles to generate.
- `mean_pos::Vector{Float64}`: A 3-element vector representing the mean position (x, y, z).
- `cov_matrix::Matrix{Float64}`: A 3x3 covariance matrix to specify the standard deviations and correlations.

# Returns
- `Array{Float64, 2}`: A 2D array of particle positions, where each row is a particle (x, y, z).
"""
function generate_particles(num_particles::Integer, mean_pos::Vector{Float64}, cov_matrix::Matrix{Float64})
    dist = MvNormal(mean_pos, cov_matrix)
    particles = rand(dist, num_particles)'
    return particles
end

"""
    lafun(x, y, z)

Helper function for the Integrated Green's Function calculation.
This is the analytical integral of the 1/r Green's function, as described in
R.D. Ryne, PRSTAB 1, 034401 (1998).
"""
function lafun(x::Real, y::Real, z::Real)
    r = sqrt(x^2 + y^2 + z^2)

    if r < 1e-12
        return 0.0
    end

    # Handle potential singularities in atan and log terms
    term1 = (abs(z) < 1e-12) ? 0.0 : -0.5 * z^2 * atan(x * y, z * r)
    term2 = (abs(y) < 1e-12) ? 0.0 : -0.5 * y^2 * atan(x * z, y * r)
    term3 = (abs(x) < 1e-12) ? 0.0 : -0.5 * x^2 * atan(y * z, x * r)

    term4 = (abs(y) < 1e-12 || abs(z) < 1e-12 || (x + r) < 1e-12) ? 0.0 : y * z * log(x + r)
    term5 = (abs(x) < 1e-12 || abs(z) < 1e-12 || (y + r) < 1e-12) ? 0.0 : x * z * log(y + r)
    term6 = (abs(x) < 1e-12 || abs(y) < 1e-12 || (z + r) < 1e-12) ? 0.0 : x * y * log(z + r)

    return term1 + term2 + term3 + term4 + term5 + term6
end


"""
    green_function(nx, ny, nz, dx, dy, dz; gamma=1.0)

Computes the Integrated Green's function (IGF) for the electrostatic potential
in free space on a 3D Cartesian grid. This method avoids singularities by
averaging the Green's function over each grid cell.

# Arguments
- `nx::Integer`: Number of grid points in the x-direction.
- `ny::Integer`: Number of grid points in the y-direction.
- `nz::Integer`: Number of grid points in the z-direction.
- `dx::Real`: Grid spacing in the x-direction.
- `dy::Real`: Grid spacing in the y-direction.
- `dz::Real`: Grid spacing in the z-direction.
- `gamma::Real`: Relativistic gamma factor (defaults to 1.0).

# Returns
- `Array{Float64, 3}`: A 3D array representing the Green's function on the grid.
"""
function green_function(nx::Integer, ny::Integer, nz::Integer, dx::Real, dy::Real, dz::Real; gamma::Real=1.0)
    g = zeros(Float64, nx, ny, nz)
    for i in 1:nx
        for j in 1:ny
            for k in 1:nz
                u = (i - 1) * dx
                v = (j - 1) * dy
                w = (k - 1) * dz

                # Center the grid for correct distance calculation (periodicity)
                if i > nx / 2; u = (i - nx - 1) * dx; end
                if j > ny / 2; v = (j - ny - 1) * dy; end
                if k > nz / 2; w = (k - nz - 1) * dz; end

                x1 = u - 0.5 * dx
                x2 = u + 0.5 * dx
                y1 = v - 0.5 * dy
                y2 = v + 0.5 * dy
                z1 = (w - 0.5 * dz) * gamma
                z2 = (w + 0.5 * dz) * gamma

                val = (lafun(x2, y2, z2) - lafun(x1, y2, z2) - lafun(x2, y1, z2) - lafun(x2, y2, z1) -
                       lafun(x1, y1, z1) + lafun(x1, y1, z2) + lafun(x1, y2, z1) + lafun(x2, y1, z1))

                g[i, j, k] = val / (dx * dy * dz * gamma)
            end
        end
    end
    return g
end
