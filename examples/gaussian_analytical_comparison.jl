using SpaceCharge
using Plots
using Random

# Constants
const EPSILON_0 = 8.8541878128e-12 # Permittivity of free space

# Simpson's Rule Integrator (copied from runtests.jl)
function simpsons_rule(f, a, b, n)
    if n % 2 != 0
        n += 1 # Ensure n is even
    end
    h = (b - a) / n
    s = f(a) + f(b)
    for i = 1:2:n-1
        s += 4 * f(a + i * h)
    end
    for i = 2:2:n-2
        s += 2 * f(a + i * h)
    end
    return s * h / 3
end

# Analytical electric field components for a 3D Gaussian charge distribution in free space (copied from runtests.jl)
function analytical_efield(x, y, z, Q, sigma_x, sigma_y, sigma_z)
    n_simpson = 200000 # Number of intervals for Simpson's rule. Adjust for desired precision.
    upper_bound = 500.0 # Integration upper bound. Adjust if integrand doesn't decay sufficiently.

    # Ex component
    integrand_Ex(lambda) = begin
        denom_x = lambda^2 * sigma_x^2 + 1
        denom_y = lambda^2 * sigma_y^2 + 1
        denom_z = lambda^2 * sigma_z^2 + 1

        exp_term_x = exp(-lambda^2 * x^2 / (2 * denom_x))
        exp_term_y = exp(-lambda^2 * y^2 / (2 * denom_y))
        exp_term_z = exp(-lambda^2 * z^2 / (2 * denom_z))

        sqrt_denom = sqrt(denom_x * denom_y * denom_z)

        return lambda^2 * exp_term_x * exp_term_y * exp_term_z / sqrt_denom * (x / denom_x)
    end
    Ex_integral = simpsons_rule(integrand_Ex, 0.0, upper_bound, n_simpson)
    Ex = Q / (4 * pi * EPSILON_0) * sqrt(2 / pi) * Ex_integral

    # Ey component
    integrand_Ey(lambda) = begin
        denom_x = lambda^2 * sigma_x^2 + 1
        denom_y = lambda^2 * sigma_y^2 + 1
        denom_z = lambda^2 * sigma_z^2 + 1

        exp_term_x = exp(-lambda^2 * x^2 / (2 * denom_x))
        exp_term_y = exp(-lambda^2 * y^2 / (2 * denom_y))
        exp_term_z = exp(-lambda^2 * z^2 / (2 * denom_z))

        sqrt_denom = sqrt(denom_x * denom_y * denom_z)

        return lambda^2 * exp_term_x * exp_term_y * exp_term_z / sqrt_denom * (y / denom_y)
    end
    Ey_integral = simpsons_rule(integrand_Ey, 0.0, upper_bound, n_simpson)
    Ey = Q / (4 * pi * EPSILON_0) * sqrt(2 / pi) * Ey_integral

    # Ez component
    integrand_Ez(lambda) = begin
        denom_x = lambda^2 * sigma_x^2 + 1
        denom_y = lambda^2 * sigma_y^2 + 1
        denom_z = lambda^2 * sigma_z^2 + 1

        exp_term_x = exp(-lambda^2 * x^2 / (2 * denom_x))
        exp_term_y = exp(-lambda^2 * y^2 / (2 * denom_y))
        exp_term_z = exp(-lambda^2 * z^2 / (2 * denom_z))

        sqrt_denom = sqrt(denom_x * denom_y * denom_z)

        return lambda^2 * exp_term_x * exp_term_y * exp_term_z / sqrt_denom * (z / denom_z)
    end
    Ez_integral = simpsons_rule(integrand_Ez, 0.0, upper_bound, n_simpson)
    Ez = Q / (4 * pi * EPSILON_0) * sqrt(2 / pi) * Ez_integral

    return Ex, Ey, Ez
end

function main()
    # Setup mesh and particles
    grid_size = (16, 16, 16)
    min_bounds = (-0.05, -0.05, -0.05)
    max_bounds = (0.05, 0.05, 0.05)
    mesh = SpaceCharge.Mesh3D(grid_size, min_bounds, max_bounds)

    num_particles = 10000
    total_charge = 1.0
    charge_per_particle = total_charge / num_particles

    sigma_x = 0.01
    sigma_y = 0.01
    sigma_z = 0.01

    Random.seed!(123)
    particles_x = randn(num_particles) .* sigma_x
    particles_y = randn(num_particles) .* sigma_y
    particles_z = randn(num_particles) .* sigma_z
    particles_q = fill(charge_per_particle, num_particles)

    SpaceCharge.deposit!(mesh, particles_x, particles_y, particles_z, particles_q)
    SpaceCharge.solve!(mesh, SpaceCharge.FreeSpace())

    # Extract data for plotting along z-axis (x=0, y=0)
    z_coords = [mesh.min_bounds[3] + (k - 1) * mesh.delta[3] for k in 1:grid_size[3]]
    
    # Find the indices closest to x=0 and y=0
    # Assuming the grid is centered or includes 0
    x_center_idx = argmin(abs.([mesh.min_bounds[1] + (i - 1) * mesh.delta[1] for i in 1:grid_size[1]]))
    y_center_idx = argmin(abs.([mesh.min_bounds[2] + (j - 1) * mesh.delta[2] for j in 1:grid_size[2]]))

    computed_Ez = [mesh.efield[x_center_idx, y_center_idx, k, 3] for k in 1:grid_size[3]]
    analytical_Ez = [analytical_efield(0.0, 0.0, z, total_charge, sigma_x, sigma_y, sigma_z)[3] for z in z_coords]

    # Plotting
    gr()
    plot(z_coords, computed_Ez, label="Computed Ez", xlabel="Z-coordinate (m)", ylabel="Ez Field (V/m)",
         title="Ez Field Comparison along Centerline (x=0, y=0)", linewidth=2)
    savefig("examples/Ez_computed.png")
    plot(z_coords, analytical_Ez, label="Analytical Ez", linestyle=:dash, linewidth=2)
    savefig("examples/Ez_analytical.png")

    println("Plot saved to Ez_computed.png and Ez_analytical.png")
end

main()
