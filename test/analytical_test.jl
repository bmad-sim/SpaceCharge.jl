using SpaceCharge
using Test
using Random
using LinearAlgebra
using SpecialFunctions

@testset "Analytical Comparison" begin
    # Constants
    EPSILON_0 = 8.8541878128e-12 # Permittivity of free space

    # Exact solution for isotropic Gaussian bunch
    function E_analytical(r;Q=1,σ=1)
        if norm(r) == 0
            return [0.0, 0.0, 0.0]
        end
        return Q/(4π*EPSILON_0*norm(r)^3)*(erf(norm(r)/(sqrt(2)*σ)) - sqrt(2/π)*norm(r)/σ*exp(-(norm(r)/σ)^2/2)).*r
    end

    # Setup particles first
    num_particles = 1000000 # Fewer particles for faster testing
    total_charge = 1.0e-9
    charge_per_particle = total_charge / num_particles

    sigma = 0.001

    Random.seed!(123)
    particles_x = randn(num_particles) .* sigma
    particles_y = randn(num_particles) .* sigma
    particles_z = randn(num_particles) .* sigma
    particles_q = fill(charge_per_particle, num_particles)

    grid_size = (32, 32, 32)
    mesh = SpaceCharge.Mesh3D(grid_size, particles_x, particles_y, particles_z, total_charge=total_charge)

    SpaceCharge.deposit!(mesh, particles_x, particles_y, particles_z, particles_q)
    SpaceCharge.solve!(mesh, SpaceCharge.FreeSpace())

    # Extract data for comparison along z-axis (x=0, y=0)
    z_coords = [mesh.min_bounds[3] + (k - 1) * mesh.delta[3] for k in 1:grid_size[3]]

    # Find the indices closest to x=0 and y=0
    x_center_idx = argmin(abs.([mesh.min_bounds[1] + (i - 1) * mesh.delta[1] for i in 1:grid_size[1]]))
    y_center_idx = argmin(abs.([mesh.min_bounds[2] + (j - 1) * mesh.delta[2] for j in 1:grid_size[2]]))

    computed_Ez = [mesh.efield[x_center_idx, y_center_idx, k, 3] for k in 1:grid_size[3]]
    analytical_Ez = [E_analytical([0.0, 0.0, z], Q=total_charge, σ=sigma)[3] for z in z_coords]

    # Maximum absolute error
    max_abs_error = maximum(abs.(computed_Ez .- analytical_Ez)) / maximum(abs.(analytical_Ez))
    @test max_abs_error < 0.10

end
