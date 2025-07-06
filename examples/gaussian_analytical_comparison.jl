using SpaceCharge
using Plots
using Random
using SpecialFunctions: erf
using LinearAlgebra: norm

# Constants
const EPSILON_0 = 8.8541878128e-12 # Permittivity of free space

# Exact solution for isotropic Gaussian bunch
function E(r;Q=1,σ=1)
    return Q/(4π*EPSILON_0*norm(r)^3)*(erf(norm(r)/(sqrt(2)*σ)) - sqrt(2/π)*norm(r)/σ*exp(-(norm(r)/σ)^2/2)).*r
end

function main()
    # Setup mesh and particles
    grid_size = (32, 32, 32)

    num_particles = 10000000
    total_charge = 1.0e-9
    charge_per_particle = total_charge / num_particles

    sigma_x = 0.001
    sigma_y = 0.001
    sigma_z = 0.001

    Random.seed!(123)
    particles_x = randn(num_particles) .* sigma_x
    particles_y = randn(num_particles) .* sigma_y
    particles_z = randn(num_particles) .* sigma_z
    particles_q = fill(charge_per_particle, num_particles)

    mesh = Mesh3D(grid_size, particles_x, particles_y, particles_z, total_charge=total_charge)
    deposit!(mesh, particles_x, particles_y, particles_z, particles_q)
    solve!(mesh)

    # Extract data for plotting along z-axis (x=0, y=0)
    z_coords = [mesh.min_bounds[3] + (k - 1) * mesh.delta[3] for k in 1:grid_size[3]]
    
    # Find the indices closest to x=0 and y=0
    # Assuming the grid is centered or includes 0
    x_center_idx = argmin(abs.([mesh.min_bounds[1] + (i - 1) * mesh.delta[1] for i in 1:grid_size[1]]))
    y_center_idx = argmin(abs.([mesh.min_bounds[2] + (j - 1) * mesh.delta[2] for j in 1:grid_size[2]]))

    computed_Ez = [mesh.efield[x_center_idx, y_center_idx, k, 3] for k in 1:grid_size[3]]
    analytical_Ez = [E([0.0, 0.0, z],Q=1e-9,σ=1e-3)[3] for z in z_coords]


    println("Maximum relative error: ", maximum(abs.(computed_Ez .- analytical_Ez) ./ abs.(analytical_Ez)))
    println("Maximum absolute error: ", maximum(abs.(computed_Ez .- analytical_Ez)) / maximum(abs.(analytical_Ez)))
    println("Index of maximum relative error: ", argmax(abs.(computed_Ez .- analytical_Ez) ./ abs.(analytical_Ez)))
    println("Index of maximum absolute error: ", argmax(abs.(computed_Ez .- analytical_Ez)))

    # Plotting
    gr()
    plot(z_coords, computed_Ez, label="Computed Ez", xlabel="Z-coordinate (m)", ylabel="Ez Field (V/m)",
         title="Ez Field along Centerline of isotropic 1nC bunch (x=0, y=0)", linewidth=2)
    plot!(z_coords, analytical_Ez, label="Analytical Ez", linestyle=:dash, linewidth=2)
    savefig("examples/Ez_comparison.png")

    println("Plot saved to Ez_comparison.png")
end

main()
