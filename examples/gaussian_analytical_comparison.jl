using SpaceCharge
using Plots
using Random
using SpecialFunctions: erf
using LinearAlgebra: norm

using AtomicAndPhysicalConstants: EPS_0, E_CHARGE

# Constants
const EPSILON_0 = EPS_0 * E_CHARGE # SI permittivity in F/m

# Exact solution for isotropic Gaussian bunch
function E(r;Q=1,σ=1)
    iszero(norm(r)) && return zero(r)
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

    sc = FFTIGF(grid_size)
    solve!(sc, particles_x, particles_y, particles_z, particles_q, 1)
    # Keep evaluation points inside the fitted domain.
    z_coords = collect(range(max(sc.min_bounds[3],-0.004),min(sc.max_bounds[3],0.004);length=101))
    xy = zeros(length(z_coords))
    fields = ntuple(_->similar(z_coords),3)
    field!(fields...,sc,xy,xy,z_coords)
    computed_Ez = fields[3]
    analytical_Ez = [E([0.0, 0.0, z],Q=1e-9,σ=1e-3)[3] for z in z_coords]


    println("Maximum absolute error: ", maximum(abs.(computed_Ez .- analytical_Ez)) / maximum(abs.(analytical_Ez)))
    println("Index of maximum absolute error: ", argmax(abs.(computed_Ez .- analytical_Ez)))

    # Plotting
    gr()
    plot(z_coords, computed_Ez, label="Computed Ez", xlabel="Z-coordinate (m)", ylabel="Ez Field (V/m)",
         title="Ez Field along Centerline of isotropic 1nC bunch (x=0, y=0)", linewidth=2)
    plot!(z_coords, analytical_Ez, label="Analytical Ez", linestyle=:dash, linewidth=2)
    savefig(joinpath(@__DIR__,"Ez_comparison.png"))

    println("Plot saved to Ez_comparison.png")
end

main()
