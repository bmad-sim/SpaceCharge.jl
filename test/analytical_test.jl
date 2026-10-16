using SpaceCharge
using Test
using Random
using LinearAlgebra
using SpecialFunctions
using AtomicAndPhysicalConstants: EPS_0, E_CHARGE

@testset "Analytical Comparison" begin
    # Constants
    EPSILON_0 = EPS_0 * E_CHARGE # SI permittivity in F/m

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
    sc = FFTIGF(grid_size; fft_flags=FFTW.ESTIMATE)
    solve!(sc, particles_x, particles_y, particles_z, particles_q)
    # Independent evaluation points along the physical axis, in metres.
    z_coords = collect(range(-0.004, 0.004; length=101))
    zeros_xy = zeros(length(z_coords))
    outputs = ntuple(_ -> similar(z_coords), 3)
    field!(outputs..., sc, zeros_xy, zeros_xy, z_coords)
    computed_Ez = outputs[3]
    analytical_Ez = [E_analytical([0.0, 0.0, z], Q=total_charge, σ=sigma)[3] for z in z_coords]

    # Maximum absolute error
    max_abs_error = maximum(abs.(computed_Ez .- analytical_Ez)) / maximum(abs.(analytical_Ez))
    @test max_abs_error < 0.10

end
