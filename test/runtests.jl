using SpaceCharge
using Test
using CUDA
using KernelAbstractions
using FFTW # For plan_fft
# using QuadGK # Removed for Simpson's rule

# Constants
const EPSILON_0 = 8.8541878128e-12 # Permittivity of free space

# Simpson's Rule Integrator
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

# Analytical potential for a 3D Gaussian charge distribution in free space
# (This function is not used in the test, but kept for completeness if needed later)
function analytical_potential(x, y, z, Q, sigma_x, sigma_y, sigma_z)
    integrand(lambda) = begin
        denom_x = lambda^2 * sigma_x^2 + 1
        denom_y = lambda^2 * sigma_y^2 + 1
        denom_z = lambda^2 * sigma_z^2 + 1

        exp_term_x = exp(-lambda^2 * x^2 / (2 * denom_x))
        exp_term_y = exp(-lambda^2 * y^2 / (2 * denom_y))
        exp_term_z = exp(-lambda^2 * z^2 / (2 * denom_z))

        sqrt_denom = sqrt(denom_x * denom_y * denom_z)

        return exp_term_x * exp_term_y * exp_term_z / sqrt_denom
    end

    # Numerical integration from 0 to infinity. Choose a sufficiently large upper bound.
    # The integrand decays rapidly for large lambda.
    # Using a fixed number of points for Simpson's rule. Adjust 'n_simpson' for desired precision.
    n_simpson = 10000 # Number of intervals for Simpson's rule
    integral_val = simpsons_rule(integrand, 0.0, 100.0, n_simpson) # Integrate up to 100, adjust if needed

    return Q / (4 * pi * EPSILON_0) * sqrt(2 / pi) * integral_val
end

# Analytical electric field components for a 3D Gaussian charge distribution in free space
function analytical_efield(x, y, z, Q, sigma_x, sigma_y, sigma_z)
    n_simpson = 100000 # Number of intervals for Simpson's rule. Adjust for desired precision.
    upper_bound = 200.0 # Integration upper bound. Adjust if integrand doesn't decay sufficiently.

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


@testset "SpaceCharge.jl" begin
    # Test Mesh3D constructor
    @testset "Mesh3D Constructor" begin
        grid_size = (10, 20, 30)
        min_bounds = (-1.0, -2.0, -3.0)
        max_bounds = (1.0, 2.0, 3.0)
        mesh = SpaceCharge.Mesh3D(grid_size, min_bounds, max_bounds)

        @test mesh.grid_size == grid_size
        @test mesh.min_bounds == min_bounds
        @test mesh.max_bounds == max_bounds
        @test mesh.delta[1] ≈ (max_bounds[1] - min_bounds[1]) / (grid_size[1] - 1)
        @test mesh.delta[2] ≈ (max_bounds[2] - min_bounds[2]) / (grid_size[2] - 1)
        @test mesh.delta[3] ≈ (max_bounds[3] - min_bounds[3]) / (grid_size[3] - 1)
        @test eltype(mesh.rho) == Float64
        @test typeof(mesh.rho) == Array{Float64, 3}
        @test size(mesh.rho) == grid_size
        @test size(mesh.efield) == (grid_size..., 3)

        # Test with different float type and array type
        if CUDA.functional()
            mesh_gpu = SpaceCharge.Mesh3D(grid_size, min_bounds, max_bounds; T=Float32, array_type=CuArray)
            @test eltype(mesh_gpu.rho) == Float32
            @test typeof(mesh_gpu.rho) == CuArray{Float32, 3}
        end

        # Test validation
        @test_throws ErrorException SpaceCharge.Mesh3D((0, 1, 1), min_bounds, max_bounds)
        @test_throws ErrorException SpaceCharge.Mesh3D((1, 1, 1), (0,0,0), (0,0,0))
    end

    # Test deposit! and interpolate_field
    @testset "Deposit and Interpolate" begin
        grid_size = (2, 2, 2)
        min_bounds = (0.0, 0.0, 0.0)
        max_bounds = (1.0, 1.0, 1.0)
        mesh = SpaceCharge.Mesh3D(grid_size, min_bounds, max_bounds)

        # Single particle at center of a cell
        particles_x = [0.25]
        particles_y = [0.25]
        particles_z = [0.25]
        particles_q = [1.0]

        SpaceCharge.deposit!(mesh, particles_x, particles_y, particles_z, particles_q)

        # For a single particle at (0.25, 0.25, 0.25) in a (0,0,0)-(1,1,1) grid with size (2,2,2)
        # delta = (1.0, 1.0, 1.0)
        # norm_x = 0.25, norm_y = 0.25, norm_z = 0.25
        # ix = 0, iy = 0, iz = 0
        # dx = 0.25, dy = 0.25, dz = 0.25
        # The particle is in the first cell (indices 1,1,1)
        # It should deposit primarily to (1,1,1) and its neighbors

        # Check total charge conservation (approximately)
        @test sum(mesh.rho) ≈ sum(particles_q)

        # Simple check for non-zero values where charge should be deposited
        @test mesh.rho[1, 1, 1] > 0
        @test mesh.rho[2, 1, 1] > 0
        @test mesh.rho[1, 2, 1] > 0
        @test mesh.rho[1, 1, 2] > 0

        # Set some dummy field values for interpolation test
        mesh.efield[1,1,1,1] = 1.0 # Ex at (0,0,0)
        mesh.efield[2,1,1,1] = 2.0 # Ex at (1,0,0)
        mesh.efield[1,2,1,1] = 3.0 # Ex at (0,1,0)
        mesh.efield[1,1,2,1] = 4.0 # Ex at (0,0,1)

        # Interpolate at the particle position
        Ex, Ey, Ez, Bx, By, Bz = SpaceCharge.interpolate_field(mesh, particles_x, particles_y, particles_z)

        # For a single particle at (0.25, 0.25, 0.25) and a 2x2x2 grid
        # The interpolation should be a weighted sum of the 8 surrounding points.
        # Since we only set Ex at (1,1,1,1), (2,1,1,1), (1,2,1,1), (1,1,2,1) and others are zero
        # The interpolated Ex should be a weighted sum of these.
        # w_000 = (0.75)*(0.75)*(0.75) = 0.421875
        # w_100 = (0.25)*(0.75)*(0.75) = 0.140625
        # w_010 = (0.75)*(0.25)*(0.75) = 0.140625
        # w_001 = (0.75)*(0.75)*(0.25) = 0.140625

        expected_Ex = (
            mesh.efield[1,1,1,1] * (0.75*0.75*0.75) +
            mesh.efield[2,1,1,1] * (0.25*0.75*0.75) +
            mesh.efield[1,2,1,1] * (0.75*0.25*0.75) +
            mesh.efield[2,2,1,1] * (0.25*0.25*0.75) +
            mesh.efield[1,1,2,1] * (0.75*0.75*0.25) +
            mesh.efield[2,1,2,1] * (0.25*0.75*0.25) +
            mesh.efield[1,2,2,1] * (0.75*0.25*0.25) +
            mesh.efield[2,2,2,1] * (0.25*0.25*0.25)
        )
        # Since only the first 4 are set, and others are 0
        expected_Ex_simplified = (
            1.0 * (0.75*0.75*0.75) +
            2.0 * (0.25*0.75*0.75) +
            3.0 * (0.75*0.25*0.75) +
            0.0 * (0.25*0.25*0.75) +
            4.0 * (0.75*0.75*0.25) +
            0.0 * (0.25*0.75*0.25) +
            0.0 * (0.75*0.25*0.25) +
            0.0 * (0.25*0.25*0.25)
        )

        @test Ex[1] ≈ expected_Ex_simplified
        @test Ey[1] ≈ 0.0 # Should be zero as we didn't set Ey values
        @test Ez[1] ≈ 0.0 # Should be zero as we didn't set Ez values
        @test Bx[1] ≈ 0.0 # Should be zero as we didn't set Bx values
    end

    # Test generate_igf_kernel! to be nonzero. Needs more quantitative test.
    @testset "generate_igf_kernel!" begin
        grid_size = (4, 4, 4)
        delta = (0.1, 0.1, 0.1)
        green_function_array = zeros(Float64, grid_size)
        backend = CPU()
        kernel! = SpaceCharge.generate_igf_kernel!(backend)

        # Test for Ex component (component_idx = 1)
        kernel!(green_function_array, grid_size, delta, 1, 1.0, ndrange=grid_size)
        @test any(green_function_array .!= 0.0) # Ensure some values are calculated

        # Test for Ey component (component_idx = 2)
        fill!(green_function_array, 0.0) # Reset array
        kernel!(green_function_array, grid_size, delta, 2, 1.0, ndrange=grid_size)
        @test any(green_function_array .!= 0.0)

        # Test for Ez component (component_idx = 3)
        fill!(green_function_array, 0.0) # Reset array
        kernel!(green_function_array, grid_size, delta, 3, 1.0, ndrange=grid_size)
        @test any(green_function_array .!= 0.0)
    end

    # Test FreeSpace Solver. Needs quantitative test.
    @testset "FreeSpace Solver" begin
        grid_size = (16, 16, 16) # Larger grid for more realistic test
        min_bounds = (-0.1, -0.1, -0.1)
        max_bounds = (0.1, 0.1, 0.1)
        mesh = SpaceCharge.Mesh3D(grid_size, min_bounds, max_bounds)

        # Place a single charge at the center of the mesh
        particles_x = [0.0]
        particles_y = [0.0]
        particles_z = [0.0]
        particles_q = [1.0]

        # Deposit charge
        SpaceCharge.deposit!(mesh, particles_x, particles_y, particles_z, particles_q)

        # Solve for fields
        SpaceCharge.solve!(mesh, SpaceCharge.FreeSpace())

        # More rigorous tests comparing against analytical solutions or Fortran output will be added here.

        # For a single point charge at the center, E-field should be radial
        # and B-field should be azimuthal (if gamma > 1). For gamma=1, B-field is 0.
        # More rigorous tests comparing against analytical solutions or Fortran output will be added here.

        # Check that the sum of rho is conserved (deposit! should handle this)
        @test sum(mesh.rho) ≈ sum(particles_q)
    end

    # Analytical Test for Gaussian Charge Distribution in Free Space
    @testset "Gaussian Free Space Analytical Test" begin
        grid_size = (16, 16, 16)
        min_bounds = (-0.05, -0.05, -0.05) # Smaller bounds to contain the Gaussian
        max_bounds = (0.05, 0.05, 0.05)
        mesh = SpaceCharge.Mesh3D(grid_size, min_bounds, max_bounds)

        num_particles = 10000
        total_charge = 1.0 # Total charge of the Gaussian
        charge_per_particle = total_charge / num_particles

        # Gaussian parameters (standard deviations)
        sigma_x = 0.01
        sigma_y = 0.01
        sigma_z = 0.01

        # Generate particles with a Gaussian distribution
        particles_x = randn(num_particles) .* sigma_x
        particles_y = randn(num_particles) .* sigma_y
        particles_z = randn(num_particles) .* sigma_z
        particles_q = fill(charge_per_particle, num_particles)

        # Deposit charge onto the mesh
        SpaceCharge.deposit!(mesh, particles_x, particles_y, particles_z, particles_q)

        # Solve for fields using SpaceCharge.jl
        SpaceCharge.solve!(mesh, SpaceCharge.FreeSpace())

        # Calculate analytical electric field at each mesh point
        analytical_efield_x = zeros(grid_size)
        analytical_efield_y = zeros(grid_size)
        analytical_efield_z = zeros(grid_size)

        for k in 1:grid_size[3]
            z_coord = mesh.min_bounds[3] + (k - 1) * mesh.delta[3]
            for j in 1:grid_size[2]
                y_coord = mesh.min_bounds[2] + (j - 1) * mesh.delta[2]
                for i in 1:grid_size[1]
                    x_coord = mesh.min_bounds[1] + (i - 1) * mesh.delta[1]
                    
                    Ex_analytical, Ey_analytical, Ez_analytical = analytical_efield(
                        x_coord, y_coord, z_coord, total_charge, sigma_x, sigma_y, sigma_z
                    )
                    analytical_efield_x[i, j, k] = Ex_analytical
                    analytical_efield_y[i, j, k] = Ey_analytical
                    analytical_efield_z[i, j, k] = Ez_analytical
                end
            end
        end

        # Compare computed field with analytical field
        # Use a relative tolerance for comparison
        rtol = 0.05 # 5% relative tolerance, adjust as needed

        # Ex
        diff_Ex = abs.(mesh.efield[:, :, :, 1] .- analytical_efield_x)
        max_diff_Ex = maximum(diff_Ex)
        max_analytical_Ex = maximum(abs.(analytical_efield_x))
        @test max_diff_Ex / max_analytical_Ex < rtol

        # Ey
        diff_Ey = abs.(mesh.efield[:, :, :, 2] .- analytical_efield_y)
        max_diff_Ey = maximum(diff_Ey)
        max_analytical_Ey = maximum(abs.(analytical_efield_y))
        @test max_diff_Ey / max_analytical_Ey < rtol

        # Ez
        diff_Ez = abs.(mesh.efield[:, :, :, 3] .- analytical_efield_z)
        max_diff_Ez = maximum(diff_Ez)
        max_analytical_Ez = maximum(abs.(analytical_efield_z))
        @test max_diff_Ez / max_analytical_Ez < rtol

        # Also check total charge conservation
        @test sum(mesh.rho) ≈ total_charge
    end

    # Test Cathode Solver. Only checks that FreeSpace field != Cathode field (due to image charge effect). Needs quantitative test.
    @testset "Cathode Solver" begin
        grid_size = (16, 16, 16)
        min_bounds = (-0.1, -0.1, 0.0) # Cathode at z=0
        max_bounds = (0.1, 0.1, 0.2)
        mesh = SpaceCharge.Mesh3D(grid_size, min_bounds, max_bounds; gamma=2.0)

        # Place a single charge near the cathode
        particles_x = [0.0]
        particles_y = [0.0]
        particles_z = [0.01]
        particles_q = [1.0]

        # Deposit charge
        SpaceCharge.deposit!(mesh, particles_x, particles_y, particles_z, particles_q)

        # Store initial rho for comparison
        initial_rho = deepcopy(mesh.rho)

        # Solve for fields with cathode boundary condition
        SpaceCharge.solve!(mesh, SpaceCharge.FreeSpace(); at_cathode=true)

        # More rigorous tests comparing against analytical solutions or Fortran output will be added here.

        # Basic check for image charge effect: E-field should be different from free-space
        # More rigorous tests comparing against analytical solutions or Fortran output will be added here.
        mesh_free_space = SpaceCharge.Mesh3D(grid_size, min_bounds, max_bounds)
        SpaceCharge.deposit!(mesh_free_space, particles_x, particles_y, particles_z, particles_q)
        SpaceCharge.solve!(mesh_free_space, SpaceCharge.FreeSpace())

        @test !isapprox(mesh.efield, mesh_free_space.efield)
        @test !isapprox(mesh.bfield, mesh_free_space.bfield)

        # Verify image charge distribution (qualitative check)
        # The image charge should be negative and flipped in z
        # This is hard to test directly without exposing internal image_rho
        # but we can infer from the field differences.
    end

    # Test RectangularPipe Solver. Needs quantitative test.
    @testset "RectangularPipe Solver" begin
        grid_size = (16, 16, 16)
        min_bounds = (-0.1, -0.1, -0.1)
        max_bounds = (0.1, 0.1, 0.1)
        mesh = SpaceCharge.Mesh3D(grid_size, min_bounds, max_bounds)

        # Place a single charge at the center of the mesh
        particles_x = [0.0]
        particles_y = [0.0]
        particles_z = [0.0]
        particles_q = [1.0]

        # Deposit charge
        SpaceCharge.deposit!(mesh, particles_x, particles_y, particles_z, particles_q)

        # Solve for fields
        SpaceCharge.solve!(mesh, SpaceCharge.RectangularPipe())

        # More rigorous tests comparing against analytical solutions or Fortran output will be added here.
    end

    # Test GPU functionality if CUDA is available
    if CUDA.functional()
        @testset "GPU Functionality" begin
            grid_size = (10, 10, 10)
            min_bounds = (0.0, 0.0, 0.0)
            max_bounds = (1.0, 1.0, 1.0)
            mesh_gpu = SpaceCharge.Mesh3D(grid_size, min_bounds, max_bounds; array_type=CuArray)

            particles_x_gpu = CuArray([0.5])
            particles_y_gpu = CuArray([0.5])
            particles_z_gpu = CuArray([0.5])
            particles_q_gpu = CuArray([1.0])

            SpaceCharge.deposit!(mesh_gpu, particles_x_gpu, particles_y_gpu, particles_z_gpu, particles_q_gpu)
            @test sum(mesh_gpu.rho) ≈ sum(particles_q_gpu)

            # Set dummy field values on GPU for interpolation test
            mesh_gpu.efield[1,1,1,1] = 1.0f0
            mesh_gpu.efield[2,1,1,1] = 2.0f0
            mesh_gpu.efield[1,2,1,1] = 3.0f0
            mesh_gpu.efield[1,1,2,1] = 4.0f0

            Ex_gpu, Ey_gpu, Ez_gpu, Bx_gpu, By_gpu, Bz_gpu = SpaceCharge.interpolate_field(mesh_gpu, particles_x_gpu, particles_y_gpu, particles_z_gpu)

            expected_Ex_gpu = (
                mesh_gpu.efield[1,1,1,1] * (0.75f0*0.75f0*0.75f0) +
                mesh_gpu.efield[2,1,1,1] * (0.25f0*0.75f0*0.75f0) +
                mesh_gpu.efield[1,2,1,1] * (0.75f0*0.25f0*0.75f0) +
                mesh_gpu.efield[2,2,1,1] * (0.25f0*0.25f0*0.75f0) +
                mesh_gpu.efield[1,1,2,1] * (0.75f0*0.75f0*0.25f0) +
                mesh_gpu.efield[2,1,2,1] * (0.25f0*0.75f0*0.25f0) +
                mesh_gpu.efield[1,2,2,1] * (0.75f0*0.25f0*0.25f0) +
                mesh_gpu.efield[2,2,2,1] * (0.25f0*0.25f0*0.25f0)
            )
            expected_Ex_gpu_simplified = (
                1.0f0 * (0.75f0*0.75f0*0.75f0) +
                2.0f0 * (0.25f0*0.75f0*0.75f0) +
                3.0f0 * (0.75f0*0.25f0*0.75f0) +
                0.0f0 * (0.25f0*0.25f0*0.75f0) +
                4.0f0 * (0.75f0*0.75f0*0.25f0) +
                0.0f0 * (0.25f0*0.75f0*0.25f0) +
                0.0f0 * (0.75f0*0.25f0*0.25f0) +
                0.0f0 * (0.25f0*0.25f0*0.25f0)
            )

            @test Ex_gpu[1] ≈ expected_Ex_gpu_simplified
            @test Ey_gpu[1] ≈ 0.0f0
        end
    end
end
