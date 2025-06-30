using SpaceCharge
using Test
using CUDA
using KernelAbstractions
using FFTW # For plan_fft

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

    # Test generate_igf_kernel!
    @testset "generate_igf_kernel!" begin
        grid_size = (4, 4, 4)
        delta = (0.1, 0.1, 0.1)
        green_function_array = zeros(Float64, grid_size)
        backend = CPU()
        kernel! = SpaceCharge.generate_igf_kernel!(backend)

        # Test for Ex component (component_idx = 1)
        kernel!(green_function_array, grid_size, delta, 1, ndrange=grid_size)
        @test any(green_function_array .!= 0.0) # Ensure some values are calculated

        # Test for Ey component (component_idx = 2)
        fill!(green_function_array, 0.0) # Reset array
        kernel!(green_function_array, grid_size, delta, 2, ndrange=grid_size)
        @test any(green_function_array .!= 0.0)

        # Test for Ez component (component_idx = 3)
        fill!(green_function_array, 0.0) # Reset array
        kernel!(green_function_array, grid_size, delta, 3, ndrange=grid_size)
        @test any(green_function_array .!= 0.0)
    end

    # Test FreeSpace Solver
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

        # Verify that efield and bfield are populated (not all zeros)
        @test any(mesh.efield .!= 0.0)
        @test any(mesh.bfield .!= 0.0)

        # For a single point charge at the center, E-field should be radial
        # and B-field should be azimuthal (if gamma > 1). For gamma=1, B-field is 0.
        # With placeholder lafun2/xlafun2, we can't check exact values, but can check general properties.
        # For example, E-field should be strongest near the center.
        # This is a very basic check due to placeholder Green's functions.

        # Check that the sum of rho is conserved (deposit! should handle this)
        @test sum(mesh.rho) ≈ sum(particles_q)
    end

    # Test Cathode Solver
    @testset "Cathode Solver" begin
        grid_size = (16, 16, 16)
        min_bounds = (-0.1, -0.1, 0.0) # Cathode at z=0
        max_bounds = (0.1, 0.1, 0.2)
        mesh = SpaceCharge.Mesh3D(grid_size, min_bounds, max_bounds)

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

        # Verify that efield and bfield are populated (not all zeros)
        @test any(mesh.efield .!= 0.0)
        @test any(mesh.bfield .!= 0.0)

        # Basic check for image charge effect: E-field should be different from free-space
        # This is a very high-level check due to placeholder Green's functions.
        # A more rigorous test would compare against analytical solutions or Fortran output.
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

    # Test RectangularPipe Solver
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

        # Verify that efield and bfield are populated (not all zeros)
        @test any(mesh.efield .!= 0.0)
        @test any(mesh.bfield .!= 0.0)
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
