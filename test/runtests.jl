using SpaceCharge
using Test
using CUDA
using KernelAbstractions
using FFTW

@testset "SpaceCharge.jl" begin
    # Test Mesh3D constructor
    @testset "Mesh3D Constructor" begin
        grid_size = (10, 20, 30)
        min_bounds = (-1.0, -2.0, -3.0)
        max_bounds = (1.0, 2.0, 3.0)
        mesh = Mesh3D(grid_size, min_bounds, max_bounds)

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
            mesh_gpu = Mesh3D(grid_size, min_bounds, max_bounds; T=Float32, array_type=CuArray)
            @test eltype(mesh_gpu.rho) == Float32
            @test mesh_gpu.rho isa CuArray{Float32, 3}
        end

        # Test validation
        @test_throws ErrorException Mesh3D((0, 1, 1), min_bounds, max_bounds)
        @test_throws ErrorException Mesh3D((1, 1, 1), (0,0,0), (0,0,0))
    end

    # Test deposit! and interpolate_field
    @testset "Deposit and Interpolate" begin
        grid_size = (2, 2, 2)
        min_bounds = (0.0, 0.0, 0.0)
        max_bounds = (1.0, 1.0, 1.0)
        mesh = Mesh3D(grid_size, min_bounds, max_bounds)

        # Single particle at center of a cell
        particles_x = [0.25]
        particles_y = [0.25]
        particles_z = [0.25]
        particles_q = [1.0]

        deposit!(mesh, particles_x, particles_y, particles_z, particles_q)

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
        Ex, Ey, Ez, Bx, By, Bz = interpolate_field(mesh, particles_x, particles_y, particles_z)

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

    # Test FreeSpace Solver. Needs quantitative test.
    @testset "FreeSpace Solver" begin
        grid_size = (16, 16, 16) # Larger grid for more realistic test
        min_bounds = (-0.1, -0.1, -0.1)
        max_bounds = (0.1, 0.1, 0.1)
        mesh = Mesh3D(grid_size, min_bounds, max_bounds)

        # Place a single charge at the center of the mesh
        particles_x = [0.0]
        particles_y = [0.0]
        particles_z = [0.0]
        particles_q = [1.0]

        # Deposit charge
        deposit!(mesh, particles_x, particles_y, particles_z, particles_q)

        # Solve for fields
        solve!(mesh, FreeSpace())

        # Check that the sum of rho is conserved (deposit! should handle this)
        @test sum(mesh.rho) ≈ sum(particles_q)
    end

    # Test Cathode Solver.
    @testset "Cathode Solver" begin
        grid_size = (16, 16, 16)
        min_bounds = (-0.1, -0.1, 0.0) # Cathode at z=0
        max_bounds = (0.1, 0.1, 0.2)
        mesh = Mesh3D(grid_size, min_bounds, max_bounds; gamma=2.0)

        particles_x = [0.0]
        particles_y = [0.0]
        particles_z = [0.01]
        particles_q = [1.0]

        deposit!(mesh, particles_x, particles_y, particles_z, particles_q)
        solve!(mesh, FreeSpace(); at_cathode=true)

        mesh_free_space = Mesh3D(grid_size, min_bounds, max_bounds)
        deposit!(mesh_free_space, particles_x, particles_y, particles_z, particles_q)
        solve!(mesh_free_space, FreeSpace())

        @test !isapprox(mesh.efield, mesh_free_space.efield)
        @test !isapprox(mesh.bfield, mesh_free_space.bfield)
    end

    # Test RectangularPipe Solver.
    @testset "RectangularPipe Solver" begin
        grid_size = (16, 16, 16)
        min_bounds = (-0.1, -0.1, -0.1)
        max_bounds = (0.1, 0.1, 0.1)
        mesh = Mesh3D(grid_size, min_bounds, max_bounds)

        particles_x = [0.0]
        particles_y = [0.0]
        particles_z = [0.0]
        particles_q = [1.0]

        deposit!(mesh, particles_x, particles_y, particles_z, particles_q)
        solve!(mesh, RectangularPipe())
        
        @test any(mesh.efield .!= 0.0)
    end

    # Test GPU functionality if CUDA is available
    if CUDA.functional()
        @testset "GPU Functionality" begin
            grid_size = (10, 10, 10)
            min_bounds = (0.0, 0.0, 0.0)
            max_bounds = (1.0, 1.0, 1.0)
            mesh_gpu = Mesh3D(grid_size, min_bounds, max_bounds; T=Float32, array_type=CuArray)

            particles_x_gpu = CuArray([0.5f0])
            particles_y_gpu = CuArray([0.5f0])
            particles_z_gpu = CuArray([0.5f0])
            particles_q_gpu = CuArray([1.0f0])

            deposit!(mesh_gpu, particles_x_gpu, particles_y_gpu, particles_z_gpu, particles_q_gpu)
            @test sum(Array(mesh_gpu.rho)) ≈ sum(Array(particles_q_gpu))

            Ex_gpu, Ey_gpu, Ez_gpu, Bx_gpu, By_gpu, Bz_gpu = interpolate_field(mesh_gpu, particles_x_gpu, particles_y_gpu, particles_z_gpu)

            @test Ex_gpu isa CuArray{Float32, 1}
            @test Ey_gpu isa CuArray{Float32, 1}
            @test Ez_gpu isa CuArray{Float32, 1}
            @test Bx_gpu isa CuArray{Float32, 1}
            @test By_gpu isa CuArray{Float32, 1}
            @test Bz_gpu isa CuArray{Float32, 1}
        end
    end
end

