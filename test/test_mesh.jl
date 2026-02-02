using SpaceCharge
using Test
using CUDA

function run_mesh_tests()
    @testset "Mesh3D Constructor" begin
        # Test particle-based constructor (recommended)
        @testset "Particle-based Constructor" begin
            grid_size = (10, 20, 30)
            particles_x = [-1.0, 1.0]
            particles_y = [-2.0, 2.0]
            particles_z = [-3.0, 3.0]
            
            mesh = Mesh3D(grid_size, particles_x, particles_y, particles_z)

            @test mesh.grid_size == grid_size
            @test eltype(mesh.rho) == Float64
            @test typeof(mesh.rho) == Array{Float64, 3}
            @test size(mesh.rho) == grid_size
            @test size(mesh.efield) == (grid_size..., 3)
            
            # Check that bounds contain all particles with padding
            @test mesh.min_bounds[1] < minimum(particles_x)
            @test mesh.max_bounds[1] > maximum(particles_x)
            @test mesh.min_bounds[2] < minimum(particles_y)
            @test mesh.max_bounds[2] > maximum(particles_y)
            @test mesh.min_bounds[3] < minimum(particles_z)
            @test mesh.max_bounds[3] > maximum(particles_z)
            
            # Check delta calculation
            @test mesh.delta[1] ≈ (mesh.max_bounds[1] - mesh.min_bounds[1]) / (grid_size[1] - 1)
            @test mesh.delta[2] ≈ (mesh.max_bounds[2] - mesh.min_bounds[2]) / (grid_size[2] - 1)
            @test mesh.delta[3] ≈ (mesh.max_bounds[3] - mesh.min_bounds[3]) / (grid_size[3] - 1)
        end

        # Test manual bounds constructor (legacy)
        @testset "Manual Bounds Constructor" begin
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
        end

        # Test with different float type and array type
        @testset "Type Parameters" begin
            grid_size = (5, 5, 5)
            particles_x = [0.0]
            particles_y = [0.0]
            particles_z = [0.0]
            
            mesh_float32 = Mesh3D(grid_size, particles_x, particles_y, particles_z; T=Float32)
            @test eltype(mesh_float32.rho) == Float32
            @test eltype(mesh_float32.efield) == Float32
            
            if CUDA.functional()
                mesh_gpu = Mesh3D(grid_size, particles_x, particles_y, particles_z; 
                                 T=Float32, backend=CUDABackend())
                @test eltype(mesh_gpu.rho) == Float32
                @test mesh_gpu.rho isa CuArray{Float32, 3}
                @test mesh_gpu.efield isa CuArray{Float32, 4}
            end
        end

        # Test validation
        @testset "Validation" begin
            # Invalid grid size (must be at least 2 in each dimension)
            @test_throws ErrorException Mesh3D((0, 2, 2), [0.0], [0.0], [0.0])
            @test_throws ErrorException Mesh3D((2, 0, 2), [0.0], [0.0], [0.0])
            @test_throws ErrorException Mesh3D((2, 2, 0), [0.0], [0.0], [0.0])
            @test_throws ErrorException Mesh3D((1, 2, 2), [0.0], [0.0], [0.0])
            @test_throws ErrorException Mesh3D((2, 1, 2), [0.0], [0.0], [0.0])
            @test_throws ErrorException Mesh3D((2, 2, 1), [0.0], [0.0], [0.0])

            # Empty particle arrays
            @test_throws ErrorException Mesh3D((2, 2, 2), Float64[], Float64[], Float64[])

            # Mismatched particle array lengths
            @test_throws ErrorException Mesh3D((4, 4, 4), [0.0, 1.0], [0.0], [0.0])

            # Invalid bounds (for manual constructor)
            @test_throws ErrorException Mesh3D((2, 2, 2), (0,0,0), (0,0,0))
            @test_throws ErrorException Mesh3D((2, 2, 2), (1,1,1), (0,0,0))
        end

        # Test physics parameters
        @testset "Physics Parameters" begin
            grid_size = (5, 5, 5)
            particles_x = [0.0]
            particles_y = [0.0]
            particles_z = [0.0]
            
            mesh = Mesh3D(grid_size, particles_x, particles_y, particles_z; 
                         gamma=2.0, total_charge=1.0)
            
            @test mesh.gamma == 2.0
            @test mesh.total_charge == 1.0
        end
    end
end 