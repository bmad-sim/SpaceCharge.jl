using SpaceCharge
using Test
using CUDA

function run_gpu_tests()
    @testset "GPU Functionality" begin
        # Test GPU mesh creation
        @testset "GPU Mesh Creation" begin
            grid_size = (10, 10, 10)
            particles_x = [0.5]
            particles_y = [0.5]
            particles_z = [0.5]
            particles_q = [1.0]
            
            mesh_gpu = Mesh3D(grid_size, particles_x, particles_y, particles_z; 
                             T=Float32, backend=CUDABackend())
            
            @test eltype(mesh_gpu.rho) == Float32
            @test mesh_gpu.rho isa CuArray{Float32, 3}
            @test mesh_gpu.efield isa CuArray{Float32, 4}
        end

        # Test GPU deposition
        @testset "GPU Deposition" begin
            grid_size = (8, 8, 8)
            particles_x_gpu = CuArray([0.5f0])
            particles_y_gpu = CuArray([0.5f0])
            particles_z_gpu = CuArray([0.5f0])
            particles_q_gpu = CuArray([1.0f0])
            
            mesh_gpu = Mesh3D(grid_size, Array(particles_x_gpu), Array(particles_y_gpu), Array(particles_z_gpu); 
                             T=Float32, backend=CUDABackend())
            
            deposit!(mesh_gpu, particles_x_gpu, particles_y_gpu, particles_z_gpu, particles_q_gpu)
            
            # Check total charge conservation
            @test sum(Array(mesh_gpu.rho)) ≈ sum(Array(particles_q_gpu)) atol=1e-6
            
            # Check that charge is deposited
            @test any(Array(mesh_gpu.rho) .> 0)
        end

        # Test GPU interpolation
        @testset "GPU Interpolation" begin
            grid_size = (6, 6, 6)
            particles_x_gpu = CuArray([0.5f0])
            particles_y_gpu = CuArray([0.5f0])
            particles_z_gpu = CuArray([0.5f0])
            particles_q_gpu = CuArray([1.0f0])
            
            mesh_gpu = Mesh3D(grid_size, Array(particles_x_gpu), Array(particles_y_gpu), Array(particles_z_gpu); 
                             T=Float32, backend=CUDABackend())
            
            # Set up a simple field pattern (use broadcasting to avoid scalar indexing)
            efield_cpu = zeros(Float32, (6,6,6,3))
            efield_cpu[1,1,1,1] = 1.0f0
            efield_cpu[2,1,1,1] = 2.0f0
            mesh_gpu.efield .= CuArray(efield_cpu)
            
            Ex_gpu, Ey_gpu, Ez_gpu = interpolate_field(mesh_gpu, particles_x_gpu, particles_y_gpu, particles_z_gpu)
            
            # Check return types
            @test Ex_gpu isa CuArray{Float32, 1}
            @test Ey_gpu isa CuArray{Float32, 1}
            @test Ez_gpu isa CuArray{Float32, 1}
            
            # Check that interpolation gives reasonable values
            @test isfinite(Array(Ex_gpu)[1])
            @test isfinite(Array(Ey_gpu)[1])
            @test isfinite(Array(Ez_gpu)[1])
        end

        # Test GPU solver
        @testset "GPU Solver" begin
            grid_size = (8, 8, 8)
            particles_x_gpu = CuArray([0.0f0])
            particles_y_gpu = CuArray([0.0f0])
            particles_z_gpu = CuArray([0.0f0])
            particles_q_gpu = CuArray([1.0f0])
            
            mesh_gpu = Mesh3D(grid_size, Array(particles_x_gpu), Array(particles_y_gpu), Array(particles_z_gpu); 
                             T=Float32, backend=CUDABackend(), gamma=2.0)
            
            # Deposit charge and solve
            deposit!(mesh_gpu, particles_x_gpu, particles_y_gpu, particles_z_gpu, particles_q_gpu)
            solve!(mesh_gpu)
            
            # Check that fields are computed
            @test any(Array(mesh_gpu.efield) .!= 0.0f0)
            
            # Check charge conservation
            @test sum(Array(mesh_gpu.rho)) ≈ sum(Array(particles_q_gpu)) atol=1e-6
        end

        # Test GPU with multiple particles
        @testset "GPU Multiple Particles" begin
            grid_size = (6, 6, 6)
            particles_x_gpu = CuArray([0.2f0, 0.8f0])
            particles_y_gpu = CuArray([0.3f0, 0.7f0])
            particles_z_gpu = CuArray([0.4f0, 0.6f0])
            particles_q_gpu = CuArray([1.0f0, -1.0f0])
            
            mesh_gpu = Mesh3D(grid_size, Array(particles_x_gpu), Array(particles_y_gpu), Array(particles_z_gpu); 
                             T=Float32, backend=CUDABackend())
            
            # Deposit and solve
            deposit!(mesh_gpu, particles_x_gpu, particles_y_gpu, particles_z_gpu, particles_q_gpu)
            solve!(mesh_gpu)
            
            # Interpolate fields
            Ex_gpu, Ey_gpu, Ez_gpu = interpolate_field(mesh_gpu, particles_x_gpu, particles_y_gpu, particles_z_gpu)
            
            # Check results
            @test length(Ex_gpu) == 2
            @test all(isfinite.(Array(Ex_gpu)))
            @test all(isfinite.(Array(Ey_gpu)))
            @test all(isfinite.(Array(Ez_gpu)))
            
            # Check charge conservation
            @test sum(Array(mesh_gpu.rho)) ≈ sum(Array(particles_q_gpu)) atol=1e-6
        end

        # Test GPU vs CPU consistency
        @testset "GPU vs CPU Consistency" begin
            grid_size = (4, 4, 4)
            particles_x = [0.5]
            particles_y = [0.5]
            particles_z = [0.5]
            particles_q = [1.0]
            
            # CPU mesh
            mesh_cpu = Mesh3D(grid_size, particles_x, particles_y, particles_z; T=Float32)
            deposit!(mesh_cpu, particles_x, particles_y, particles_z, particles_q)
            solve!(mesh_cpu)
            
            # GPU mesh
            particles_x_gpu = CuArray(Float32.(particles_x))
            particles_y_gpu = CuArray(Float32.(particles_y))
            particles_z_gpu = CuArray(Float32.(particles_z))
            particles_q_gpu = CuArray(Float32.(particles_q))
            
            mesh_gpu = Mesh3D(grid_size, particles_x, particles_y, particles_z; T=Float32, backend=CUDABackend())
            deposit!(mesh_gpu, particles_x_gpu, particles_y_gpu, particles_z_gpu, particles_q_gpu)
            solve!(mesh_gpu)
            
            # Check that results are similar (allowing for floating point differences)
            @test isapprox(sum(Array(mesh_gpu.rho)), sum(mesh_cpu.rho), atol=1e-5)
            @test isapprox(sum(Array(mesh_gpu.efield)), sum(mesh_cpu.efield), rtol=1e-5)
        end

        # Test GPU clear_mesh!
        @testset "GPU Clear Mesh" begin
            grid_size = (4, 4, 4)
            particles_x_gpu = CuArray([0.5f0])
            particles_y_gpu = CuArray([0.5f0])
            particles_z_gpu = CuArray([0.5f0])
            particles_q_gpu = CuArray([1.0f0])
            
            mesh_gpu = Mesh3D(grid_size, Array(particles_x_gpu), Array(particles_y_gpu), Array(particles_z_gpu); 
                             T=Float32, backend=CUDABackend())
            
            # Deposit some charge
            deposit!(mesh_gpu, particles_x_gpu, particles_y_gpu, particles_z_gpu, particles_q_gpu)
            @test sum(Array(mesh_gpu.rho)) ≈ sum(Array(particles_q_gpu)) atol=1e-6
            
            # Clear the mesh
            clear_mesh!(mesh_gpu)
            @test all(Array(mesh_gpu.rho) .== 0.0f0)
        end
    end
end 