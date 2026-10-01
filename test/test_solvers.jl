using SpaceCharge
using Test

function run_solver_tests()
    @testset "Solvers" begin
        @testset "Cathode boundary and repeated superposition" begin
            mesh = Mesh3D((7,8,9), (-0.02,-0.03,0.0), (0.02,0.03,0.04); gamma=2.0)
            mesh.rho[3,5,4] = 1e-12
            mesh.rho[6,2,7] = 2e-12
            source = copy(mesh.rho)
            solve!(mesh)
            free = copy(mesh.efield)
            for cached in (false, true, true)
                solve!(mesh; at_cathode=true, cache_green=cached)
                @test mesh.rho == source
                scale = maximum(abs, mesh.efield)
                @test maximum(abs, mesh.efield[:,:,1,1:2]) < 1e-10 * scale
                @test mesh.efield[:,:,1,3] ≈ 2free[:,:,1,3] rtol=1e-10
            end
        end
        # Test FreeSpace Solver
        @testset "FreeSpace Solver" begin
            grid_size = (16, 16, 16)
            particles_x = [0.0]
            particles_y = [0.0]
            particles_z = [0.0]
            particles_q = [1.0]
            
            mesh = Mesh3D(grid_size, particles_x, particles_y, particles_z; gamma=2.0)
            
            # Deposit charge
            deposit!(mesh, particles_x, particles_y, particles_z, particles_q)
            
            # Solve for fields
            solve!(mesh)
            
            # Check that the sum of rho is conserved
            @test sum(mesh.rho) ≈ sum(particles_q) atol=1e-10
            
            # Check that fields are computed
            @test any(mesh.efield .!= 0.0)
            
            # Check that fields are finite (avoiding NaN issues)
            @test all(isfinite.(mesh.efield))
        end

        # Test FreeSpace Solver with cathode
        @testset "FreeSpace Solver with Cathode" begin
            grid_size = (16, 16, 16)
            particles_x = [0.0]
            particles_y = [0.0]
            particles_z = [0.01]  # Slightly above cathode
            particles_q = [1.0]
            
            mesh = Mesh3D(grid_size, particles_x, particles_y, particles_z; gamma=2.0)
            
            deposit!(mesh, particles_x, particles_y, particles_z, particles_q)
            solve!(mesh; at_cathode=true)
            
            # Create comparison mesh without cathode
            mesh_free_space = Mesh3D(grid_size, particles_x, particles_y, particles_z; gamma=2.0)
            deposit!(mesh_free_space, particles_x, particles_y, particles_z, particles_q)
            solve!(mesh_free_space)
            
            # Fields should be different with cathode
            @test !isapprox(mesh.efield, mesh_free_space.efield)
            
            # Check that fields are computed and finite
            @test any(mesh.efield .!= 0.0)
            @test all(isfinite.(mesh.efield))
        end

        # Test solver with different gamma values
        @testset "Solver with Different Gamma" begin
            grid_size = (8, 8, 8)
            particles_x = [0.0]
            particles_y = [0.0]
            particles_z = [0.0]
            particles_q = [1.0]
            
            # Test with gamma = 1.0 (non-relativistic)
            mesh1 = Mesh3D(grid_size, particles_x, particles_y, particles_z; gamma=1.0)
            deposit!(mesh1, particles_x, particles_y, particles_z, particles_q)
            solve!(mesh1)
            
            # Test with gamma = 10.0 (relativistic)
            mesh2 = Mesh3D(grid_size, particles_x, particles_y, particles_z; gamma=10.0)
            deposit!(mesh2, particles_x, particles_y, particles_z, particles_q)
            solve!(mesh2)
            
            # Fields should be different for different gamma values
            @test !isapprox(mesh1.efield, mesh2.efield)
        end
    end
end 
