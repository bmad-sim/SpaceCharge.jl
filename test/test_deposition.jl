using SpaceCharge
using Test

function run_deposition_tests()
    @testset "Deposition" begin
        # Test basic deposition
        @testset "Basic Deposition" begin
            grid_size = (4, 4, 4)
            particles_x = [0.5]
            particles_y = [0.5]
            particles_z = [0.5]
            particles_q = [1.0]
            
            mesh = Mesh3D(grid_size, particles_x, particles_y, particles_z)
            
            # Clear mesh and deposit
            clear_mesh!(mesh)
            deposit!(mesh, particles_x, particles_y, particles_z, particles_q)
            
            # Check total charge conservation
            @test sum(mesh.rho) ≈ sum(particles_q) atol=1e-10
            
            # Check that charge is deposited in the correct cells
            # For a particle at (0.5, 0.5, 0.5), it should affect cells around the center
            @test any(mesh.rho .> 0)
        end

        # Test multiple particles
        @testset "Multiple Particles" begin
            grid_size = (6, 6, 6)
            particles_x = [0.2, 0.8]
            particles_y = [0.3, 0.7]
            particles_z = [0.4, 0.6]
            particles_q = [1.0, -1.0]
            
            mesh = Mesh3D(grid_size, particles_x, particles_y, particles_z)
            
            deposit!(mesh, particles_x, particles_y, particles_z, particles_q)
            
            # Check total charge conservation
            @test sum(mesh.rho) ≈ sum(particles_q) atol=1e-10
            
            # Check that we have both positive and negative charge
            @test any(mesh.rho .> 0)
            @test any(mesh.rho .< 0)
        end

        # Test accumulation (clear=false)
        @testset "Charge Accumulation" begin
            grid_size = (4, 4, 4)
            particles_x = [0.5]
            particles_y = [0.5]
            particles_z = [0.5]
            particles_q = [1.0]
            
            mesh = Mesh3D(grid_size, particles_x, particles_y, particles_z)
            
            # First deposit
            deposit!(mesh, particles_x, particles_y, particles_z, particles_q)
            first_sum = sum(mesh.rho)
            
            # Second deposit without clearing
            deposit!(mesh, particles_x, particles_y, particles_z, particles_q; clear=false)
            second_sum = sum(mesh.rho)
            
            # Should be double the charge
            @test second_sum ≈ 2 * first_sum atol=1e-10
        end

        # Test edge cases
        @testset "Edge Cases" begin
            grid_size = (4, 4, 4)
            
            # Test particle at grid point
            particles_x = [0.0]
            particles_y = [0.0]
            particles_z = [0.0]
            particles_q = [1.0]
            
            mesh = Mesh3D(grid_size, particles_x, particles_y, particles_z)
            deposit!(mesh, particles_x, particles_y, particles_z, particles_q)
            
            @test sum(mesh.rho) ≈ sum(particles_q) atol=1e-10
            
            # Test particle at boundary
            particles_x = [mesh.max_bounds[1]]
            particles_y = [mesh.max_bounds[2]]
            particles_z = [mesh.max_bounds[3]]
            
            mesh2 = Mesh3D(grid_size, particles_x, particles_y, particles_z)
            deposit!(mesh2, particles_x, particles_y, particles_z, particles_q)
            
            @test sum(mesh2.rho) ≈ sum(particles_q) atol=1e-10
        end

        # Test clear_mesh! function
        @testset "Clear Mesh" begin
            grid_size = (4, 4, 4)
            particles_x = [0.5]
            particles_y = [0.5]
            particles_z = [0.5]
            particles_q = [1.0]
            
            mesh = Mesh3D(grid_size, particles_x, particles_y, particles_z)
            
            # Deposit some charge
            deposit!(mesh, particles_x, particles_y, particles_z, particles_q)
            @test sum(mesh.rho) ≈ sum(particles_q) atol=1e-10
            
            # Clear the mesh
            clear_mesh!(mesh)
            @test all(mesh.rho .== 0)
        end
    end
end 