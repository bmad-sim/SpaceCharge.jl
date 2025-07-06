using SpaceCharge
using CUDA
using BenchmarkTools
using Random
using LinearAlgebra
using SpecialFunctions

"""
Constants for analytical calculations
"""
EPSILON_0 = 8.8541878128e-12 # Permittivity of free space

"""
Exact analytical solution for isotropic Gaussian bunch
"""
function E_analytical(r; Q=1, σ=1)
    if norm(r) == 0
        return [0.0, 0.0, 0.0]
    end
    return Q/(4π*EPSILON_0*norm(r)^3)*(erf(norm(r)/(sqrt(2)*σ)) - sqrt(2/π)*norm(r)/σ*exp(-(norm(r)/σ)^2/2)).*r
end

"""
Test correctness of solve! against analytical solution
"""
function test_correctness(grid_size, n_particles; total_charge=1.0e-9, sigma=0.001)
    println("\n" * "="^60)
    println("Correctness Test")
    println("Grid: $(grid_size[1])³, Particles: $n_particles")
    println("="^60)
    
    # Generate test particles
    Random.seed!(123)
    charge_per_particle = total_charge / n_particles
    
    particles_x = randn(n_particles) .* sigma
    particles_y = randn(n_particles) .* sigma
    particles_z = randn(n_particles) .* sigma
    particles_q = fill(charge_per_particle, n_particles)
    
    # Create mesh and deposit charge
    mesh = Mesh3D(grid_size, particles_x, particles_y, particles_z, total_charge=total_charge)
    deposit!(mesh, particles_x, particles_y, particles_z, particles_q)
    
    # Solve for fields
    solve!(mesh)
    
    # Extract data for comparison along z-axis (x=0, y=0)
    z_coords = [mesh.min_bounds[3] + (k - 1) * mesh.delta[3] for k in 1:grid_size[3]]
    
    # Find the indices closest to x=0 and y=0
    x_center_idx = argmin(abs.([mesh.min_bounds[1] + (i - 1) * mesh.delta[1] for i in 1:grid_size[1]]))
    y_center_idx = argmin(abs.([mesh.min_bounds[2] + (j - 1) * mesh.delta[2] for j in 1:grid_size[2]]))
    
    computed_Ez = [mesh.efield[x_center_idx, y_center_idx, k, 3] for k in 1:grid_size[3]]
    analytical_Ez = [E_analytical([0.0, 0.0, z], Q=total_charge, σ=sigma)[3] for z in z_coords]
    
    # Calculate errors
    max_abs_error = maximum(abs.(computed_Ez .- analytical_Ez)) / maximum(abs.(analytical_Ez))
    rms_error = sqrt(mean((computed_Ez .- analytical_Ez).^2)) / sqrt(mean(analytical_Ez.^2))
    
    println("Max relative error: $(round(max_abs_error * 100, digits=4))%")
    println("RMS relative error: $(round(rms_error * 100, digits=4))%")
    
    return (max_abs_error, rms_error)
end

"""
Compare CPU vs GPU performance for solve! function, including correctness
"""
function compare_cpu_gpu_solve(grid_size, n_particles; total_charge=1.0e-9, sigma=0.001)
    println("\n" * "="^60)
    println("CPU vs GPU Performance Comparison")
    println("Grid: $(grid_size[1])³, Particles: $n_particles")
    println("="^60)

    # Generate test data
    Random.seed!(42)
    charge_per_particle = total_charge / n_particles
    particles_x = randn(n_particles) .* sigma
    particles_y = randn(n_particles) .* sigma
    particles_z = randn(n_particles) .* sigma
    particles_q = fill(charge_per_particle, n_particles)

    # CPU benchmark
    println("Benchmarking CPU...")
    mesh_cpu = Mesh3D(grid_size, particles_x, particles_y, particles_z, total_charge=total_charge)
    deposit!(mesh_cpu, particles_x, particles_y, particles_z, particles_q)
    cpu_time = @belapsed solve!($mesh_cpu)

    # Correctness: compare Ez along z-axis to analytical
    z_coords = [mesh_cpu.min_bounds[3] + (k - 1) * mesh_cpu.delta[3] for k in 1:grid_size[3]]
    x_center_idx = argmin(abs.([mesh_cpu.min_bounds[1] + (i - 1) * mesh_cpu.delta[1] for i in 1:grid_size[1]]))
    y_center_idx = argmin(abs.([mesh_cpu.min_bounds[2] + (j - 1) * mesh_cpu.delta[2] for j in 1:grid_size[2]]))
    computed_Ez = [mesh_cpu.efield[x_center_idx, y_center_idx, k, 3] for k in 1:grid_size[3]]
    analytical_Ez = [E_analytical([0.0, 0.0, z], Q=total_charge, σ=sigma)[3] for z in z_coords]
    max_error = maximum(abs.(computed_Ez .- analytical_Ez)) / maximum(abs.(analytical_Ez)) * 100

    println("CPU: $(round(cpu_time * 1000, digits=2)) ms (max error: $(round(max_error, digits=6))%)")

    # GPU benchmark (if available)
    gpu_time = NaN
    gpu_error = NaN
    speedup = NaN
    if CUDA.functional()
        println("Benchmarking GPU...")
        particles_x_gpu = CuArray(particles_x)
        particles_y_gpu = CuArray(particles_y)
        particles_z_gpu = CuArray(particles_z)
        particles_q_gpu = CuArray(particles_q)
        mesh_gpu = Mesh3D(grid_size, particles_x, particles_y, particles_z; array_type=CuArray, total_charge=total_charge)
        deposit!(mesh_gpu, particles_x_gpu, particles_y_gpu, particles_z_gpu, particles_q_gpu)
        gpu_time = @belapsed begin
            solve!($mesh_gpu)
            CUDA.synchronize()
        end
        # Compare Ez along z-axis
        computed_Ez_gpu = [Array(mesh_gpu.efield)[x_center_idx, y_center_idx, k, 3] for k in 1:grid_size[3]]
        gpu_error = maximum(abs.(computed_Ez_gpu .- analytical_Ez)) / maximum(abs.(analytical_Ez)) * 100
        speedup = cpu_time / gpu_time
        println("GPU: $(round(gpu_time * 1000, digits=2)) ms (max error: $(round(gpu_error, digits=6))%)")
        println("Speedup: $(round(speedup, digits=1))x")
    else
        println("GPU: Not available")
    end
    return (cpu_time, gpu_time, speedup, max_error, gpu_error)
end

"""
Comprehensive CPU vs GPU benchmark across different problem sizes (with correctness)
"""
function comprehensive_benchmark_solve()
    println("\n🚀 Comprehensive CPU vs GPU Benchmark")
    println("Testing performance and correctness across different problem sizes...")
    test_configs = [
        ((32, 32, 32), 10_000),
        ((32, 32, 32), 100_000),
        ((64, 64, 64), 100_000),
        ((64, 64, 64), 1_000_000),
        ((128, 128, 128), 100_000),
        ((128, 128, 128), 1_000_000),
    ]
    all_results = []
    for (grid_size, n_particles) in test_configs
        try
            results = compare_cpu_gpu_solve(grid_size, n_particles)
            push!(all_results, (grid_size, n_particles, results))
        catch e
            println("❌ Failed for grid $(grid_size[1])³, particles $n_particles: $e")
        end
    end
    # Summary
    println("\n" * "="^70)
    println("COMPREHENSIVE BENCHMARK SUMMARY")
    println("="^70)
    if !isempty(all_results)
        println("Performance Overview:")
        println("Grid Size | Particles  | CPU (ms) | GPU (ms) | Speedup | Max Error | GPU Error")
        println("-"^70)
        total_speedup = 0.0
        valid_gpu_tests = 0
        for (grid_size, n_particles, (cpu_time, gpu_time, speedup, max_error, gpu_error)) in all_results
            grid_str = "$(grid_size[1])³"
            cpu_ms = round(cpu_time * 1000, digits=2)
            if !isnan(gpu_time)
                gpu_ms = round(gpu_time * 1000, digits=2)
                speedup_str = "$(round(speedup, digits=1))x"
                gpu_error_str = "$(round(gpu_error, digits=6))%"
                total_speedup += speedup
                valid_gpu_tests += 1
            else
                gpu_ms = "N/A"
                speedup_str = "N/A"
                gpu_error_str = "N/A"
            end
            max_error_str = "$(round(max_error, digits=6))%"
            println("$(rpad(grid_str, 9)) | $(rpad(n_particles, 10)) | $(rpad(cpu_ms, 8)) | $(rpad(gpu_ms, 8)) | $(rpad(speedup_str, 7)) | $(rpad(max_error_str, 9)) | $gpu_error_str")
        end
        if valid_gpu_tests > 0
            avg_speedup = total_speedup / valid_gpu_tests
            println("\nAverage GPU speedup: $(round(avg_speedup, digits=1))x")
        else
            println("\nGPU benchmarks not available")
        end
    else
        println("No successful benchmark results to display")
    end
end

comprehensive_benchmark_solve() 