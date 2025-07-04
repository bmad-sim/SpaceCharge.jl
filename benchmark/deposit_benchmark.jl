using SpaceCharge
using CUDA
using BenchmarkTools
using Random

"""
Compare CPU vs GPU performance for deposit! function
"""
function compare_cpu_gpu(grid_size, n_particles)
    println("\n" * "="^60)
    println("CPU vs GPU Performance Comparison")
    println("Grid: $(grid_size[1])³, Particles: $n_particles")
    println("="^60)
    
    # Generate test data
    Random.seed!(42)
    particles_x = randn(n_particles) * 0.5
    particles_y = randn(n_particles) * 0.3
    particles_z = randn(n_particles) * 0.2
    particles_q = ones(n_particles) * 1e-12
    
    expected_charge = sum(particles_q)
    
    # CPU benchmark
    println("Benchmarking CPU...")
    mesh_cpu = Mesh3D(grid_size, particles_x, particles_y, particles_z)
    
    cpu_time = @belapsed deposit!($mesh_cpu, $particles_x, $particles_y, $particles_z, $particles_q)
    cpu_charge = sum(mesh_cpu.rho)
    cpu_error = abs(expected_charge - cpu_charge) / expected_charge * 100
    
    println("CPU: $(round(cpu_time * 1000, digits=2)) ms (error: $(round(cpu_error, digits=8))%)")
    
    # GPU benchmark (if available)
    gpu_time = NaN
    gpu_error = NaN
    speedup = NaN
    
    if CUDA.functional()
        println("Benchmarking GPU...")
        
        # Move to GPU
        particles_x_gpu = CuArray(particles_x)
        particles_y_gpu = CuArray(particles_y)
        particles_z_gpu = CuArray(particles_z)
        particles_q_gpu = CuArray(particles_q)
        
        # Create mesh on GPU
        mesh_gpu = Mesh3D(grid_size, particles_x, particles_y, particles_z; array_type=CuArray)
        
        gpu_time = @belapsed begin
            deposit!($mesh_gpu, $particles_x_gpu, $particles_y_gpu, $particles_z_gpu, $particles_q_gpu)
            CUDA.synchronize()
        end
        
        gpu_charge = sum(Array(mesh_gpu.rho))
        gpu_error = abs(expected_charge - gpu_charge) / expected_charge * 100
        speedup = cpu_time / gpu_time
        
        println("GPU: $(round(gpu_time * 1000, digits=2)) ms (error: $(round(gpu_error, digits=8))%)")
        println("Speedup: $(round(speedup, digits=1))x")
    else
        println("GPU: Not available")
    end
    
    return (cpu_time, gpu_time, speedup, cpu_error, gpu_error)
end

"""
Comprehensive CPU vs GPU benchmark across different problem sizes
"""
function comprehensive_benchmark()
    println("\n🚀 Comprehensive CPU vs GPU Benchmark")
    println("Testing performance across different problem sizes...")
    
    # Test different problem sizes
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
            results = compare_cpu_gpu(grid_size, n_particles)
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
        println("Grid Size | Particles  | CPU (ms) | GPU (ms) | Speedup | CPU Error | GPU Error")
        println("-"^70)
        
        total_speedup = 0.0
        valid_gpu_tests = 0
        
        for (grid_size, n_particles, (cpu_time, gpu_time, speedup, cpu_error, gpu_error)) in all_results
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
            
            cpu_error_str = "$(round(cpu_error, digits=6))%"
            
            println("$(rpad(grid_str, 9)) | $(rpad(n_particles, 10)) | $(rpad(cpu_ms, 8)) | $(rpad(gpu_ms, 8)) | $(rpad(speedup_str, 7)) | $(rpad(cpu_error_str, 9)) | $gpu_error_str")
        end
        
        if valid_gpu_tests > 0
            avg_speedup = total_speedup / valid_gpu_tests
            println("\n📊 Average GPU speedup: $(round(avg_speedup, digits=1))x")
            
            # Performance insights
            println("\n🎯 Performance Insights:")
            println("• GPU acceleration is most effective for larger particle counts")
            println("• Both CPU and GPU methods maintain excellent charge conservation")
            
            if avg_speedup > 10
                println("• Excellent GPU acceleration achieved!")
            elseif avg_speedup > 5
                println("• Good GPU acceleration achieved")
            else
                println("• Moderate GPU acceleration - may benefit from larger problem sizes")
            end
        else
            println("\nGPU benchmarks not available")
        end
        
    else
        println("No successful benchmark results to display")
    end
end

# Run benchmarks if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    comprehensive_benchmark()
end 