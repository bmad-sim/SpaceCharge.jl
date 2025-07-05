using SpaceCharge
using CUDA
using BenchmarkTools
using Random
using Test

"""
Benchmark deposit! + solve! + interpolate_field for CPU and GPU
"""
function compare_cpu_gpu_full_pipeline(grid_size, n_particles; total_charge=1.0e-9, sigma=0.001)
    println("\n" * "="^60)
    println("CPU vs GPU Full Pipeline Benchmark (deposit! + solve! + interpolate_field)")
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
    cpu_time = @belapsed begin
        deposit!($mesh_cpu, $particles_x, $particles_y, $particles_z, $particles_q)
        solve!($mesh_cpu)
        interpolate_field($mesh_cpu, $particles_x, $particles_y, $particles_z)
    end
    println("CPU: $(round(cpu_time * 1000, digits=2)) ms")

    # GPU benchmark (if available)
    gpu_time = NaN
    speedup = NaN
    if CUDA.functional()
        println("Benchmarking GPU...")
        particles_x_gpu = CuArray(particles_x)
        particles_y_gpu = CuArray(particles_y)
        particles_z_gpu = CuArray(particles_z)
        particles_q_gpu = CuArray(particles_q)
        mesh_gpu = Mesh3D(grid_size, particles_x, particles_y, particles_z; array_type=CuArray, total_charge=total_charge)
        gpu_time = @belapsed begin
            deposit!($mesh_gpu, $particles_x_gpu, $particles_y_gpu, $particles_z_gpu, $particles_q_gpu)
            solve!($mesh_gpu)
            interpolate_field($mesh_gpu, $particles_x_gpu, $particles_y_gpu, $particles_z_gpu)
            CUDA.synchronize()
        end
        speedup = cpu_time / gpu_time
        println("GPU: $(round(gpu_time * 1000, digits=2)) ms")
        println("Speedup: $(round(speedup, digits=1))x")
    else
        println("GPU: Not available")
    end
    return (cpu_time, gpu_time, speedup)
end

"""
Comprehensive CPU vs GPU full pipeline benchmark across different problem sizes
"""
function comprehensive_benchmark_full_pipeline()
    println("\n🚀 Comprehensive CPU vs GPU Full Pipeline Benchmark")
    println("Testing performance for deposit! + solve! + interpolate_field across different problem sizes...")
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
            results = compare_cpu_gpu_full_pipeline(grid_size, n_particles)
            push!(all_results, (grid_size, n_particles, results))
        catch e
            println("❌ Failed for grid $(grid_size[1])³, particles $n_particles: $e")
        end
    end
    # Summary
    println("\n" * "="^70)
    println("COMPREHENSIVE FULL PIPELINE BENCHMARK SUMMARY")
    println("="^70)
    if !isempty(all_results)
        println("Grid Size | Particles  | CPU (ms) | GPU (ms) | Speedup ")
        println("-"^55)
        total_speedup = 0.0
        valid_gpu_tests = 0
        for (grid_size, n_particles, (cpu_time, gpu_time, speedup)) in all_results
            grid_str = "$(grid_size[1])³"
            cpu_ms = round(cpu_time * 1000, digits=2)
            if !isnan(gpu_time)
                gpu_ms = round(gpu_time * 1000, digits=2)
                speedup_str = "$(round(speedup, digits=1))x"
                total_speedup += speedup
                valid_gpu_tests += 1
            else
                gpu_ms = "N/A"
                speedup_str = "N/A"
            end
            println("$(rpad(grid_str, 9)) | $(rpad(n_particles, 10)) | $(rpad(cpu_ms, 8)) | $(rpad(gpu_ms, 8)) | $(rpad(speedup_str, 7))")
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

if abspath(PROGRAM_FILE) == @__FILE__
    comprehensive_benchmark_full_pipeline()
end 