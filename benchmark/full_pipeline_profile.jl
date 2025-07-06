# CPU Function-Level Profiling Script for Full Pipeline
#
# Usage:
#   julia --project benchmark/full_pipeline_benchmark_profile.jl
#
# This script profiles the CPU full pipeline (deposit! + solve! + interpolate_field)
# using Julia's built-in Profile and optionally StatProfilerHTML for an HTML report.
#
# To install StatProfilerHTML (optional):
#   import Pkg; Pkg.add("StatProfilerHTML")

using SpaceCharge
using Random
using CUDA
using Profile
using StatProfilerHTML

# Standalone CPU full pipeline for profiling
function run_cpu_full_pipeline(grid_size, n_particles; total_charge=1.0e-9, sigma=0.001)
    Random.seed!(42)
    charge_per_particle = total_charge / n_particles
    particles_x = randn(n_particles) .* sigma
    particles_y = randn(n_particles) .* sigma
    particles_z = randn(n_particles) .* sigma
    particles_q = fill(charge_per_particle, n_particles)
    mesh_cpu = Mesh3D(grid_size, particles_x, particles_y, particles_z, total_charge=total_charge)
    deposit!(mesh_cpu, particles_x, particles_y, particles_z, particles_q)
    solve!(mesh_cpu)
    interpolate_field(mesh_cpu, particles_x, particles_y, particles_z)
end

# Standalone GPU full pipeline for profiling
function run_gpu_full_pipeline(grid_size, n_particles; total_charge=1.0e-9, sigma=0.001)
    if !CUDA.functional()
        println("GPU: Not available")
        return
    end
    Random.seed!(42)
    charge_per_particle = total_charge / n_particles
    particles_x = randn(n_particles) .* sigma
    particles_y = randn(n_particles) .* sigma
    particles_z = randn(n_particles) .* sigma
    particles_q = fill(charge_per_particle, n_particles)
    particles_x_gpu = CuArray(particles_x)
    particles_y_gpu = CuArray(particles_y)
    particles_z_gpu = CuArray(particles_z)
    particles_q_gpu = CuArray(particles_q)
    mesh_gpu = Mesh3D(grid_size, particles_x, particles_y, particles_z; array_type=CuArray, total_charge=total_charge)
    deposit!(mesh_gpu, particles_x_gpu, particles_y_gpu, particles_z_gpu, particles_q_gpu)
    solve!(mesh_gpu)
    interpolate_field(mesh_gpu, particles_x_gpu, particles_y_gpu, particles_z_gpu)
    CUDA.synchronize()
end 

function print_usage()
    println("Usage: julia --project benchmark/full_pipeline_profile.jl [--cpu|--gpu|--both]")
    println("  --cpu   Profile only the CPU full pipeline")
    println("  --gpu   Profile only the GPU full pipeline")
    println("  --both  Profile both CPU and GPU (default)")
end

function main()
    profile_cpu = false
    profile_gpu = false
    if length(ARGS) == 0
        profile_cpu = true
        profile_gpu = true
    else
        for arg in ARGS
            if arg == "--cpu"
                profile_cpu = true
            elseif arg == "--gpu"
                profile_gpu = true
            elseif arg == "--both"
                profile_cpu = true
                profile_gpu = true
            else
                print_usage()
                exit(1)
            end
        end
    end

    if profile_cpu
        try
            println("\nProfiling CPU full pipeline (function-level)...")
            # Warm-up run (not profiled)
            run_cpu_full_pipeline((64, 64, 64), 100_000)
            Profile.clear()
            Profile.init()
            @profile run_cpu_full_pipeline((64, 64, 64), 100_000)
            open("profile_output.txt", "w") do io
                Profile.print(io)
            end
            println("[Info] CPU profile output written to profile_output.txt")
            # Optional: Uncomment to view HTML report (requires StatProfilerHTML)
            statprofilehtml()
        catch e
            println("[Warning] Profiling tools not available: $e")
        end
    end

    if profile_gpu
        try
            if CUDA.functional()
                println("\nProfiling GPU full pipeline (kernel-level)...")
                # Warm-up run (not profiled)
                run_gpu_full_pipeline((64, 64, 64), 100_000)
                try
                    CUDA.@profile run_gpu_full_pipeline((64, 64, 64), 100_000)
                    println("[Info] CUDA profiling complete. Use NVIDIA Nsight Systems or nvprof to analyze the results if available.")
                catch e
                    println("[Warning] CUDA profiling failed: $e")
                end
            else
                println("[Info] CUDA GPU not available, skipping GPU profiling.")
            end
        catch e
            println("[Warning] CUDA.jl not available: $e")
        end
    end
end

main()
