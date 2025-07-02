
using SpaceCharge
using BenchmarkTools
using CUDA

# Function to generate random particles
function generate_particles(n_particles, array_type=Array)
    x = array_type(rand(Float32, n_particles))
    y = array_type(rand(Float32, n_particles))
    z = array_type(rand(Float32, n_particles))
    q = array_type(rand(Float32, n_particles))
    return x, y, z, q
end

# --- Benchmarking Functions ---

function benchmark_deposit!(mesh, particles)
    return @benchmarkable deposit!($mesh, $particles...)
end

function benchmark_solve!(mesh)
    # The first run of solve! on GPU can be slow due to compilation.
    # We can warm up by running it once before benchmarking.
    if typeof(mesh.rho) <: CuArray
        solve!(mesh, FreeSpace())
    end
    return @benchmarkable solve!($mesh, FreeSpace())
end

function run_benchmarks(grid_size, n_particles, array_type)
    println("--------------------------------------------------")
    println("Running benchmarks for:")
    println("  Grid Size:   $grid_size")
    println("  N Particles: $n_particles")
    println("  Array Type:  $array_type")
    println("--------------------------------------------------")

    # Setup
    min_bounds = (-0.05f0, -0.05f0, -0.05f0)
    max_bounds = (0.05f0, 0.05f0, 0.05f0)
    mesh = Mesh3D(grid_size, min_bounds, max_bounds; array_type=array_type)
    particles = generate_particles(n_particles, array_type)

    # Create benchmark suite
    suite = BenchmarkGroup()
    # Warm up deposit! before benchmarking
    deposit!(mesh, particles...)
    suite["deposit!"] = benchmark_deposit!(mesh, particles)
    suite["solve!"] = benchmark_solve!(mesh)

    # Run benchmarks
    results = run(suite, verbose = true)

    return results
end

function main()
    # Define benchmark parameters
    grid_sizes = [(32, 32, 32), (64, 64, 64), (128, 128, 128)]
    particle_counts = [100_000]
    
    # --- CPU Benchmarks ---
    println("==================================================")
    println("                 CPU Benchmarks")
    println("==================================================")
    for gs in grid_sizes
        for pc in particle_counts
            run_benchmarks(gs, pc, Array)
        end
    end
    
    # --- GPU Benchmarks ---
    if CUDA.functional()
        println("\n==================================================")
        println("                 GPU Benchmarks")
        println("==================================================")
        for gs in grid_sizes
            for pc in particle_counts
                run_benchmarks(gs, pc, CuArray)
            end
        end
    else
        println("\nCUDA not functional. Skipping GPU benchmarks.")
    end
end

main()
