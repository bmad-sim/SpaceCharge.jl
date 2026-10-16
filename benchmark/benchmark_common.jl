# Shared workload, command-line options, and timing output for the three benchmarks.
using SpaceCharge
using CUDA
using FFTW
using KernelAbstractions: get_backend
using BenchmarkTools
using Random
using Statistics
using Dates

function benchmark_usage(kind)
    println("""
    Usage: julia --startup-file=no --threads=1 --project=benchmark benchmark/$(kind)_benchmark.jl [options]
      --backend cpu|gpu|both       Default: both (also --cpu, --gpu, --both)
      --particles N               Default: 100000
      --grid N or NX,NY,NZ         Default: 64 (64,64,64)
      --precision Float32|Float64  Default: Float64
      --gamma G                   Default: 10
      --fft-threads N              Default: 1
      --seconds S                 Maximum sampling time per case, default: 2
      --samples N                 Maximum samples per case, default: 100
      --out DIRECTORY             Parent for a fresh run directory; default: depot scratchspaces
      --help                      Show this help
    """)
    if kind != "deposit"
        println("  --method direct|potential_fd|both  Default: both")
        println("  --cache cached|rebuilt|both         Default: both")
    end
    kind == "full_pipeline" && println("  --bounds adaptive|explicit|both     Default: both")
    println("\nOne workload size per invocation. Larger sweeps must be requested explicitly.")
    println("Setup, transfers, FFT planning and five warmups precede timing; GPU samples synchronize.")
end

function benchmark_options(args, kind)
    values = Dict("backend" => "both", "particles" => "100000", "grid" => "64",
                  "precision" => "Float64", "gamma" => "10", "fft-threads" => "1",
                  "seconds" => "2", "samples" => "100",
                  "out" => joinpath(first(DEPOT_PATH), "scratchspaces", "spacecharge-benchmarks"))
    kind != "deposit" && merge!(values, Dict("method" => "both", "cache" => "both"))
    kind == "full_pipeline" && (values["bounds"] = "both")
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("--cpu", "--gpu", "--both")
            values["backend"] = arg[3:end]
        else
            startswith(arg, "--") || error("Expected an option, got $arg; use --help")
            key = arg[3:end]
            haskey(values, key) || error("Unknown option $arg; use --help")
            i += 1
            i <= length(args) || error("Missing value for $arg")
            values[key] = args[i]
        end
        i += 1
    end
    for (key, choices) in (("backend", ("cpu", "gpu", "both")),
                           ("precision", ("Float32", "Float64")),
                           ("method", ("direct", "potential_fd", "both")),
                           ("cache", ("cached", "rebuilt", "both")),
                           ("bounds", ("adaptive", "explicit", "both")))
        haskey(values, key) && !(values[key] in choices) && error("Invalid --$key")
    end
    dims = parse.(Int, split(values["grid"], ','))
    length(dims) == 1 && (dims = fill(only(dims), 3))
    length(dims) == 3 && all(>=(2), dims) || error("Grid needs three dimensions >= 2")
    particles = parse(Int, values["particles"])
    samples = parse(Int, values["samples"])
    fft_threads = parse(Int, values["fft-threads"])
    seconds = parse(Float64, values["seconds"])
    T = values["precision"] == "Float64" ? Float64 : Float32
    gamma = parse(T, values["gamma"])
    particles > 0 && samples > 0 && fft_threads > 0 || error("Counts must be positive")
    isfinite(seconds) && seconds > 0 || error("Seconds must be finite and positive")
    isfinite(gamma) && gamma >= 1 || error("Gamma must be finite and >= 1")
    return (; backend=values["backend"], particles, grid=Tuple(dims), T, gamma,
            fft_threads, seconds, samples, method=get(values, "method", "direct"),
            cache=get(values, "cache", "cached"), bounds=get(values, "bounds", "fixed"),
            out=abspath(values["out"]))
end

function benchmark_backends(opts)
    opts.backend == "cpu" && return ("cpu",)
    if !CUDA.functional()
        opts.backend == "gpu" && error("CUDA is not functional")
        println("CUDA is not functional; running CPU cases only.")
        return ("cpu",)
    end
    CUDA.allowscalar(false)
    return opts.backend == "both" ? ("cpu", "gpu") : ("gpu",)
end

function prepare_benchmark(opts, gpu, method, cache)
    # Recreate exactly the same host particles for every backend/method/cache case.
    rng = MersenneTwister(42)
    xyz = ntuple(_ -> randn(rng, opts.T, opts.particles) .* opts.T(0.001), 3)
    q = fill(opts.T(1e-9) / opts.particles, opts.particles)
    if gpu
        xyz = CuArray.(xyz)
        q = CuArray(q)
    end
    mesh = FFTIGF(opts.grid; T=opts.T, backend=get_backend(q),
                  field_method=Symbol(method), cache_green=cache == "cached")
    # The public particle solve establishes geometry and deposits charge once.
    solve!(mesh, xyz..., q, opts.gamma)
    gpu && CUDA.synchronize()
    outputs = ntuple(_ -> similar(q), 3)
    # Reuse the fitted box for explicit cases so bounds modes have the same mesh.
    return (; mesh, xyz, q, outputs, gamma=opts.gamma,
            fitted_bounds=(mesh.min_bounds, mesh.max_bounds))
end

function measure_benchmark(operation, gpu, opts)
    run = () -> begin
        operation()
        gpu && CUDA.synchronize()
        nothing
    end
    for _ in 1:5
        run()
    end
    GC.gc()
    gpu && CUDA.synchronize()
    return @benchmark $run() evals=1 samples=opts.samples seconds=opts.seconds
end

function benchmark_output(opts, kind, args, backends)
    repo = dirname(@__DIR__)
    realpath(dirname(dirname(pathof(SpaceCharge)))) == realpath(repo) ||
        error("The active environment must develop this SpaceCharge checkout: $repo")
    FFTW.set_num_threads(opts.fft_threads)
    mkpath(opts.out)
    dir = mktempdir(opts.out; prefix=Dates.format(now(), "yyyymmdd-HHMMSS") * "-$kind-", cleanup=false)
    open(joinpath(dir, "metadata.txt"), "w") do io
        println(io, "Date (UTC): $(now(UTC))")
        println(io, "HEAD: ", readchomp(`git -C $repo rev-parse HEAD`))
        println(io, "Working tree:\n", readchomp(`git -C $repo status --short`))
        println(io, "Julia: $VERSION executable=$(joinpath(Sys.BINDIR, Base.julia_exename()))")
        println(io, "Project: $(Base.active_project())\nSource: $(pathof(SpaceCharge))")
        println(io, "Script: $(kind)_benchmark.jl\nArguments: $(repr(args))")
        println(io, "CPU: $(Sys.cpu_info()[1].model)")
        println(io, "Julia threads=$(Threads.nthreads()) FFTW threads=$(FFTW.get_num_threads()) FFTW=$(pkgversion(FFTW))")
        println(io, "BenchmarkTools=$(pkgversion(BenchmarkTools)) CUDA.jl=$(pkgversion(CUDA))")
        println(io, "Options: $opts\nBackends run: $backends")
        println(io, "Workload: fixed Gaussian particles, seed=42, sigma=0.001 m, total_charge=1e-9 C")
        println(io, "Solver: free space, pruned_fft=true (CPU only), FFTW.MEASURE")
        println(io, "cached: cache_green=true, warmed fixed geometry/gamma; rebuilt: cache_green=false")
        println(io, "Rebuilt spectra do not include new mesh allocation or FFT planning.")
        println(io, "Five warmups; evals=1; GPU synchronization included in every timing sample.")
        println(io, "Setup, host/device transfers and FFT planning excluded. Host allocation excludes device memory.")
        "gpu" in backends && CUDA.versioninfo(io)
    end
    println("Benchmark outputs: $dir")
    return dir
end

function write_result(io, kind, backend, method, bounds, cache, opts, case, trial)
    low, high = quantile(trial.times, [0.1, 0.9]) ./ 1e6
    best, middle = minimum(trial), median(trial)
    row = (kind, backend, opts.T, opts.particles, join(opts.grid, 'x'), opts.gamma,
           method, bounds, cache, length(trial), best.time / 1e6, middle.time / 1e6,
           low, high, best.memory, best.allocs,
           repr(case.mesh.min_bounds), repr(case.mesh.max_bounds))
    println(io, join(row, '\t'))
    flush(io)
    println("$backend $method bounds=$bounds cache=$cache: median=$(round(middle.time / 1e6; digits=3)) ms, min=$(round(best.time / 1e6; digits=3)) ms ($(length(trial)) samples)")
end

const RESULT_HEADER = "stage\tbackend\tprecision\tparticles\tgrid\tgamma\tmethod\tbounds\tcache\tsamples\tmin_ms\tmedian_ms\tp10_ms\tp90_ms\thost_bytes\thost_allocs\tmin_bounds\tmax_bounds"
