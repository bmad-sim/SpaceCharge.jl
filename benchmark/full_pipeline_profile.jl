# Profile the warmed public solve!(particles...) + field! pipeline.
#   julia --startup-file=no --threads=1 --project=benchmark benchmark/full_pipeline_profile.jl --help
# Reports go to a fresh directory in the Julia depot's scratchspaces by default.

using SpaceCharge
using CUDA
using FFTW
using KernelAbstractions: get_backend
using Random
using Profile
using Dates
using Logging

function print_usage()
    println("""
    Usage: julia --startup-file=no --threads=1 --project=benchmark benchmark/full_pipeline_profile.jl [options]
      --backend cpu|gpu|both       Default: both (also --cpu, --gpu, --both)
      --method direct|potential_fd Default: direct
      --particles N               Default: 100000
      --grid N or NX,NY,NZ         Default: 64 (64,64,64)
      --precision Float32|Float64  Default: Float64
      --gamma G                   Default: 10
      --bounds adaptive|explicit   Default: adaptive
      --cache cached|rebuilt       Default: cached
      --fft-threads N              Default: 1
      --seconds S                 CPU sampling duration, default: 5
      --gpu-repeats N              GPU summary repetitions, default: 20
      --out DIRECTORY             Parent for a new, preserved run directory
      --help                      Show this help

    Each run saves metadata, CPU flat/tree reports, and/or a GPU summary plus
    a separate one-pipeline trace. Setup, transfers, FFT planning, compilation,
    and warmup precede capture. Particles stay fixed; explicit bounds reuse
    the initially fitted box. Rebuilt mode disables Green caching while
    retaining FFT plans and mesh storage. Free space is used. Profile times include
    instrumentation overhead and are NOT clean benchmark measurements.
    Windows CPU sampling covers only the main thread; prefer --threads=1.
    GPU profiling must be able to launch CUDA.jl's bundled demumble executable.
    """)
end

function parse_options(args)
    values = Dict("backend" => "both", "method" => "direct", "particles" => "100000",
                  "grid" => "64", "precision" => "Float64", "gamma" => "10",
                  "bounds" => "adaptive", "cache" => "cached", "fft-threads" => "1",
                  "seconds" => "5", "gpu-repeats" => "20",
                  "out" => joinpath(first(DEPOT_PATH), "scratchspaces", "spacecharge-profiles"))
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
    values["backend"] in ("cpu", "gpu", "both") || error("Invalid backend")
    values["method"] in ("direct", "potential_fd") || error("Invalid field method")
    values["bounds"] in ("adaptive", "explicit") || error("Invalid bounds mode")
    values["cache"] in ("cached", "rebuilt") || error("Invalid cache mode")
    values["precision"] in ("Float32", "Float64") || error("Invalid precision")
    dims = parse.(Int, split(values["grid"], ','))
    length(dims) == 1 && (dims = fill(only(dims), 3))
    length(dims) == 3 && all(>=(2), dims) || error("Grid needs three dimensions >= 2")
    particles = parse(Int, values["particles"])
    seconds = parse(Float64, values["seconds"])
    repeats = parse(Int, values["gpu-repeats"])
    fft_threads = parse(Int, values["fft-threads"])
    T = values["precision"] == "Float64" ? Float64 : Float32
    gamma = parse(T, values["gamma"])
    particles > 0 || error("Particle count must be positive")
    isfinite(seconds) && seconds > 0 || error("Seconds must be finite and positive")
    repeats > 0 || error("GPU repetitions must be positive")
    fft_threads > 0 || error("FFTW threads must be positive")
    isfinite(gamma) && gamma >= 1 || error("Gamma must be finite and >= 1")
    return (; backend=values["backend"], method=Symbol(values["method"]), particles,
            grid=Tuple(dims), T, gamma, bounds=values["bounds"], cache=values["cache"],
            fft_threads, seconds, repeats, out=abspath(values["out"]))
end

function prepare_case(opts, gpu)
    rng = MersenneTwister(42)
    xyz = ntuple(_ -> randn(rng, opts.T, opts.particles) .* opts.T(0.001), 3)
    q = fill(opts.T(1e-9) / opts.particles, opts.particles)
    if gpu
        xyz = CuArray.(xyz)
        q = CuArray(q)
    end
    mesh = FFTIGF(opts.grid; T=opts.T, backend=get_backend(q), field_method=opts.method,
                  cache_green=opts.cache == "cached")
    outputs = ntuple(_ -> similar(q), 3)
    bounds = nothing
    if opts.bounds == "explicit"
        solve!(mesh, xyz..., q, opts.gamma)
        bounds = (mesh.min_bounds, mesh.max_bounds)
    end
    return (; mesh, xyz, q, outputs, gamma=opts.gamma, bounds)
end

function pipeline!(case)
    solve!(case.mesh, case.xyz..., case.q, case.gamma; bounds=case.bounds)
    field!(case.outputs..., case.mesh, case.xyz...)
    return nothing
end

function repeat_pipeline!(case, repeats, gpu)
    for _ in 1:repeats
        pipeline!(case)
        gpu && CUDA.synchronize()
    end
end

function sample_cpu!(case, seconds)
    start = time_ns()
    repeats = 0
    while (time_ns() - start) / 1e9 < seconds
        pipeline!(case)
        repeats += 1
    end
    return repeats
end

function profile_cpu(case, opts, dir, log)
    # Compile the sampling loop itself before enabling the sampler.
    sample_cpu!(case, 0.01)
    Profile.init(n=10_000_000, delay=0.001)
    Profile.clear()
    GC.gc()
    repeats = Profile.@profile sample_cpu!(case, opts.seconds)
    data, frames = Profile.retrieve()
    samples = count(iszero, Profile.fetch(include_meta=false))
    full = Profile.is_buffer_full()
    println(log, "CPU repetitions=$repeats samples=$samples buffer_full=$full interval_seconds=0.001")
    full && println(log, "WARNING: CPU capture truncated: sample buffer filled.")
    samples < 100 && println(log, "WARNING: Sparse CPU capture; increase --seconds.")
    for format in (:flat, :tree)
        open(joinpath(dir, "cpu-$format.txt"), "w") do io
            println(io, "CPU $format: repetitions=$repeats samples=$samples buffer_full=$full")
            println(io, "Includes C/FFTW frames; counts are samples, not call counts. Largest flat counts are last.")
            Sys.iswindows() && println(io, "Windows: only the main thread was sampled.")
            println(io, "All rows/depths retained; lines wider than 240 columns may be abbreviated.")
            Profile.print(IOContext(io, :displaysize => (10000, 240), :color => false),
                          data, frames; format, C=true, sortedby=:count)
        end
    end
    samples > 0 || error("CPU profiler collected no samples")
    println("CPU captured $samples samples over $repeats pipelines.")
end

function profile_gpu(case, opts, dir, log)
    # CUDA.jl 5.9 returns ProfileResults; a script must explicitly render it.
    for (label, repeats, trace) in (("summary", opts.repeats, false), ("trace", 1, true))
        GC.gc()
        CUDA.synchronize()
        result = CUDA.@profile external=false trace=trace raw=!trace repeat_pipeline!(case, repeats, true)
        any(!ismissing, result.device.grid) || error("CUPTI recorded no GPU kernels")
        # CUPTI's global activity buffers use the NULL context and stream 0.
        dropped = Ref{Csize_t}(0)
        CUDA.CUPTI.cuptiActivityGetNumDroppedRecords(C_NULL, 0, dropped)
        println(log, "GPU $label repetitions=$repeats dropped_records=$(dropped[])")
        dropped[] > 0 && println(log, "WARNING: CUPTI dropped records; this capture is incomplete.")
        open(joinpath(dir, "gpu-$label.txt"), "w") do io
            println(io, "GPU $label: repetitions=$repeats dropped_records=$(dropped[])")
            println(io, "Host API time and device activity overlap; do not add them or use as benchmark timings.")
            println(io, "Each pipeline ends with CUDA.synchronize(); summary excludes profiler setup.")
            println(io, "Summary retains all API calls; trace uses CUDA.jl's setup/query filtering. No display cropping.")
            show(IOContext(io, :limit => false, :color => false), result)
            println(io)
        end
        println("GPU $label captured; dropped records: $(dropped[]).")
    end
end

function main(args=ARGS)
    if "--help" in args
        print_usage()
        return 0
    end
    opts = parse_options(args)
    FFTW.set_num_threads(opts.fft_threads)
    repo = dirname(@__DIR__)
    realpath(dirname(dirname(pathof(SpaceCharge)))) == realpath(repo) ||
        error("The active environment must develop this SpaceCharge checkout: $repo")
    mkpath(opts.out)
    dir = mktempdir(opts.out; prefix=Dates.format(now(), "yyyymmdd-HHMMSS") * "-", cleanup=false)
    println("Profile outputs: $dir")
    open(joinpath(dir, "metadata.txt"), "w") do io
        println(io, "Date (UTC): $(now(UTC))")
        println(io, "HEAD: ", readchomp(`git -C $repo rev-parse HEAD`))
        println(io, "Working tree:\n", readchomp(`git -C $repo status --short`))
        println(io, "Julia: $VERSION executable=$(joinpath(Sys.BINDIR, Base.julia_exename()))")
        println(io, "Project: $(Base.active_project())")
        println(io, "Source: $(pathof(SpaceCharge))")
        println(io, "Script: $(@__FILE__)\nArguments: $(repr(args))")
        println(io, "CPU: $(Sys.cpu_info()[1].model)")
        println(io, "Julia threads=$(Threads.nthreads()) FFTW threads=$(FFTW.get_num_threads()) FFTW=$(pkgversion(FFTW))")
        println(io, "CUDA.jl: $(pkgversion(CUDA))")
        println(io, "Options: $opts")
        println(io, "Workload: fixed Gaussian particles, seed=42, sigma=0.001 m, total_charge=1e-9 C")
        println(io, "Pipeline: solve!(mesh,x,y,z,q,gamma; bounds); field!(Ex,Ey,Ez,mesh,x,y,z)")
        println(io, "Solver: cache_green=$(opts.cache == "cached"), at_cathode=false, pruned_fft=true (CPU only), FFTW.MEASURE")
        println(io, "Five warmup pipelines, including compilation/planning/cache creation, precede capture.")
        println(io, "Profiled times include instrumentation overhead; these are not clean benchmarks.")
        Sys.iswindows() && println(io, "WARNING: Julia CPU profiling on Windows samples only the main thread.")
    end
    success = true
    backends = opts.backend == "both" ? ("cpu", "gpu") : (opts.backend,)
    for backend in backends
        println("Preparing $backend / $(opts.method) ...")
        flush(stdout)
        open(joinpath(dir, "$backend-log.txt"), "w") do log
            with_logger(SimpleLogger(log)) do
                try
                    gpu = backend == "gpu"
                    if gpu
                        CUDA.functional() || error("CUDA is not functional")
                        CUDA.allowscalar(false)
                        CUDA.versioninfo(log)
                    end
                    case = prepare_case(opts, gpu)
                    repeat_pipeline!(case, 5, gpu)
                    println(log, "Warmup complete; bounds=$(case.mesh.min_bounds), $(case.mesh.max_bounds)")
                    flush(log)
                    if gpu
                        profile_gpu(case, opts, dir, log)
                    else
                        profile_cpu(case, opts, dir, log)
                    end
                    println(log, "SUCCESS: capture complete.")
                catch err
                    success = false
                    println(log, "FAILED: ")
                    # A spawn error can embed the entire process environment in its message.
                    if err isa Base.IOError && occursin("could not spawn", err.msg)
                        println(log, "Subprocess launch failed (IOError code=$(err.code)); environment omitted.")
                        Base.show_backtrace(log, catch_backtrace())
                    else
                        showerror(log, err, catch_backtrace())
                    end
                    println(log)
                    println(stderr, "$backend profiling failed; see $(joinpath(dir, "$backend-log.txt"))")
                end
            end
        end
    end
    println("Reports saved: $dir")
    return success ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
