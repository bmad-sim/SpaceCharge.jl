# Main benchmark: public particle solve! followed by field! at those particles.
include("benchmark_common.jl")

function pipeline!(case, bounds)
    solve!(case.mesh, case.xyz..., case.q, case.gamma; bounds)
    field!(case.outputs..., case.mesh, case.xyz...)
    return nothing
end

function benchmark_pipeline(io, opts, backend, method, cache)
    gpu = backend == "gpu"
    case = prepare_benchmark(opts, gpu, method, cache)
    modes = opts.bounds == "both" ? ("adaptive", "explicit") : (opts.bounds,)
    for mode in modes
        bounds = mode == "explicit" ? case.fitted_bounds : nothing
        trial = measure_benchmark(() -> pipeline!(case, bounds), gpu, opts)
        write_result(io, "full_pipeline", backend, method, mode, cache, opts, case, trial)
    end
end

function main(args=ARGS)
    if "--help" in args
        benchmark_usage("full_pipeline")
        return
    end
    opts = benchmark_options(args, "full_pipeline")
    backends = benchmark_backends(opts)
    dir = benchmark_output(opts, "full_pipeline", args, backends)
    methods = opts.method == "both" ? ("direct", "potential_fd") : (opts.method,)
    caches = opts.cache == "both" ? ("cached", "rebuilt") : (opts.cache,)
    open(joinpath(dir, "results.tsv"), "w") do io
        println(io, RESULT_HEADER)
        for method in methods, cache in caches, backend in backends
            benchmark_pipeline(io, opts, backend, method, cache)
            GC.gc()
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
