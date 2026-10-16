# Diagnostic benchmark: convolve an already deposited charge grid.
include("benchmark_common.jl")

function benchmark_solve(io, opts, backend, method, cache)
    gpu = backend == "gpu"
    case = prepare_benchmark(opts, gpu, method, cache)
    trial = measure_benchmark(() -> solve!(case.mesh, case.gamma), gpu, opts)
    write_result(io, "solve", backend, method, "fixed", cache, opts, case, trial)
end

function main(args=ARGS)
    if "--help" in args
        benchmark_usage("solve")
        return
    end
    opts = benchmark_options(args, "solve")
    backends = benchmark_backends(opts)
    dir = benchmark_output(opts, "solve", args, backends)
    methods = opts.method == "both" ? ("direct", "potential_fd") : (opts.method,)
    caches = opts.cache == "both" ? ("cached", "rebuilt") : (opts.cache,)
    open(joinpath(dir, "results.tsv"), "w") do io
        println(io, RESULT_HEADER)
        for method in methods, cache in caches, backend in backends
            benchmark_solve(io, opts, backend, method, cache)
            GC.gc()
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
