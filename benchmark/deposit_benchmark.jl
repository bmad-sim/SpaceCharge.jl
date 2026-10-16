# Diagnostic benchmark: shared deposition, including the default grid clearing.
include("benchmark_common.jl")

function benchmark_deposit(io, opts, backend)
    gpu = backend == "gpu"
    case = prepare_benchmark(opts, gpu, "direct", "cached")
    trial = measure_benchmark(() -> deposit!(case.mesh, case.xyz..., case.q), gpu, opts)
    # Field method and Green cache do not participate in deposit!.
    write_result(io, "deposit", backend, "shared", "fixed", "n/a", opts, case, trial)
end

function main(args=ARGS)
    if "--help" in args
        benchmark_usage("deposit")
        return
    end
    opts = benchmark_options(args, "deposit")
    backends = benchmark_backends(opts)
    dir = benchmark_output(opts, "deposit", args, backends)
    open(joinpath(dir, "results.tsv"), "w") do io
        println(io, RESULT_HEADER)
        for backend in backends
            benchmark_deposit(io, opts, backend)
            GC.gc()
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
