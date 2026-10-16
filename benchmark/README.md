# Benchmarks and profiling

Run from the repository root with the benchmark environment, which must develop
this checkout. Install its dependencies once:

```sh
julia --startup-file=no --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

The default workload is 100,000 fixed Gaussian particles, a 64×64×64 grid,
Float64, and gamma 10. Every case uses seed 42, sigma 0.001 m on each lab-frame
axis, and total charge 1e-9 C. CPU and GPU use the same host-generated arrays.
The scripts use free space and the solver's default FFT planning and transform
options. CPU FFTW threads default to one; set Julia threads at startup.

## Timing

```sh
julia --startup-file=no --threads=1 --project=benchmark benchmark/full_pipeline_benchmark.jl
julia --startup-file=no --threads=1 --project=benchmark benchmark/solve_benchmark.jl
julia --startup-file=no --threads=1 --project=benchmark benchmark/deposit_benchmark.jl
```

| Script | Timed operation | Default cases |
| --- | --- | --- |
| `full_pipeline_benchmark.jl` | Public `solve!(mesh, x, y, z, q, gamma; bounds)` then `field!` at the particles | CPU/GPU × direct/potential_fd × adaptive/explicit bounds × cached/rebuilt spectra |
| `solve_benchmark.jl` | `solve!(mesh, gamma)` on an already deposited grid | CPU/GPU × direct/potential_fd × cached/rebuilt spectra |
| `deposit_benchmark.jl` | `deposit!` on established geometry, including grid clearing | CPU/GPU; deposition is shared by both field methods |

Adaptive bounds refit the fixed particles on every call. Explicit bounds reuse
the initially fitted box, so both modes use the same geometry. Cached cases
reuse warmed Green spectra with fixed geometry and gamma. Rebuilt cases set
`cache_green=false`, regenerating Green spectra on every call while reusing the
mesh, workspace, and FFT plans. This isolates recurring spectrum cost; it does
not measure cold startup or a moving bunch. The diagnostic solve and deposition
timings exclude bounds fitting and particle interpolation.

All setup, transfers, compilation, FFT planning, and five warmups precede timing.
Each BenchmarkTools sample performs one operation; GPU samples include a final
`CUDA.synchronize()`. Results report median, minimum, p10/p90 time and host
allocations; host allocations do not measure GPU memory. Sampling stops at the
requested sample count or time budget. Run cases serially on an otherwise idle
machine and compare rows with matching workload/options. Do not average speedups
across unrelated cases. Cases run in a fixed order; repeat close comparisons with
separate invocations in reversed order before drawing performance conclusions.
With `--backend both`, unavailable CUDA is reported and
CPU cases still run; explicit `--backend gpu` fails if CUDA is unavailable.

Use `--help` for the supported options. For example:

```sh
julia --startup-file=no --threads=1 --project=benchmark benchmark/full_pipeline_benchmark.jl --method potential_fd --bounds adaptive --cache cached --samples 200 --seconds 5
julia --startup-file=no --threads=8 --project=benchmark benchmark/full_pipeline_benchmark.jl --cpu --fft-threads 8 --particles 1000000 --grid 128 --precision Float32
```

Each invocation uses one particle count/grid/precision. Larger sweeps are
explicit separate invocations, rather than the default workload.

## Profiling

```sh
julia --startup-file=no --threads=1 --project=benchmark benchmark/full_pipeline_profile.jl --cpu --seconds 5
julia --startup-file=no --threads=1 --project=benchmark benchmark/full_pipeline_profile.jl --gpu --method potential_fd --bounds explicit --cache rebuilt --gpu-repeats 20
```

The profiler uses the same workload defaults and bounds/cache meanings, selecting
one field method (default `direct`), bounds mode (default `adaptive`), and cache
mode (default `cached`) per capture. It preserves CPU flat/tree plaintext reports
and CUDA summary plus one-pipeline trace reports. Profiling includes
instrumentation overhead; use the timing scripts for performance comparisons.
On Windows the CPU sampler only samples the main thread, so prefer `--threads=1`.
CUDA profiling requires permission to execute CUDA.jl's bundled `demumble` helper;
run in an approved environment without changing security settings.

## Results and correctness

Each invocation saves a fresh directory under the Julia depot's `scratchspaces`
(`spacecharge-benchmarks` or `spacecharge-profiles`). These are outside this Git
checkout. `--out` selects another parent directory; keep generated results outside
the checkout. Metadata records commit/dirty state, source and environment paths,
hardware, package versions, Julia/FFTW threads, precision, particle count, grid,
gamma, field method, bounds and cache choices. Benchmark rows also record the
actual fitted bounds. Profiler logs record capture status and dropped records.

Correctness belongs in `test/`: Gaussian accuracy is covered by
`analytical_test.jl`, deposition conservation by `test_deposition.jl` and
`test_threaded_deposition.jl`, independent field/cache references by
`test_solver_reference.jl` and `test_potential_fd.jl`, and GPU/API comparisons by
`test_gpu.jl` and `test_potential_accuracy.jl`. Benchmarks do not calculate analytic
errors, charge conservation, or field correctness metrics.

The release-specific `api_baseline.jl` runner is retired. Release tags, Git
history, and external historical measurements remain the sources for historical
comparisons. Impact-TJ comparisons belong in a separate optional environment and
harness, with matched physics and workloads; Impact-TJ is not a dependency or
default run of these benchmarks.
