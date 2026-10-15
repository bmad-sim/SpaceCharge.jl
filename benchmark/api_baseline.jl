# Unchanged release comparison. Export v1.3.0 with git archive to a separate
# directory and set SPACECHARGE_BASELINE to it. Run in an environment containing
# this checkout, BenchmarkTools, and (optionally) CUDA. No checkout is modified.
using SpaceCharge, FFTW, Random, Statistics, BenchmarkTools, KernelAbstractions

module Released
include(joinpath(ENV["SPACECHARGE_BASELINE"], "src", "SpaceCharge.jl"))
end
const Old = Released.SpaceCharge

const GPU_RUN = "--gpu" in ARGS
if GPU_RUN
    @eval using CUDA
    CUDA.functional(true)
    CUDA.allowscalar(false)
    @eval device_bytes(f) = CUDA.@allocated f()
else
    device_bytes(f) = missing
end
FFTW.set_num_threads(parse(Int, get(ENV, "SPACECHARGE_FFT_THREADS", "1")))

function report(label, a, b, synchronize)
    old = () -> (a(); synchronize(); nothing)
    new = () -> (b(); synchronize(); nothing)
    old(); new(); old(); new()
    # Alternate paired samples to reduce thermal/order bias. Setup and compilation
    # are excluded. Report distribution and constant host allocation separately.
    ta = Float64[]; tb = Float64[]
    for k in 1:30
        if isodd(k)
            push!(ta, @elapsed old()); push!(tb, @elapsed new())
        else
            push!(tb, @elapsed new()); push!(ta, @elapsed old())
        end
    end
    ba = @benchmark $old() samples=10 evals=1
    bb = @benchmark $new() samples=10 evals=1
    println(label, " old_ms=", round(median(ta)*1000;digits=3),
        " new_ms=", round(median(tb)*1000;digits=3),
        " ratio=", round(median(tb)/median(ta);digits=3),
        " old_p10_p90=", round.(quantile(ta,[.1,.9])*1000;digits=3),
        " new_p10_p90=", round.(quantile(tb,[.1,.9])*1000;digits=3),
        " host_bytes=", (minimum(ba).memory,minimum(bb).memory),
        " device_bytes=", (device_bytes(old),device_bytes(new)))
end

function run_case(T, n, count)
    rng=MersenneTwister(42)
    coords=ntuple(_ -> randn(rng,T,count).*T(.001),3)
    q=fill(T(1e-9/count),count)
    backend=GPU_RUN ? CUDABackend() : CPU()
    adapt=GPU_RUN ? CuArray : identity
    sync=GPU_RUN ? CUDA.synchronize : ()->nothing
    x,y,z=map(adapt,coords); q=adapt(q)
    bounds=(T.((-.01,-.01,-.01)),T.((.01,.01,.01)))
    a=Old.Mesh3D(n,bounds...;T,backend)
    b=FFTIGF(n;T,backend,bounds,fft_flags=FFTW.ESTIMATE)
    oa=ntuple(_->similar(x),3); ob=ntuple(_->similar(x),3)
    oldsolve=()->(Old.deposit!(a,x,y,z,q);Old.solve!(a;fft_flags=FFTW.ESTIMATE))
    newsolve=()->solve!(b,x,y,z,q)
    oldfield=()->Old.interpolate_field!(oa...,a,x,y,z)
    newfield=()->field!(ob...,b,x,y,z)
    setup_old=@elapsed (oldsolve();sync())
    setup_new=@elapsed (newsolve();sync())
    oldfield();newfield();sync()
    tol=T===Float32 ? 3e-5 : 3e-12
    @assert all(isapprox(Array(oa[i]),Array(ob[i]);rtol=tol) for i in 1:3)
    println("case T=$T grid=$n N=$count gpu=$GPU_RUN first_prepare_s=",(setup_old,setup_new))
    report("fixed prepare",oldsolve,newsolve,sync)
    report("staged solve",()->Old.solve!(a;fft_flags=FFTW.ESTIMATE),()->solve!(b),sync)
    report("field",oldfield,newfield,sync)
    report("pipeline",()->(oldsolve();oldfield()),()->(newsolve();newfield()),sync)
    # Exactly changing gamma forces spectrum regeneration on every measured call.
    ga=Ref(T(1));gb=Ref(T(1))
    report("changing gamma",
        ()->(ga[]+=T(.001);a.gamma=ga[];oldsolve()),
        ()->(gb[]+=T(.001);solve!(b,x,y,z,q;gamma=gb[])),sync)
    if T == Float64
        adaptive=FFTIGF(n;T,backend,fft_flags=FFTW.ESTIMATE)
        solve!(adaptive,x,y,z,q)
        # Release has no combined refit operation. Recompute its original bounds
        # formula in-place so the comparison does not charge it mesh allocations.
        function oldfit!()
            ex,ey,ez=extrema(x),extrema(y),extrema(z)
            lo=(ex[1],ey[1],ez[1]);hi=(ex[2],ey[2],ez[2])
            d=(hi.-lo)./(n.-1)
            a.min_bounds=T.(lo .- 1e-6.*d);a.max_bounds=T.(hi .+ 1e-6.*d)
            a.delta=T.((a.max_bounds.-a.min_bounds)./(n.-1));a.gamma=one(T)
            oldsolve()
        end
        report("adaptive prepare",oldfit!,()->solve!(adaptive,x,y,z,q),sync)
        # Mutate the source geometry without allocating a new particle batch.
        report("changing geometry",
            ()->(x .*= T(1.00001);oldfit!()),
            ()->(x .*= T(1.00001);solve!(adaptive,x,y,z,q)),sync)
    else
        println("Float32 adaptive baseline skipped: released padding can round away, causing out-of-bounds deposition.")
    end
end

println("Julia=",VERSION," Julia_threads=",Threads.nthreads()," FFTW_threads=",FFTW.get_num_threads())
for T in (Float64,Float32)
    run_case(T,(32,32,32),100_000)
end
"--large" in ARGS && run_case(Float64,(64,64,64),1_000_000)
