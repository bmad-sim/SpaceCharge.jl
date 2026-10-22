using SpaceCharge, Test, CUDA, FFTW, Random

function run_gpu_tests()
    CUDA.allowscalar(false)
    test_float32_adaptive_endpoints(CUDABackend())
    test_cathode_geometry(CUDABackend())
    @testset "CUDA direct fields against independent reference" begin
        for T in (Float32,Float64), cathode in (false,true)
            gpu = reference_test_mesh(T,(5,6,4),10.;backend=CUDABackend(),at_cathode=cathode)
            snapshot = (rho=Array(gpu.rho),efield=Array(gpu.efield),delta=gpu.delta,
                        gamma=gpu.gamma,min_bounds=gpu.min_bounds,max_bounds=gpu.max_bounds)
            test_field_against_reference(snapshot;at_cathode=cathode)
        end
    end
    @testset "CUDA masked arrays and strided views" begin
        for T in (Float32,Float64)
            rng=MersenneTwister(71);n=5001
            host=ntuple(_->rand(rng,T,n).*T(.002),3)
            q=fill(T(1e-12/n),n); host_mask=fill(true,n); host_mask[3:7:end].=false
            for a in (host...,q);a[.!host_mask].=T(NaN);end
            particles=CuArray.((host...,q));mask=CuArray(host_mask)
            gpu=FFTIGF((8,9,10);T,backend=CUDABackend())
            cpu=FFTIGF((8,9,10);T,fft_flags=FFTW.ESTIMATE)
            solve!(gpu,particles...,2;mask)
            solve!(cpu,host...,q,2;mask=host_mask)
            out=ntuple(_->similar(particles[1]),3);expected=ntuple(_->zeros(T,n),3)
            field!(out...,gpu,particles[1:3]...;mask)
            field!(expected...,cpu,host...;mask=host_mask)
            tol=T==Float32 ? 2e-4 : 1e-10
            @test all(isapprox(Array(out[i]),expected[i];rtol=tol) for i in 1:3)
            @test all(all(iszero,Array(a)[.!host_mask]) for a in out)
            # Strided coordinate/mask views must retain the same reduction semantics.
            strided=ntuple(4) do i
                a=CUDA.zeros(T,2n); a[1:2:end] .= particles[i]
                view(a,1:2:2n)
            end
            mask_storage=CUDA.zeros(Bool,2n); mask_storage[1:2:end] .= mask
            mask_view=view(mask_storage,1:2:2n)
            solve!(gpu,strided...,2;mask=mask_view)
            field!(out...,gpu,particles[1:3]...;mask)
            @test all(isapprox(Array(out[i]),expected[i];rtol=tol) for i in 1:3)
            fill!(mask,false)
            field!(out...,gpu,particles[1:3]...;mask)
            @test all(a->all(iszero,Array(a)),out)
            fill!(mask,true)
            @test_throws ArgumentError solve!(gpu,particles...,1;mask=host_mask)
            @test_throws DimensionMismatch solve!(gpu,particles...,1;mask=CUDA.ones(Bool,1))
            @test_throws ArgumentError field!(expected...,gpu,particles[1:3]...)
        end
    end
end
