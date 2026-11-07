using SpaceCharge, Test, CUDA, FFTW, Random

function run_gpu_tests()
    CUDA.allowscalar(false)
    test_float32_adaptive_endpoints(CUDABackend())
    test_cathode_geometry(CUDABackend())
    @testset "CUDA CIC faces, masks and accumulation" begin
        for T in (Float32,Float64)
            gpu = FFTIGF((3,5,9);T,backend=CUDABackend())
            x = T[0,1,.25,0,1,NaN]
            y = T[0,1,.375,1,0,NaN]
            z = T[0,1,.4375,1,0,NaN]
            q = T[1,-2,4,8,-4,NaN]
            mask = CuArray(Bool[1,1,1,1,1,0])
            particles = CuArray.((x,y,z,q))
            expected = zeros(T,3,5,9)
            expected[1,1,1] = 1; expected[3,5,9] = -2
            expected[1:2,2:3,4:5] .= T(.5)
            expected[1,5,9] = 8; expected[3,1,1] = -4
            solve!(gpu,particles...,1;mask,bounds=((0,0,0),(1,1,1)))
            @test Array(gpu.rho) == expected
            @test deposit!(gpu,particles...;mask,clear=false) === gpu
            @test Array(gpu.rho) == 2expected
            deposit!(gpu,particles...;mask)
            @test Array(gpu.rho) == expected
            # Inactive NaNs and empty additions leave existing charge unchanged.
            fill!(mask,false)
            deposit!(gpu,particles...;mask,clear=false)
            @test Array(gpu.rho) == expected
            empty = CUDA.zeros(T,0)
            deposit!(gpu,empty,empty,empty,empty;clear=false)
            @test Array(gpu.rho) == expected
            @test_throws ArgumentError deposit!(gpu,particles...;mask)
        end
    end
    @testset "CUDA contended signed deposition" begin
        for T in (Float32,Float64)
            gpu = FFTIGF((5,5,5);T,backend=CUDABackend())
            n = 83_017
            x,y,z = (CUDA.fill(T(v),n) for v in (.125,.375,.625))
            q = CuArray([isodd(i) ? T(2.0^-30) : -T(2.0^-31) for i in 1:n])
            solve!(gpu,x,y,z,q,1;bounds=((0,0,0),(1,1,1)))
            expected = zeros(T,5,5,5)
            expected[1:2,2:3,3:4] .= sum(Array(q))/8
            @test Array(gpu.rho) == expected
            deposit!(gpu,x,y,z,q;clear=false)
            @test Array(gpu.rho) == 2expected
            deposit!(gpu,x,y,z,-q)
            @test Array(gpu.rho) == -expected
        end
    end
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
