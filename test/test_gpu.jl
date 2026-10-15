using SpaceCharge, Test, CUDA, FFTW, Random

function run_gpu_tests()
    CUDA.allowscalar(false)
    @testset "CUDA array API and independent reference" begin
        for T in (Float32,Float64), cathode in (false,true)
            ref=reference_test_mesh(T,(5,6,4),10.)
            gpu=FFTIGF(ref.grid_size;T,backend=CUDABackend(),
                bounds=(ref.min_bounds,ref.max_bounds),at_cathode=cathode)
            cpu=FFTIGF(ref.grid_size;T,bounds=(ref.min_bounds,ref.max_bounds),
                at_cathode=cathode,fft_flags=FFTW.ESTIMATE)
            # Deposit nodal sources through the new public stages.
            coords=ntuple(d->collect(range(ref.min_bounds[d],ref.max_bounds[d];length=ref.grid_size[d])),3)
            points=collect(Iterators.product(coords...))
            x,y,z=ntuple(d->vec(getindex.(points,d)),3);q=vec(copy(ref.rho))
            dx,dy,dz,dq=CuArray.((x,y,z,q))
            @test deposit!(gpu,dx,dy,dz,dq) === gpu
            deposit!(cpu,x,y,z,q)
            for gamma in (10.,10.,11.3,1.)
                @test solve!(gpu;gamma) === gpu
                solve!(cpu;gamma)
                CUDA.synchronize()
                ref.rho .= cpu.mesh.rho;ref.gamma=T(gamma)
                copyto!(ref.efield,Array(gpu.mesh.efield))
                test_field_against_reference(ref;at_cathode=cathode)
                targets=(T[0,0.01],T[0,0.01],T[0.002,0.003])
                dtargets=CuArray.(targets)
                fields=CUDA.zeros(T,2,3);out=ntuple(i->view(fields,:,i),3)
                expected=ntuple(_->zeros(T,2),3)
                @test field!(out...,gpu,dtargets...) === out
                field!(expected...,cpu,targets...)
                tol=T==Float32 ? 5e-4 : 3e-10
                @test all(isapprox(Array(out[i]),expected[i];rtol=tol) for i in 1:3)
                fill!(fields,T(NaN));field!(out...,gpu,dtargets...)
                @test all(isfinite,Array(fields))
                @test_throws ArgumentError field!(expected...,gpu,dtargets...)
            end
            @test_throws ArgumentError solve!(gpu,x,y,z,q)
            @test !gpu.prepared
        end
    end
    @testset "CUDA adaptive masks and reusable storage" begin
        for T in (Float32,Float64)
            rng=MersenneTwister(71);n=5001
            host=ntuple(_->rand(rng,T,n).*T(.002),3)
            q=fill(T(1e-12/n),n); mask=fill(true,n); mask[3:7:end].=false
            for a in (host...,q);a[.!mask].=T(NaN);end
            src=CuArray.((host...,q));active=CuArray(mask)
            gpu=FFTIGF((8,9,10);T,backend=CUDABackend())
            cpu=FFTIGF((8,9,10);T,fft_flags=FFTW.ESTIMATE)
            solve!(gpu,src...;active,gamma=2); solve!(cpu,host...,q;active=mask,gamma=2)
            out=ntuple(_->similar(src[1]),3);expected=ntuple(_->zeros(T,n),3)
            field!(out...,gpu,src[1:3]...;active)
            field!(expected...,cpu,host...;active=mask)
            tol=T==Float32 ? 2e-4 : 1e-10
            @test all(isapprox(Array(out[i]),expected[i];rtol=tol) for i in 1:3)
            @test all(all(iszero,Array(a)[.!mask]) for a in out)
            summary=gpu.extent_device; workspace=gpu.mesh._workspace
            frozen=Array.(out)
            for a in src;fill!(a,T(NaN));end
            targets=CuArray.(host)
            field!(out...,gpu,targets...;active)
            @test Array.(out) == frozen
            @test_throws ArgumentError solve!(gpu,src...;active)
            @test !gpu.prepared
            @test_throws ArgumentError field!(out...,gpu,targets...)
            fill!(active,false);solve!(gpu,src...;active)
            field!(out...,gpu,targets...;active)
            @test all(a->all(iszero,Array(a)),out)
            empty=CUDA.zeros(T,0)
            solve!(gpu,empty,empty,empty,empty)
            @test field!(empty,empty,empty,gpu,empty,empty,empty) === (empty,empty,empty)
            src=CuArray.((host...,q));active=CuArray(mask)
            solve!(gpu,src...;active)
            @test gpu.extent_device === summary
            @test gpu.mesh._workspace === workspace
            @test_throws ArgumentError solve!(gpu,src...;active=mask)
            @test_throws ArgumentError solve!(gpu,src...;gamma=Inf)
        end
    end
end
