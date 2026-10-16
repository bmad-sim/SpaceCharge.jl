using SpaceCharge, Test, CUDA, FFTW, Random

function run_gpu_tests()
    CUDA.allowscalar(false)
    test_float32_adaptive_endpoints(CUDABackend())
    test_cathode_geometry(CUDABackend())
    @testset "CUDA array API and independent reference" begin
        for T in (Float32,Float64), cathode in (false,true)
            ref=reference_test_mesh(T,(5,6,4),10.)
            bounds=(ref.min_bounds,ref.max_bounds)
            gpu=FFTIGF(ref.grid_size;T,backend=CUDABackend(),at_cathode=cathode)
            cpu=FFTIGF(ref.grid_size;T,at_cathode=cathode,fft_flags=FFTW.ESTIMATE)
            # Deposit particles at grid nodes through the new public stages.
            coords=ntuple(d->collect(range(ref.min_bounds[d],ref.max_bounds[d];length=ref.grid_size[d])),3)
            points=collect(Iterators.product(coords...))
            x,y,z=ntuple(d->vec(getindex.(points,d)),3);q=vec(copy(ref.rho))
            dx,dy,dz,dq=CuArray.((x,y,z,q))
            @test_throws MethodError solve!(gpu,dx,dy,dz,dq)
            @test_throws MethodError solve!(gpu)
            @test solve!(gpu,dx,dy,dz,dq,1;bounds) === nothing
            solve!(cpu,x,y,z,q,1;bounds)
            @test deposit!(gpu,dx,dy,dz,dq) === gpu
            deposit!(cpu,x,y,z,q)
            for gamma in (10.,10.,11.3,1.)
                @test solve!(gpu,gamma) === nothing
                solve!(cpu,gamma)
                CUDA.synchronize()
                ref.rho .= cpu.rho;ref.gamma=T(gamma)
                copyto!(ref.efield,Array(gpu.efield))
                test_field_against_reference(ref;at_cathode=cathode)
                evaluation_points=(T[0,0.01],T[0,0.01],T[0.002,0.003])
                device_points=CuArray.(evaluation_points)
                fields=CUDA.zeros(T,2,3);out=ntuple(i->view(fields,:,i),3)
                expected=ntuple(_->zeros(T,2),3)
                @test field!(out...,gpu,device_points...) === out
                field!(expected...,cpu,evaluation_points...)
                tol=T==Float32 ? 5e-4 : 3e-10
                @test all(isapprox(Array(out[i]),expected[i];rtol=tol) for i in 1:3)
                fill!(fields,T(NaN));field!(out...,gpu,device_points...)
                @test all(isfinite,Array(fields))
                @test_throws ArgumentError field!(expected...,gpu,device_points...)
            end
            @test_throws ArgumentError solve!(gpu,x,y,z,q,1)
        end
    end
    @testset "CUDA adaptive masks and reusable storage" begin
        for T in (Float32,Float64)
            rng=MersenneTwister(71);n=5001
            host=ntuple(_->rand(rng,T,n).*T(.002),3)
            q=fill(T(1e-12/n),n); host_mask=fill(true,n); host_mask[3:7:end].=false
            for a in (host...,q);a[.!host_mask].=T(NaN);end
            particles=CuArray.((host...,q));mask=CuArray(host_mask)
            gpu=FFTIGF((8,9,10);T,backend=CUDABackend())
            cpu=FFTIGF((8,9,10);T,fft_flags=FFTW.ESTIMATE)
            @test solve!(gpu,particles...,2;mask) === nothing
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
            workspace=gpu._workspace
            frozen=Array.(out)
            for a in particles;fill!(a,T(NaN));end
            evaluation_points=CuArray.(host)
            field!(out...,gpu,evaluation_points...;mask)
            @test Array.(out) == frozen
            fill!(mask,false)
            @test_throws ArgumentError solve!(gpu,particles...,1;mask)
            field!(out...,gpu,evaluation_points...;mask)
            @test all(a->all(iszero,Array(a)),out)
            empty=CUDA.zeros(T,0)
            @test_throws ArgumentError solve!(gpu,empty,empty,empty,empty,1)
            @test field!(empty,empty,empty,gpu,empty,empty,empty) === (empty,empty,empty)
            before=Array(gpu.rho)
            for (a,m) in ((empty,nothing),(particles[1],mask))
                @test_throws ArgumentError deposit!(gpu,a,a,a,a;mask=m)
                @test deposit!(gpu,a,a,a,a;mask=m,clear=false) === gpu
                @test Array(gpu.rho) == before
            end
            fixed_bounds=(gpu.min_bounds,gpu.max_bounds)
            fixed=FFTIGF(gpu.grid_size;T,backend=CUDABackend())
            for (a,m) in ((empty,nothing),(particles[1],mask))
                @test_throws ArgumentError solve!(fixed,a,a,a,a,1;mask=m,bounds=fixed_bounds)
            end
            zero=CUDA.zeros(T,1); point=CuArray(T[0.001])
            for (solver,bounds) in ((gpu,nothing),(fixed,fixed_bounds))
                solve!(solver,point,point,point,zero,1;bounds)
                @test all(iszero,Array(solver.efield))
            end
            particles=CuArray.((host...,q));mask=CuArray(host_mask)
            solve!(gpu,particles...,1;mask,bounds=((-0.01,-0.01,-0.01),(0.01,0.01,0.01)))
            @test gpu.min_bounds == T.((-0.01,-0.01,-0.01))
            solve!(gpu,particles...,1;mask)
            @test gpu.min_bounds == cpu.min_bounds && gpu.max_bounds == cpu.max_bounds
            @test gpu._workspace === workspace
            fitted_bounds=(gpu.min_bounds,gpu.max_bounds)
            fitted_delta=gpu.delta; fitted_field=Array(gpu.efield)
            solve!(gpu,particles...,1;mask,bounds=fitted_bounds)
            @test gpu.delta == fitted_delta
            @test isapprox(Array(gpu.efield),fitted_field;rtol=tol)
            @test_throws ArgumentError solve!(gpu,particles...,1;mask=host_mask)
            @test_throws DimensionMismatch solve!(gpu,particles...,1;mask=CUDA.ones(Bool,1))
            @test_throws ArgumentError solve!(gpu,particles...,Inf)
            for bad in (T(NaN),T(Inf),T(-Inf))
                dx=CuArray(T[0,bad]); zero=CUDA.zeros(T,2); charge=CUDA.ones(T,2)
                @test_throws ArgumentError solve!(gpu,dx,zero,zero,charge,1)
            end
        end
    end
end
