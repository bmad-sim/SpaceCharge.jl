using Test, SpaceCharge, Random
using AtomicAndPhysicalConstants: EPS_0, E_CHARGE

function test_float32_adaptive_endpoints(backend)
    @testset "Float32 adaptive endpoints on $backend" begin
        sc=FFTIGF((64,64,64);T=Float32,backend,fft_flags=FFTW.ESTIMATE)
        x=KernelAbstractions.zeros(backend,Float32,2)
        q=similar(x);fill!(q,1f-12)
        out=ntuple(_->similar(x),3)
        for points in (Float32[-.001,.001],Float32[1f6,nextfloat(1f6)],
                       Float32[-1f6,nextfloat(-1f6)])
            copyto!(x,points)
            solve!(sc,x,x,x,q,1)
            # Ordinary padding may round away; both grid faces remain valid.
            @test all(sc.min_bounds .<= points[1])
            @test all(sc.max_bounds .>= points[2])
            @test isapprox(sum(sc.rho),2f-12;rtol=2f-6)
            field!(out...,sc,x,x,x)
            @test all(a->all(isfinite,Array(a)),out)
            fill!(sc.efield,2f0)
            field!(out...,sc,x,x,x)
            @test all(a->all(v->isapprox(v,2f0;rtol=2f-6),Array(a)),out)
        end
    end
end

function test_cathode_geometry(backend)
    @testset "Cathode fields on supplied and fitted domains on $backend" begin
        for T in (Float32,Float64)
            n=(5,6,7)
            cathode=FFTIGF(n;T,backend,at_cathode=true,fft_flags=FFTW.ESTIMATE)
            x,y,z,q=ntuple(_->KernelAbstractions.zeros(backend,T,2),4)
            copyto!(x,T[-.007,.011]);copyto!(y,T[.004,-.009]);copyto!(q,T[1e-12,-2e-12])
            for (points,bounds) in (
                ((-.017,-.003),((-.02,-.03,-.04),(.02,.03,-.002))),
                ((-.017,.003),((-.02,-.03,-.023),(.02,.03,.019))),
                ((-.017,-.003),nothing),
                ((0.,.012),nothing),
            )
                copyto!(z,T[points...])
                solve!(cathode,x,y,z,q,2;bounds)
                if bounds === nothing
                    @test cathode.min_bounds[3] <= min(points...) <= max(points...) <= cathode.max_bounds[3]
                else
                    @test (cathode.min_bounds,cathode.max_bounds) == map(b->T.(b),bounds)
                end
                snapshot=(rho=Array(cathode.rho),efield=Array(cathode.efield),
                          delta=cathode.delta,gamma=cathode.gamma,
                          min_bounds=cathode.min_bounds,max_bounds=cathode.max_bounds)
                test_field_against_reference(snapshot;at_cathode=true)
            end
            # Ordinary geometry validation still applies in cathode mode.
            for bounds in (((0,0,0),(1,1,0)),((0,0,0),(1,1,Inf)))
                @test_throws ArgumentError solve!(cathode,x,y,z,q,1;bounds)
            end
        end
    end
end

function run_solver_tests()
    @testset "FFTIGF prepared field lifetime and physics" begin
        for T in (Float32,Float64)
            tol = T == Float32 ? 3e-5 : 2e-12
            n=(8,9,10); bounds=((-0.02,-0.03,-0.04),(0.02,0.03,0.04))
            sc=FFTIGF(n;T,fft_flags=FFTW.ESTIMATE)
            staged=FFTIGF(n;T,fft_flags=FFTW.ESTIMATE)
            x=T[-0.01,0.009]; y=T[0.003,-0.008]; z=T[-0.012,0.013]; q=T[1e-12,-1e-12]
            evaluation_points=(T[-0.005,0,0.006],T[0.001,0.002,-0.003],T[0,0.003,0.01])
            o=ntuple(_->zeros(T,3),3); p=ntuple(_->zeros(T,3),3)
            solve!(sc,x,y,z,q,2;bounds)
            solve!(staged,x,y,z,q,1;bounds)
            deposit!(staged,x,y,z,q); solve!(staged,2)
            field!(o...,sc,evaluation_points...); field!(p...,staged,evaluation_points...)
            @test all(o[i]≈p[i] for i in 1:3)
            frozen=copy.(o); particles=copy.((x,y,z,q))
            # Particle column views, including non-unit particle stride, use the
            # same preparation path without packing an intermediate bunch.
            for stride in (1,2)
                positions=zeros(T,stride*length(x),3)
                rows=1:stride:size(positions,1)
                positions[rows,:] .= hcat(x,y,z)
                components=ntuple(i->view(positions,rows,i),3)
                solve!(staged,components...,q,2;bounds)
                field!(p...,staged,evaluation_points...)
                @test all(isapprox(p[i],frozen[i];rtol=tol) for i in 1:3)
            end
            x .*= 0; y .*=0; z .*=0; q .*=0
            field!(o...,sc,evaluation_points...)
            @test o == frozen
            x,y,z,q=particles
            solve!(sc,x,y,z,-3q,2;bounds); field!(o...,sc,evaluation_points...)
            @test all(isapprox(o[i],-3frozen[i];rtol=tol) for i in 1:3)
            solve!(sc,x,y,z,q,1;bounds); field!(o...,sc,evaluation_points...)
            # Translating particles, domain and evaluation points preserves free-space E.
            shift=T.((0.003,0.007,0.011))
            translated=FFTIGF(n;T,fft_flags=FFTW.ESTIMATE)
            solve!(translated,x.+shift[1],y.+shift[2],z.+shift[3],q,1;bounds=(T.(bounds[1]).+shift,T.(bounds[2]).+shift))
            field!(p...,translated,(evaluation_points[i].+shift[i] for i in 1:3)...)
            @test all(isapprox(o[i],p[i];rtol=tol) for i in 1:3)
            # Stretch lab z exactly once; Eperp=gamma*Erest, Ez=Ezrest.
            g=T(3)
            rest=FFTIGF(n;T,fft_flags=FFTW.ESTIMATE)
            solve!(sc,x,y,z,q,g;bounds); field!(o...,sc,evaluation_points...)
            solve!(rest,x,y,g*z,q,1;bounds=((bounds[1][1:2]...,g*bounds[1][3]),(bounds[2][1:2]...,g*bounds[2][3]))); field!(p...,rest,evaluation_points[1],evaluation_points[2],g*evaluation_points[3])
            @test o[1] ≈ g*p[1] rtol=tol
            @test o[2] ≈ g*p[2] rtol=tol
            @test o[3] ≈ p[3] rtol=tol
        end
    end
    @testset "Bounds are selected per particle solve" begin
        for T in (Float32,Float64)
            sc=FFTIGF((8,8,8);T,fft_flags=FFTW.ESTIMATE)
            ref=FFTIGF((8,8,8);T,fft_flags=FFTW.ESTIMATE)
            x=T[-0.001,0.001];q=T[1e-12,-1e-12]
            bounds=(T.((-0.01,-0.01,-0.01)),T.((0.01,0.01,0.01)))
            solve!(sc,x,x,x,q,1;bounds)
            @test (sc.min_bounds,sc.max_bounds) == bounds
            # Explicit geometry remains available to the existing staged API.
            deposit!(sc,x,x,x,q;clear=false);solve!(sc,2)
            @test (sc.min_bounds,sc.max_bounds) == bounds
            # Omitting bounds fits again; the earlier override is not persistent.
            solve!(sc,x,x,x,q,1)
            solve!(ref,x,x,x,q,1)
            @test (sc.min_bounds,sc.max_bounds,sc.delta) == (ref.min_bounds,ref.max_bounds,ref.delta)
            @test sc.efield == ref.efield
            # Reusing fitted bounds explicitly gives the same grid and field.
            solve!(ref,x,x,x,q,1;bounds=(sc.min_bounds,sc.max_bounds))
            @test sc.delta == ref.delta
            @test sc.efield == ref.efield
            shifted=x.+T(.1)
            solve!(sc,shifted,shifted,shifted,q,1)
            @test all(sc.min_bounds .> T(.09))
        end
    end
    @testset "Adaptive masks, empty particle batches and degeneracy" begin
        for T in (Float32,Float64)
            sc=FFTIGF((8,8,8);T,fft_flags=FFTW.ESTIMATE)
            x=T[-0.001,0.001,NaN]; q=T[1e-12,-1e-12,NaN]; mask=Bool[1,1,0]
            solve!(sc,x,x,x,q,1;mask=mask)
            ref=FFTIGF((8,8,8);T,fft_flags=FFTW.ESTIMATE)
            solve!(ref,x[1:2],x[1:2],x[1:2],q[1:2],1)
            @test sc.efield == ref.efield
            for bad in (T(NaN),T(Inf),T(-Inf))
                @test_throws ArgumentError solve!(sc,T[bad],T[0],T[0],T[1],1)
                @test_throws ArgumentError solve!(sc,T[0,bad],T[0,0],T[0,0],T[1,1],1)
            end
            for particle_batch in ((T[],T[],nothing),(T[NaN],T[NaN],Bool[0]))
                a,b,m=particle_batch
                @test_throws ArgumentError solve!(sc,a,a,a,b,1;mask=m)
            end
            # Active zero-charge particles still establish valid geometry.
            a=T[0]; solve!(sc,a,a,a,a,1)
            o=ntuple(_->fill(T(NaN),1),3)
            field!(o...,sc,a,a,a)
            @test all(v->iszero(v[1]),o)
            for value in (T(0),T(1e6))
                a=T[value]
                solve!(sc,a,a,a,T[1e-12],1)
                @test all(isfinite,sc.efield)
                @test all(sc.min_bounds .< sc.max_bounds)
                o=ntuple(_->similar(a),3); field!(o...,sc,a,a,a)
                @test maximum(abs,first.(o)) <= 100eps(T)*maximum(abs,sc.efield)
            end
        end
    end
    test_float32_adaptive_endpoints(CPU())
    test_cathode_geometry(CPU())
    @testset "Coulomb normalization and cathode" begin
        for T in (Float32,Float64)
            sc=FFTIGF((33,33,33);T,fft_flags=FFTW.ESTIMATE)
            a=T[0]; solve!(sc,a,a,a,T[1e-12],1;bounds=((-0.016,-0.016,-0.016),(0.016,0.016,0.016)))
            o=ntuple(_->similar(a),3); field!(o...,sc,T[0.012],a,a)
            expected=inv(4pi * EPS_0 * E_CHARGE)*1e-12/0.012^2
            @test o[1][1] ≈ expected rtol=0.005
            @test abs(o[2][1])+abs(o[3][1]) < 1e-4expected
        end
        bounds=((-0.02,-0.03,0.),(0.02,0.03,0.04))
        free=FFTIGF((7,8,9);fft_flags=FFTW.ESTIMATE)
        cathode=FFTIGF((7,8,9);at_cathode=true,fft_flags=FFTW.ESTIMATE)
        x=[-0.006,0.013];y=[0.007,-0.012];z=[0.015,0.028];q=[1e-12,2e-12]
        solve!(free,x,y,z,q,2;bounds);solve!(cathode,x,y,z,q,2;bounds)
        e=cathode.efield; scale=maximum(abs,e)
        @test maximum(abs,e[:,:,1,1:2]) < 1e-10scale
        @test e[:,:,1,3] ≈ 2free.efield[:,:,1,3] rtol=1e-10
    end
end
