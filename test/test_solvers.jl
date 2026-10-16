using Test, SpaceCharge, Random

# Reading an inactive particle is an error even when its array index is valid.
struct ActiveOnlyVector{T} <: AbstractVector{T}
    data::Vector{T}
    active::Vector{Bool}
end
Base.size(a::ActiveOnlyVector) = size(a.data)
Base.IndexStyle(::Type{<:ActiveOnlyVector}) = IndexLinear()
function Base.getindex(a::ActiveOnlyVector, i::Int)
    a.active[i] || error("inactive coordinate was read")
    a.data[i]
end

struct CountedMask <: AbstractVector{Bool}
    data::Vector{Bool}
    any_calls::Base.RefValue{Int}
end
Base.size(a::CountedMask) = size(a.data)
Base.IndexStyle(::Type{CountedMask}) = IndexLinear()
KernelAbstractions.get_backend(::CountedMask) = CPU()
Base.getindex(a::CountedMask, i::Int) = a.data[i]
Base.any(a::CountedMask) = (a.any_calls[] += 1; any(a.data))

function test_float32_adaptive_endpoints(backend)
    @testset "Float32 adaptive endpoints on $backend" begin
        sc=FFTIGF((64,64,64);T=Float32,backend,fft_flags=FFTW.ESTIMATE)
        x=KernelAbstractions.zeros(backend,Float32,2)
        q=similar(x);fill!(q,1f-12)
        out=ntuple(_->similar(x),3)
        for points in (Float32[-.001,.001],Float32[1f6,nextfloat(1f6)],
                       Float32[-1f6,nextfloat(-1f6)])
            copyto!(x,points)
            solve!(sc,x,x,x,q)
            # Ordinary padding may round away; both grid faces remain valid.
            @test sc.min_bounds == (points[1],points[1],points[1])
            @test sc.max_bounds == (points[2],points[2],points[2])
            @test isapprox(sum(sc.rho),2f-12;rtol=2f-6)
            field!(out...,sc,x,x,x)
            @test all(a->all(isfinite,Array(a)),out)
            fill!(sc.efield,2f0)
            field!(out...,sc,x,x,x)
            @test all(a->all(v->isapprox(v,2f0;rtol=2f-6),Array(a)),out)
        end
        # Do not expand otherwise invalid spacing to make its inverse finite.
        copyto!(x,Float32[1f-32,nextfloat(1f-32,251)])
        @test_throws ArgumentError solve!(sc,x,x,x,q)
    end
end

function test_cathode_geometry(backend)
    @testset "Cathode preserves supplied and fitted geometry on $backend" begin
        for T in (Float32,Float64)
            n=(5,6,7)
            free=FFTIGF(n;T,backend,fft_flags=FFTW.ESTIMATE)
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
                solve!(free,x,y,z,q;gamma=2,bounds)
                @test solve!(cathode,x,y,z,q;gamma=2,bounds) === cathode
                @test (cathode.min_bounds,cathode.max_bounds,cathode.delta) ==
                      (free.min_bounds,free.max_bounds,free.delta)
                if bounds === nothing
                    # Padding around z=0 must remain negative, just as in free space.
                    @test cathode.min_bounds[3] < min(points...)
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
                @test_throws ArgumentError solve!(cathode,x,y,z,q;bounds)
            end
        end
    end
end

function run_solver_tests()
    @testset "Particle presence checked once before bounds selection" begin
        sc=FFTIGF((4,4,4);fft_flags=FFTW.ESTIMATE)
        x=[-.001,NaN,.001];q=[1e-12,NaN,-1e-12]
        active=CountedMask(Bool[1,0,1],Ref(0))
        solve!(sc,x,x,x,q;active)
        @test active.any_calls[] == 1
        bounds=(sc.min_bounds,sc.max_bounds)
        solve!(sc,x,x,x,q;active,bounds)
        @test active.any_calls[] == 2
        deposit!(sc,x,x,x,q;active)
        @test active.any_calls[] == 3
        deposit!(sc,x,x,x,q;active,clear=false)
        @test active.any_calls[] == 3
        fill!(active.data,false)
        @test_throws ArgumentError solve!(sc,x,x,x,q;active)
        @test active.any_calls[] == 4
        @test_throws ArgumentError solve!(sc,x,x,x,q;active,bounds)
        @test active.any_calls[] == 5
    end
    @testset "Gamma validation precedes solver mutation" begin
        for T in (Float32,Float64)
            sc=FFTIGF((4,4,4);T,fft_flags=FFTW.ESTIMATE)
            x=T[-.001,.001];q=T[1e-12,-1e-12]
            solve!(sc,x,x,x,q;gamma=2)
            geometry=(sc.min_bounds,sc.max_bounds,sc.delta,sc.gamma)
            rho,field=copy(sc.rho),copy(sc.efield)
            for gamma in (NaN,Inf,-Inf,.5,"invalid",big"1e1000")
                @test_throws ArgumentError solve!(sc,x.+T(1),x,x,q;gamma)
                @test_throws ArgumentError solve!(sc;gamma)
                @test (sc.min_bounds,sc.max_bounds,sc.delta,sc.gamma) == geometry
                @test sc.rho == rho && sc.efield == field
            end
        end
    end
    @testset "FFTIGF prepared field lifetime and physics" begin
        for T in (Float32,Float64)
            tol = T == Float32 ? 3e-5 : 2e-12
            n=(8,9,10); bounds=((-0.02,-0.03,-0.04),(0.02,0.03,0.04))
            sc=FFTIGF(n;T,fft_flags=FFTW.ESTIMATE)
            staged=FFTIGF(n;T,fft_flags=FFTW.ESTIMATE)
            x=T[-0.01,0.009]; y=T[0.003,-0.008]; z=T[-0.012,0.013]; q=T[1e-12,-1e-12]
            evaluation_points=(T[-0.005,0,0.006],T[0.001,0.002,-0.003],T[0,0.003,0.01])
            o=ntuple(_->zeros(T,3),3); p=ntuple(_->zeros(T,3),3)
            @test solve!(sc,x,y,z,q;gamma=2,bounds) === sc
            solve!(staged,x,y,z,q;bounds)
            deposit!(staged,x,y,z,q); @test solve!(staged;gamma=2) === staged
            field!(o...,sc,evaluation_points...); field!(p...,staged,evaluation_points...)
            @test all(o[i]≈p[i] for i in 1:3)
            @test any(!iszero,o[1]) # signed net zero is not an empty particle batch
            frozen=copy.(o); particles=copy.((x,y,z,q))
            # Particle column views, including non-unit particle stride, use the
            # same preparation path without packing an intermediate bunch.
            for stride in (1,2)
                positions=zeros(T,stride*length(x),3)
                rows=1:stride:size(positions,1)
                positions[rows,:] .= hcat(x,y,z)
                components=ntuple(i->view(positions,rows,i),3)
                solve!(staged,components...,q;gamma=2,bounds)
                field!(p...,staged,evaluation_points...)
                @test all(isapprox(p[i],frozen[i];rtol=tol) for i in 1:3)
            end
            x .*= 0; y .*=0; z .*=0; q .*=0
            field!(o...,sc,evaluation_points...)
            @test o == frozen
            x,y,z,q=particles
            key=sc._workspace.free_cache.key; spectra=sc._workspace.free_cache.spectra
            solve!(sc,x,y,z,q;gamma=2,bounds)
            @test sc._workspace.free_cache.key == key
            @test sc._workspace.free_cache.spectra === spectra
            solve!(sc,x,y,z,-3q;gamma=2,bounds); field!(o...,sc,evaluation_points...)
            @test all(isapprox(o[i],-3frozen[i];rtol=tol) for i in 1:3)
            solve!(sc) # gamma defaults to one, even in staged reuse
            @test sc.gamma == 1
            @test sc._workspace.free_cache.key != key
            solve!(sc,x,y,z,q;bounds); field!(o...,sc,evaluation_points...)
            @test sc.gamma == 1
            # Translating particles, domain and evaluation points preserves free-space E.
            shift=T.((0.003,0.007,0.011))
            translated=FFTIGF(n;T,fft_flags=FFTW.ESTIMATE)
            solve!(translated,x.+shift[1],y.+shift[2],z.+shift[3],q;bounds=(T.(bounds[1]).+shift,T.(bounds[2]).+shift))
            field!(p...,translated,(evaluation_points[i].+shift[i] for i in 1:3)...)
            @test all(isapprox(o[i],p[i];rtol=tol) for i in 1:3)
            # Stretch lab z exactly once; Eperp=gamma*Erest, Ez=Ezrest.
            g=T(3)
            rest=FFTIGF(n;T,fft_flags=FFTW.ESTIMATE)
            solve!(sc,x,y,z,q;gamma=g,bounds); field!(o...,sc,evaluation_points...)
            solve!(rest,x,y,g*z,q;bounds=((bounds[1][1:2]...,g*bounds[1][3]),(bounds[2][1:2]...,g*bounds[2][3]))); field!(p...,rest,evaluation_points[1],evaluation_points[2],g*evaluation_points[3])
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
            solve!(sc,x,x,x,q;bounds)
            @test (sc.min_bounds,sc.max_bounds) == bounds
            workspace=sc._workspace;spectra=workspace.free_cache.spectra
            key=workspace.free_cache.key
            # Explicit geometry remains available to the existing staged API.
            deposit!(sc,x,x,x,q;clear=false);solve!(sc;gamma=2)
            @test (sc.min_bounds,sc.max_bounds) == bounds
            # Omitting bounds fits again; the earlier override is not persistent.
            solve!(sc,x,x,x,q)
            solve!(ref,x,x,x,q)
            @test (sc.min_bounds,sc.max_bounds,sc.delta) == (ref.min_bounds,ref.max_bounds,ref.delta)
            @test sc.efield == ref.efield
            # Reusing fitted bounds explicitly gives the same grid and field.
            solve!(ref,x,x,x,q;bounds=(sc.min_bounds,sc.max_bounds))
            @test sc.delta == ref.delta
            @test sc.efield == ref.efield
            @test sc._workspace === workspace && workspace.free_cache.spectra === spectra
            @test workspace.free_cache.key != key
            shifted=x.+T(.1)
            solve!(sc,shifted,shifted,shifted,q)
            @test all(sc.min_bounds .> T(.09))
        end
    end
    @testset "Particle bounds mask excludes coordinate reads" begin
        sc=FFTIGF((4,4,4);fft_flags=FFTW.ESTIMATE)
        mask=Bool[0,1,0,1]
        x=ActiveOnlyVector([NaN,-2.,NaN,3.],mask)
        ext=SpaceCharge._particle_bounds(x,x,x,mask)
        @test ext.lo == (-2.,-2.,-2.) && ext.hi == (3.,3.,3.)
        mask .= false
        @test_throws ArgumentError solve!(sc,x,x,x,ones(4);active=mask)
    end
    @testset "Adaptive masks, empty particle batches and degeneracy" begin
        for T in (Float32,Float64)
            sc=FFTIGF((8,8,8);T,fft_flags=FFTW.ESTIMATE)
            x=T[-0.001,0.001,NaN]; q=T[1e-12,-1e-12,NaN]; mask=Bool[1,1,0]
            solve!(sc,x,x,x,q;active=mask)
            ref=FFTIGF((8,8,8);T,fft_flags=FFTW.ESTIMATE)
            solve!(ref,x[1:2],x[1:2],x[1:2],q[1:2])
            @test sc.efield == ref.efield
            for bad in (T(NaN),T(Inf),T(-Inf))
                @test_throws ArgumentError solve!(sc,T[bad],T[0],T[0],T[1])
                @test_throws ArgumentError solve!(sc,T[0,bad],T[0,0],T[0,0],T[1,1])
            end
            for particle_batch in ((T[],T[],nothing),(T[NaN],T[NaN],Bool[0]))
                a,b,m=particle_batch
                @test_throws ArgumentError solve!(sc,a,a,a,b;active=m)
            end
            # Active zero-charge particles still establish valid geometry.
            a=T[0]; solve!(sc,a,a,a,a)
            o=ntuple(_->fill(T(NaN),1),3)
            field!(o...,sc,a,a,a)
            @test all(v->iszero(v[1]),o)
            for value in (T(0),T(1e6))
                a=T[value]
                solve!(sc,a,a,a,T[1e-12])
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
            a=T[0]; solve!(sc,a,a,a,T[1e-12];bounds=((-0.016,-0.016,-0.016),(0.016,0.016,0.016)))
            o=ntuple(_->similar(a),3); field!(o...,sc,T[0.012],a,a)
            expected=SpaceCharge.FPEI*1e-12/0.012^2
            @test o[1][1] ≈ expected rtol=0.005
            @test abs(o[2][1])+abs(o[3][1]) < 1e-4expected
        end
        bounds=((-0.02,-0.03,0.),(0.02,0.03,0.04))
        free=FFTIGF((7,8,9);fft_flags=FFTW.ESTIMATE)
        cathode=FFTIGF((7,8,9);at_cathode=true,fft_flags=FFTW.ESTIMATE)
        x=[-0.006,0.013];y=[0.007,-0.012];z=[0.015,0.028];q=[1e-12,2e-12]
        solve!(free,x,y,z,q;gamma=2,bounds);solve!(cathode,x,y,z,q;gamma=2,bounds)
        deposited_charge=copy(cathode.rho)
        for _ in 1:3
            @test solve!(cathode;gamma=2) === cathode
            e=cathode.efield; scale=maximum(abs,e)
            @test maximum(abs,e[:,:,1,1:2]) < 1e-10scale
            @test e[:,:,1,3] ≈ 2free.efield[:,:,1,3] rtol=1e-10
            @test cathode.rho == deposited_charge
        end
    end
end
