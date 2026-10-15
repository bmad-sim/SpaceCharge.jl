using Test, SpaceCharge, Random

function run_solver_tests()
    @testset "FFTIGF source lifetime and physics" begin
        for T in (Float32,Float64)
            tol = T == Float32 ? 3e-5 : 2e-12
            n=(8,9,10); bounds=((-0.02,-0.03,-0.04),(0.02,0.03,0.04))
            sc=FFTIGF(n;T,bounds,fft_flags=FFTW.ESTIMATE)
            staged=FFTIGF(n;T,bounds,fft_flags=FFTW.ESTIMATE)
            x=T[-0.01,0.009]; y=T[0.003,-0.008]; z=T[-0.012,0.013]; q=T[1e-12,-1e-12]
            targets=(T[-0.005,0,0.006],T[0.001,0.002,-0.003],T[0,0.003,0.01])
            o=ntuple(_->zeros(T,3),3); p=ntuple(_->zeros(T,3),3)
            @test solve!(sc,x,y,z,q;gamma=2) === sc
            deposit!(staged,x,y,z,q); @test solve!(staged;gamma=2) === staged
            field!(o...,sc,targets...); field!(p...,staged,targets...)
            @test all(o[i]≈p[i] for i in 1:3)
            @test any(!iszero,o[1]) # signed net zero is not an empty source
            frozen=copy.(o); source=copy.((x,y,z,q))
            # Source column views, including non-unit particle stride, use the
            # same preparation path without packing an intermediate bunch.
            for stride in (1,2)
                positions=zeros(T,stride*length(x),3)
                rows=1:stride:size(positions,1)
                positions[rows,:] .= hcat(x,y,z)
                components=ntuple(i->view(positions,rows,i),3)
                solve!(staged,components...,q;gamma=2)
                field!(p...,staged,targets...)
                @test all(isapprox(p[i],frozen[i];rtol=tol) for i in 1:3)
            end
            x .*= 0; y .*=0; z .*=0; q .*=0
            field!(o...,sc,targets...)
            @test o == frozen
            x,y,z,q=source
            key=sc.mesh._workspace.free_cache.key; spectra=sc.mesh._workspace.free_cache.spectra
            solve!(sc,x,y,z,q;gamma=2)
            @test sc.mesh._workspace.free_cache.key == key
            @test sc.mesh._workspace.free_cache.spectra === spectra
            solve!(sc,x,y,z,-3q;gamma=2); field!(o...,sc,targets...)
            @test all(isapprox(o[i],-3frozen[i];rtol=tol) for i in 1:3)
            solve!(sc) # gamma defaults to one, even in staged reuse
            @test sc.mesh.gamma == 1
            @test sc.mesh._workspace.free_cache.key != key
            solve!(sc,x,y,z,q); field!(o...,sc,targets...)
            @test sc.mesh.gamma == 1
            # Translating source, domain and targets preserves free-space E.
            shift=T.((0.003,0.007,0.011))
            translated=FFTIGF(n;T,bounds=(T.(bounds[1]).+shift,T.(bounds[2]).+shift),fft_flags=FFTW.ESTIMATE)
            solve!(translated,x.+shift[1],y.+shift[2],z.+shift[3],q)
            field!(p...,translated,(targets[i].+shift[i] for i in 1:3)...)
            @test all(isapprox(o[i],p[i];rtol=tol) for i in 1:3)
            # Stretch lab z exactly once; Eperp=gamma*Erest, Ez=Ezrest.
            g=T(3)
            rest=FFTIGF(n;T,bounds=((bounds[1][1:2]...,g*bounds[1][3]),(bounds[2][1:2]...,g*bounds[2][3])),fft_flags=FFTW.ESTIMATE)
            solve!(sc,x,y,z,q;gamma=g); field!(o...,sc,targets...)
            solve!(rest,x,y,g*z,q); field!(p...,rest,targets[1],targets[2],g*targets[3])
            @test o[1] ≈ g*p[1] rtol=tol
            @test o[2] ≈ g*p[2] rtol=tol
            @test o[3] ≈ p[3] rtol=tol
        end
    end
    @testset "Adaptive masks, empty sources and degeneracy" begin
        for T in (Float32,Float64)
            sc=FFTIGF((8,8,8);T,fft_flags=FFTW.ESTIMATE)
            x=T[-0.001,0.001,NaN]; q=T[1e-12,-1e-12,NaN]; mask=Bool[1,1,0]
            solve!(sc,x,x,x,q;active=mask)
            ref=FFTIGF((8,8,8);T,fft_flags=FFTW.ESTIMATE)
            solve!(ref,x[1:2],x[1:2],x[1:2],q[1:2])
            @test sc.mesh.efield == ref.mesh.efield
            @test_throws ArgumentError solve!(sc,x,x,x,q)
            @test !sc.prepared
            for source in ((T[],T[],nothing),(T[NaN],T[NaN],Bool[0]),(T[0],T[0],nothing))
                a,b,m=source
                solve!(sc,a,a,a,b;active=m)
                o=ntuple(_->fill(T(NaN),1),3)
                field!(o...,sc,T[0],T[0],T[0])
                @test all(v->iszero(v[1]),o)
            end
            for value in (T(0),T(1e6))
                a=T[value]
                solve!(sc,a,a,a,T[1e-12])
                @test all(isfinite,sc.mesh.efield)
                @test all(sc.mesh.min_bounds .< sc.mesh.max_bounds)
                o=ntuple(_->similar(a),3); field!(o...,sc,a,a,a)
                @test maximum(abs,first.(o)) <= 100eps(T)*maximum(abs,sc.mesh.efield)
            end
        end
    end
    @testset "Coulomb normalization and cathode" begin
        for T in (Float32,Float64)
            sc=FFTIGF((33,33,33);T,bounds=((-0.016,-0.016,-0.016),(0.016,0.016,0.016)),fft_flags=FFTW.ESTIMATE)
            a=T[0]; solve!(sc,a,a,a,T[1e-12])
            o=ntuple(_->similar(a),3); field!(o...,sc,T[0.012],a,a)
            expected=SpaceCharge.FPEI*1e-12/0.012^2
            @test o[1][1] ≈ expected rtol=0.005
            @test abs(o[2][1])+abs(o[3][1]) < 1e-4expected
        end
        bounds=((-0.02,-0.03,0.),(0.02,0.03,0.04))
        free=FFTIGF((7,8,9);bounds,fft_flags=FFTW.ESTIMATE)
        cathode=FFTIGF((7,8,9);bounds,at_cathode=true,fft_flags=FFTW.ESTIMATE)
        x=[-0.006,0.013];y=[0.007,-0.012];z=[0.015,0.028];q=[1e-12,2e-12]
        solve!(free,x,y,z,q;gamma=2);solve!(cathode,x,y,z,q;gamma=2)
        source=copy(cathode.mesh.rho)
        for _ in 1:3
            @test solve!(cathode;gamma=2) === cathode
            e=cathode.mesh.efield; scale=maximum(abs,e)
            @test maximum(abs,e[:,:,1,1:2]) < 1e-10scale
            @test e[:,:,1,3] ≈ 2free.mesh.efield[:,:,1,3] rtol=1e-10
            @test cathode.mesh.rho == source
        end
        adaptive=FFTIGF((7,8,9);at_cathode=true,fft_flags=FFTW.ESTIMATE)
        @test_throws ArgumentError solve!(adaptive,x,y,-z,q)
    end
end
