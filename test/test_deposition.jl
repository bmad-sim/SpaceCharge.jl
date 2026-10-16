using Test, SpaceCharge

function run_deposition_tests()
    @testset "CIC stages, faces and masks" begin
        for T in (Float32,Float64)
            sc = FFTIGF((3,3,3);T,fft_flags=FFTW.ESTIMATE)
            x = T[0,0.25,1,NaN]; q = T[1,-2,3,NaN]; mask=Bool[1,1,1,0]
            bounds=((0,0,0),(1,1,1))
            solve!(sc,x,x,x,q,1;mask=mask,bounds)
            workspace=sc._workspace
            @test deposit!(sc,x,x,x,q;mask=mask) === sc
            @test sum(sc.rho) ≈ 2
            @test sc.rho[1,1,1] ≈ 0.75
            @test sc.rho[2,2,2] ≈ -0.25
            @test sc.rho[3,3,3] == 3
            @test sc._workspace === workspace
            first = copy(sc.rho)
            deposit!(sc,x,x,x,q;mask=mask,clear=false)
            @test sc.rho == 2first
            solve!(sc,2)
            deposit!(sc,x,x,x,q;mask=mask)
            @test sc.rho == first
            # Failed replacement and empty additions preserve deposited charge.
            for (a,m) in ((T[],nothing),(T[NaN],Bool[0]))
                @test_throws ArgumentError solve!(sc,a,a,a,a,1;mask=m,bounds)
                @test_throws ArgumentError deposit!(sc,a,a,a,a;mask=m)
                @test sc.rho == first
                @test deposit!(sc,a,a,a,a;mask=m,clear=false) === sc
                @test sc.rho == first
            end
            solve!(sc,1)
            ref=FFTIGF((3,3,3);T,fft_flags=FFTW.ESTIMATE)
            solve!(ref,x,x,x,q,1;mask=mask,bounds)
            @test sc.efield == ref.efield
            # Active particles with zero charge remain valid.
            a=T[0]; deposit!(sc,a,a,a,a); solve!(sc,1)
            @test all(iszero,sc.efield)
        end
    end
end
