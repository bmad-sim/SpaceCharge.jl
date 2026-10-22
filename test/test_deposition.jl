using Test, SpaceCharge

function run_deposition_tests()
    @testset "CIC stages, faces and masks" begin
        for T in (Float32,Float64)
            sc = FFTIGF((3,3,3);T,fft_flags=FFTW.ESTIMATE)
            x = T[0,0.25,1,NaN]; q = T[1,-2,3,NaN]; mask=Bool[1,1,1,0]
            bounds=((0,0,0),(1,1,1))
            solve!(sc,x,x,x,q,1;mask=mask,bounds)
            @test deposit!(sc,x,x,x,q;mask=mask) === sc
            @test sum(sc.rho) ≈ 2
            @test sc.rho[1,1,1] ≈ 0.75
            @test sc.rho[2,2,2] ≈ -0.25
            @test sc.rho[3,3,3] == 3
            expected = zeros(T,3,3,3)
            expected[1:2,1:2,1:2] .= T(-.25)
            expected[1,1,1] += 1; expected[3,3,3] = 3
            deposit!(sc,x,x,x,q;mask=mask,clear=false)
            @test sc.rho == 2expected
            solve!(sc,2)
            deposit!(sc,x,x,x,q;mask=mask)
            @test sc.rho == expected
            # Empty additions have the same expected deposited density.
            for (a,m) in ((T[],nothing),(T[NaN],Bool[0]))
                @test_throws ArgumentError solve!(sc,a,a,a,a,1;mask=m,bounds)
                @test_throws ArgumentError deposit!(sc,a,a,a,a;mask=m)
                @test deposit!(sc,a,a,a,a;mask=m,clear=false) === sc
                @test sc.rho == expected
            end
            # Active particles with zero charge remain valid.
            a=T[0]; deposit!(sc,a,a,a,a); solve!(sc,1)
            @test all(iszero,sc.efield)
        end
    end
end
