using Test, SpaceCharge

function run_deposition_tests()
    @testset "CIC stages, faces, masks and failures" begin
        for T in (Float32,Float64)
            sc = FFTIGF((3,3,3);T,bounds=((0,0,0),(1,1,1)),fft_flags=FFTW.ESTIMATE)
            x = T[0,0.25,1,NaN]; q = T[1,-2,3,NaN]; mask=Bool[1,1,1,0]
            @test deposit!(sc,x,x,x,q;active=mask) === sc
            @test sum(sc.mesh.rho) ≈ 2
            @test sc.mesh.rho[1,1,1] ≈ 0.75
            @test sc.mesh.rho[2,2,2] ≈ -0.25
            @test sc.mesh.rho[3,3,3] == 3
            @test sc.mesh._workspace === nothing
            first = copy(sc.mesh.rho)
            deposit!(sc,x,x,x,q;active=mask,clear=false)
            @test sc.mesh.rho == 2first
            solve!(sc;gamma=2)
            deposit!(sc,x,x,x,q;active=mask)
            @test !sc.prepared
            @test sc.mesh.rho == first
            bad = T[1.01]
            @test_throws ErrorException deposit!(sc,bad,bad,bad,T[1])
            @test !sc.prepared && !sc.deposited
            @test_throws ArgumentError solve!(sc)
            @test_throws ArgumentError deposit!(sc,x,x,x,q;active=mask,clear=false)
            deposit!(sc,T[],T[],T[],T[])
            solve!(sc)
            @test all(iszero,sc.mesh.efield)
            @test_throws ErrorException solve!(sc,T[NaN],T[0],T[0],T[1])
        end
    end
end
