using SpaceCharge, Test, FFTW

function run_threaded_deposition_tests()
    @testset "Batched deposition from a task" begin
        for T in (Float32,Float64)
            sc = FFTIGF((5,5,5); T, fft_flags=FFTW.ESTIMATE)
            endpoints = T[0,1]
            solve!(sc,endpoints,endpoints,endpoints,zeros(T,2),1;bounds=((0,0,0),(1,1,1)))
            # Every particle is halfway between the same eight grid nodes.
            # Binary charges keep the expected cell sums exactly representable.
            n = 83_017
            x,y,z = (fill(T(v),n) for v in (.125,.375,.625))
            q = [isodd(i) ? T(2.0^-30) : -T(2.0^-31) for i in 1:n]
            expected = zeros(T,5,5,5)
            expected[1:2,2:3,3:4] .= sum(q)/8
            atol = 128eps(T)*sum(abs,q)
            fetch(Threads.@spawn deposit!(sc,x,y,z,q))
            @test sc.rho ≈ expected rtol=128eps(T) atol=atol
            deposit!(sc,x,y,z,q;clear=false)
            @test sc.rho ≈ 2expected rtol=256eps(T) atol=2atol
            deposit!(sc,x,y,z,-q)
            @test sc.rho ≈ -expected rtol=128eps(T) atol=atol
            mask = trues(n); mask[3:3:end] .= false
            expected[1:2,2:3,3:4] .= sum(q[mask])/8
            x[.!mask] .= T(NaN); q[.!mask] .= T(Inf)
            fetch(Threads.@spawn deposit!(sc,x,y,z,q;mask))
            @test sc.rho ≈ expected rtol=128eps(T) atol=atol
        end
    end
end
