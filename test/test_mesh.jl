using Test, SpaceCharge, FFTW

function run_mesh_tests()
    @testset "Supported precision and input errors" begin
        for T in (Float32, Float64)
            sc = FFTIGF((4,5,6); T, fft_flags=FFTW.ESTIMATE)
            @test eltype(sc.rho) == T
            @test size(sc.rho) == (4,5,6)
        end
        @test_throws ArgumentError FFTIGF((1,4,4))
        @test_throws ArgumentError FFTIGF((4,4,4); T=Float16)
        @test_throws ArgumentError FFTIGF((4,4,4); backend=nothing)
        @test_throws ArgumentError FFTIGF((4,4,4); field_method=:unknown)
        sc = FFTIGF((4,4,4); fft_flags=FFTW.ESTIMATE)
        x = [-0.5,0.5]; q = [1e-12,-1e-12]; out = ntuple(_->similar(x),3)
        solve!(sc,x,x,x,q,1)
        for gamma in (0,NaN,Inf,"2",big(10)^1000)
            @test_throws ArgumentError solve!(sc,x,x,x,q,gamma)
            @test_throws ArgumentError solve!(sc,gamma)
        end
        for bounds in (((0,0),(1,1)), ((0,0,0),(0,1,1)),
                       ((1,0,0),(0,1,1)), ((0,0,0),(Inf,1,1)),
                       ((NaN,0,0),(1,1,1)),
                       ((-floatmax(Float64),0,0),(floatmax(Float64),1,1)))
            @test_throws ArgumentError solve!(sc,x,x,x,q,1;bounds)
        end
        sc32 = FFTIGF((4,4,4); T=Float32, fft_flags=FFTW.ESTIMATE)
        a = Float32[0]
        @test_throws ArgumentError solve!(sc32,a,a,a,a,1;bounds=((1e8,0,0),(1e8+1,1,1)))
        @test_throws ArgumentError solve!(sc32,a,a,a,a,1;bounds=((0,0,0),(nextfloat(0f0),1,1)))
        @test_throws DimensionMismatch solve!(sc,x,x[1:1],x,q,1)
        @test_throws DimensionMismatch deposit!(sc,x,x,x,q[1:1])
        @test_throws ArgumentError solve!(sc,reshape(x,2,1),x,x,q,1)
        @test_throws ArgumentError solve!(sc,x,x,x,Float16.(q),1)
        @test_throws ArgumentError solve!(sc,x,x,x,q,1;mask=UInt8[1,0])
        @test_throws DimensionMismatch deposit!(sc,x,x,x,q;mask=Bool[true])
        @test_throws DimensionMismatch field!(out...,sc,x,x,x[1:1])
        @test_throws ArgumentError field!(Float16.(x),out[2],out[3],sc,x,x,x)
        @test_throws DimensionMismatch field!(out...,sc,x,x,x;mask=Bool[true])
    end
end
