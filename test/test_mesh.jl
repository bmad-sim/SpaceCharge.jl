using Test, SpaceCharge, FFTW

function run_mesh_tests()
    @testset "FFTIGF construction and layout" begin
        for T in (Float32, Float64)
            sc = FFTIGF((4,5,6); T, fft_flags=FFTW.ESTIMATE)
            @test eltype(sc.rho) == T
            @test size(sc.rho) == (4,5,6)
            @test sc._workspace === nothing
            x = T[-1,1]; q = T[1e-12,-1e-12]; o = ntuple(_ -> similar(x), 3)
            @test_throws MethodError solve!(sc,x,x,x,q)
            @test_throws MethodError solve!(sc)
            @test_throws MethodError solve!(sc,x,x,x,q;gamma=1)
            @test_throws MethodError solve!(sc;gamma=1)
            @test sc._workspace === nothing
            @test solve!(sc,x,x,x,q,1) === nothing
            @test field!(o...,sc,x,x,x) === o
            @test (@inferred solve!(sc,x,x,x,q,1)) === nothing
            @test (@inferred solve!(sc,1)) === nothing
            @test (@inferred field!(o...,sc,x,x,x)) === o
        end
        # Import/construction do not change FFTW's process-global thread count.
        threads = FFTW.get_num_threads()
        FFTIGF((4,4,4))
        @test FFTW.get_num_threads() == threads
    end
    @testset "Basic input checks" begin
        @test_throws ArgumentError FFTIGF((1,4,4))
        @test_throws ArgumentError FFTIGF((4,4,4);T=Float16)
        @test_throws ArgumentError FFTIGF((4,4,4);backend=nothing)
        sc=FFTIGF((4,4,4);fft_flags=FFTW.ESTIMATE); a=[0.]
        for bounds in (((0,0),(1,1)), ((0,0,0),(0,1,1)),
                       ((1,0,0),(0,1,1)), ((0,0,0),(Inf,1,1)),
                       ((NaN,0,0),(1,1,1)), ((-floatmax(Float64),0,0),(floatmax(Float64),1,1)))
            @test_throws ArgumentError solve!(sc,a,a,a,a,1;bounds)
        end
        @test_throws ArgumentError solve!(FFTIGF((4,4,4);T=Float32),Float32[0],Float32[0],Float32[0],Float32[0],1;bounds=((1e8,0,0),(1e8+1,1,1)))
        @test_throws ArgumentError solve!(FFTIGF((4,4,4);T=Float32),Float32[0],Float32[0],Float32[0],Float32[0],1;bounds=((0,0,0),(nextfloat(0f0),1,1)))
        for T in (Float32,Float64)
            sc=FFTIGF((4,4,4);T,fft_flags=FFTW.ESTIMATE)
            x=T[-0.5,0.5]; q=T[1e-12,-1e-12]; out=ntuple(_->similar(x),3)
            solve!(sc,x,x,x,q,1)
            for gamma in (0,-1,NaN,Inf,"2",big(10)^1000)
                @test_throws ArgumentError solve!(sc,x,x,x,q,gamma)
                @test_throws ArgumentError solve!(sc,gamma)
            end
            @test_throws DimensionMismatch solve!(sc,x,x[1:1],x,q,1)
            @test_throws DimensionMismatch deposit!(sc,x,x,x,q[1:1])
            @test_throws ArgumentError solve!(sc,reshape(x,2,1),x,x,q,1)
            @test_throws ArgumentError solve!(sc,x,x,x,Float16.(q),1)
            @test_throws ArgumentError solve!(sc,x,x,x,q,1;mask=UInt8[1,0])
            @test_throws DimensionMismatch deposit!(sc,x,x,x,q;mask=Bool[true])
            @test_throws DimensionMismatch field!(out...,sc,x,x,x[1:1])
            @test_throws ArgumentError field!(Float16.(x),out[2],out[3],sc,x,x,x)
            @test_throws DimensionMismatch field!(out...,sc,x,x,x;mask=Bool[true])
            # Packed masks and ordinary mask views need no storage-unwrapping layer.
            for mask in (trues(2),view(trues(4),1:2:4))
                @test solve!(sc,x,x,x,q,1;mask=mask) === nothing
                @test field!(out...,sc,x,x,x;mask=mask) === out
            end
        end
    end
end
