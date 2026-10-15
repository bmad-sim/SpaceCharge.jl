using Test, SpaceCharge, FFTW

function run_mesh_tests()
    @testset "FFTIGF construction and layout" begin
        for T in (Float32, Float64)
            sc = FFTIGF((4,5,6); T, fft_flags=FFTW.ESTIMATE)
            @test eltype(sc.mesh.rho) == T
            @test size(sc.mesh.rho) == (4,5,6)
            @test !sc.prepared
            @test sc.mesh._workspace === nothing
            x = T[-1,1]; q = T[1e-12,-1e-12]; o = ntuple(_ -> similar(x), 3)
            @test_throws ArgumentError field!(o..., sc, x,x,x)
            @test_throws ArgumentError solve!(sc)
            @test_throws ArgumentError deposit!(sc,x,x,x,q)
            @test solve!(sc,x,x,x,q) === sc
            @test field!(o...,sc,x,x,x) === o
            @test (@inferred solve!(sc,x,x,x,q)) === sc
            @test (@inferred field!(o...,sc,x,x,x)) === o
            @test occursin("prepared=true", sprint(show,sc))
            @test_throws DimensionMismatch solve!(sc,x,x[1:1],x,q)
            @test_throws ArgumentError field!(o...,sc,x,x,x)
            @test_throws ArgumentError solve!(sc,reshape(x,2,1),x,x,q)
            @test_throws ArgumentError solve!(sc,Int[1,2],x,x,q)
            @test_throws ArgumentError solve!(sc,x,x,x,Float16.(q))
            @test_throws ArgumentError solve!(sc,x,x,x,q;active=UInt8[1,0])
            @test_throws DimensionMismatch solve!(sc,x,x,x,q;active=Bool[true])
            solve!(sc,x,x,x,q)
            @test_throws DimensionMismatch field!(o...,sc,x,x,x[1:1])
            @test_throws ArgumentError field!(Float16.(x),o[2],o[3],sc,x,x,x)
            for g in (0, -1, NaN, Inf, big(10)^1000)
                @test_throws ArgumentError solve!(sc,x,x,x,q;gamma=g)
                @test_throws ArgumentError field!(o...,sc,x,x,x)
            end
            # Failed validation never revives an older source.
            @test_throws ArgumentError solve!(sc)
        end
        @test_throws ArgumentError FFTIGF((1,4,4))
        @test_throws ArgumentError FFTIGF((4,4,4);T=Float16)
        @test_throws ArgumentError FFTIGF((4,4,4);bounds=((0,0,0),(0,1,1)))
        @test_throws ArgumentError FFTIGF((4,4,4);bounds=((0,0,0),(Inf,1,1)))
        @test_throws ArgumentError FFTIGF((4,4,4);T=Float32,bounds=((1e8,0,0),(1e8+1,1,1)))
        @test_throws ArgumentError FFTIGF((4,4,4);T=Float32,bounds=((1e8,0,0),(1e8+16,1,1)))
        @test_throws ArgumentError FFTIGF((4,4,4);at_cathode=true,bounds=((-1,-1,-1),(1,1,1)))
        @test_throws MethodError FFTIGF((4,4,4);boundary=:periodic)
        # Import/construction do not change FFTW's process-global thread count.
        threads = FFTW.get_num_threads()
        FFTIGF((4,4,4))
        @test FFTW.get_num_threads() == threads
    end
end
