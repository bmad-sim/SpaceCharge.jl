using SpaceCharge, Test, KernelAbstractions

# An offset vector is rejected before any backend-specific inspection or indexing.
struct OffsetTestVector{T} <: AbstractVector{T}
    data::Vector{T}
end
Base.size(a::OffsetTestVector) = size(a.data)
Base.axes(a::OffsetTestVector) = (0:length(a.data)-1,)

function run_shared_api_tests()
    @testset "Shared solver API and array validation" begin
        @test solve! isa Function
        @test field! isa Function
        for T in (Float32,Float64)
            owner=zeros(T,3,3,3)
            coordinates=zeros(T,4,3)
            x,y,z=ntuple(i->view(coordinates,:,i),3)
            q=ones(T,4);mask=Bool[true,false,true,false]
            @test SpaceCharge._check_arrays(owner,(x,y,z,q),nothing) === nothing
            @test SpaceCharge._check_arrays(owner,(x,y,z,q),mask) === nothing
            @test SpaceCharge._storage(x) === coordinates
            @test SpaceCharge._active(nothing,1)
            @test !SpaceCharge._active(mask,2)
            @test_throws DimensionMismatch SpaceCharge._check_arrays(owner,(x,y[1:2],z,q),nothing)
            @test_throws ArgumentError SpaceCharge._check_arrays(owner,(coordinates,y,z,q),nothing)
            @test_throws ArgumentError SpaceCharge._check_arrays(owner,(x,y,z,ones(Int,4)),nothing)
            @test_throws ArgumentError SpaceCharge._check_arrays(owner,(x,y,z,q),ones(UInt8,4))
            @test_throws DimensionMismatch SpaceCharge._check_arrays(owner,(x,y,z,q),Bool[true])
            offset=OffsetTestVector(zeros(T,4))
            @test_throws ArgumentError SpaceCharge._check_arrays(owner,(offset,offset,offset,offset),nothing)
            @test SpaceCharge._gamma(T,1) === one(T)
            @test SpaceCharge._gamma(T,2.5) === T(2.5)
            for gamma in (0,-1,Inf,NaN,big(10)^1000)
                @test_throws ArgumentError SpaceCharge._gamma(T,gamma)
            end
        end
    end
end
