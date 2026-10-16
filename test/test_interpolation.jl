using Test, SpaceCharge

function run_interpolation_tests()
    @testset "Gather domain, views and read-only evaluation" begin
        for T in (Float32,Float64)
            sc = FFTIGF((5,6,7);T,fft_flags=FFTW.ESTIMATE)
            SpaceCharge._set_grid_bounds!(sc,((-2,-3,-4),(2,3,4)))
            empty = T[]
            mesh=sc
            # Internal manufactured grid tests the gather independently of FFTs.
            linear(x,y,z)=(2x-3y+z+1,-x+4y+2z-2,3x+y-2z+3)
            for k in 1:7,j in 1:6,i in 1:5
                p=mesh.min_bounds .+ (i-1,j-1,k-1).*mesh.delta
                mesh.efield[i,j,k,:] .= linear(p...)
            end
            positions=T[-2 -3 -4; -0.7 0.2 0.4; 2 3 4; NaN Inf -Inf]
            x,y,z=(view(positions,:,i) for i in 1:3)
            output=fill(T(NaN),4,3); o=ntuple(i->view(output,:,i),3)
            mask=Bool[1,1,1,0]
            @test field!(o...,sc,x,y,z;mask=mask) === o
            for j in 1:3,i in 1:3
                @test o[j][i] ≈ linear(x[i],y[i],z[i])[j] rtol=20eps(T) atol=20eps(T)
            end
            @test output[4,:] == zeros(T,3)
            # Strided views and repeated output overwrite.
            dest=fill(T(NaN),8,3); so=ntuple(i->view(dest,1:2:8,i),3)
            field!(so...,sc,x,y,z;mask=mask)
            @test dest[1:2:8,:] == output
            @test all(isnan,dest[2:2:8,:])
            w=mesh._workspace; rho=copy(mesh.rho); grid=copy(mesh.efield)
            field!(o...,sc,x,y,z;mask=mask)
            @test mesh._workspace === w && mesh.rho == rho && mesh.efield == grid
            # Same prepared grid can be read by concurrent CPU tasks.
            results=fetch.([Threads.@spawn begin
                out=ntuple(_->zeros(T,4),3)
                field!(out...,sc,x,y,z;mask=mask)
            end for _ in 1:3])
            @test all(out->all(out[i]==o[i] for i in 1:3),results)
            @test field!(empty,empty,empty,sc,empty,empty,empty) === (empty,empty,empty)
        end
    end
end
