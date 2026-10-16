using SpaceCharge
using Test
using Random

function run_threaded_deposition_tests()
    @testset "Private-grid CPU deposition" begin
        @testset "Work selection bounds scratch and preserves serial fallback" begin
            for element_bytes in (4, 8), cells in (64, 64^3, 128^3, 256^3), workers in (1, 4, 24)
                chunks = SpaceCharge._deposit_chunk_count(10^7, cells, element_bytes, workers)
                @test 1 <= chunks <= min(workers, SpaceCharge._DEPOSIT_MAX_CHUNKS)
                if chunks > 1
                    @test chunks * cells * element_bytes <= SpaceCharge._DEPOSIT_MAX_SCRATCH_BYTES
                end
            end
            @test SpaceCharge._deposit_chunk_count(0, 64^3, 8, 8) == 1
            @test SpaceCharge._deposit_chunk_count(100, 64^3, 8, 8) == 1
            @test SpaceCharge._deposit_chunk_count(10^7, 256^3, 8, 8) == 1
        end

        # Exercise several logical chunks even in the one-thread test job.
        # The multi-thread job executes the same assertions concurrently.
        for T in (Float32, Float64)
            @testset "Signed charges and accumulation ($T)" begin
                rng = MersenneTwister(732)
                n = 3001  # Intentionally not a multiple of the chunk count.
                x, y, z = (T(0.001) .+ T(0.998) .* rand(rng, T, n) for _ in 1:3)
                q = randn(rng, T, n) .* T(1e-9)
                reference = FFTIGF((7, 9, 11); T)
                mesh = FFTIGF((7, 9, 11); T)
                SpaceCharge._set_grid_bounds!(reference,((0.,0.,0.),(1.,1.,1.)))
                SpaceCharge._set_grid_bounds!(mesh,((0.,0.,0.),(1.,1.,1.)))
                SpaceCharge.deposit_vectorized!(reference, x, y, z, q)
                scratch = SpaceCharge._get_deposit_workspace!(mesh, 5)
                SpaceCharge._deposit_threaded!(mesh, x, y, z, q, scratch, 5)
                atol = 64 * eps(T) * sum(abs, q)
                @test isapprox(mesh.rho, reference.rho; rtol=64eps(T), atol)
                @test abs(sum(mesh.rho) - sum(q)) <= atol
                @test SpaceCharge._get_deposit_workspace!(mesh, 5) === scratch
                @test mesh._workspace === nothing  # Deposition does not create FFT plans.

                # The private-grid path accumulates into an existing density.
                SpaceCharge._deposit_threaded!(mesh, x, y, z, q, scratch, 5)
                @test isapprox(mesh.rho, 2 .* reference.rho; rtol=128eps(T), atol=2atol)
                # Clearing scratch must prevent stale charge on later calls.
                fill!(mesh.rho, 0)
                SpaceCharge._deposit_threaded!(mesh, x, y, z, -q, scratch, 5)
                @test isapprox(mesh.rho, -reference.rho; rtol=64eps(T), atol)
            end
        end

        @testset "Public API and task context" begin
            rng = MersenneTwister(813)
            n = 83_017
            x, y, z = (0.001 .+ 0.998 .* rand(rng, n) for _ in 1:3)
            q = randn(rng, n) .* 1e-12
            sc = FFTIGF((9, 11, 13);fft_flags=FFTW.ESTIMATE)
            mesh = sc
            reference = FFTIGF((9, 11, 13);fft_flags=FFTW.ESTIMATE)
            SpaceCharge._set_grid_bounds!(reference,((0.,0.,0.),(1.,1.,1.)))
            solve!(sc,x,y,z,q,1;bounds=((0.,0.,0.),(1.,1.,1.)))
            SpaceCharge.deposit_vectorized!(reference, x, y, z, q)
            # No :static loop or threadid-indexed scratch: called from a task.
            fetch(Threads.@spawn deposit!(sc, x, y, z, q))
            atol = 128eps(Float64) * sum(abs, q)
            @test isapprox(mesh.rho, reference.rho; rtol=128eps(Float64), atol)
            scratch = mesh._deposit_workspace
            if Threads.nthreads() > 1
                @test scratch !== nothing
            else
                @test scratch === nothing
            end
            deposit!(sc, x, y, z, q; clear=false)
            @test isapprox(mesh.rho, 2 .* reference.rho; rtol=256eps(Float64), atol=2atol)
            @test mesh._deposit_workspace === scratch
            deposit!(sc, x, y, z, q; clear=true)
            @test isapprox(mesh.rho, reference.rho; rtol=128eps(Float64), atol)
            # Empty additions preserve charge; empty replacements are rejected.
            before_empty = copy(mesh.rho)
            empty = Float64[]
            deposit!(sc, empty, empty, empty, empty; clear=false)
            @test mesh.rho == before_empty
            @test_throws ArgumentError deposit!(sc, empty, empty, empty, empty; clear=true)
            @test mesh.rho == before_empty
            deposit!(sc,fill(NaN,n),fill(NaN,n),fill(NaN,n),fill(NaN,n); mask=falses(n),clear=false)
            @test mesh.rho == before_empty
            # A packed Boolean mask also skips invalid data in threaded batches.
            mask=trues(n);mask[2]=false;x[2]=NaN;q[2]=Inf
            fill!(reference.rho,0)
            SpaceCharge.deposit_vectorized!(reference,x,y,z,q,mask)
            fetch(Threads.@spawn deposit!(sc,x,y,z,q;mask=mask))
            @test isapprox(mesh.rho,reference.rho;rtol=128eps(Float64),atol)
        end
    end
end
