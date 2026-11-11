using Test, SpaceCharge, KernelAbstractions, FFTW

function run_prepared_tests(backend=CPU())
    to_backend(x) = copyto!(KernelAbstractions.allocate(backend, eltype(x), size(x)...), x)
    @testset "Prepared FFTIGF ($backend)" begin
        @testset "Setup contract and no physical solve" begin
            sc = FFTIGF((4, 5, 6); backend, fft_flags=FFTW.ESTIMATE)
            x = to_backend(fill(NaN, 4))
            mask = to_backend(fill(false, 4))
            prepared = prepare!(sc, x, x, x, x; mask)
            @test all(iszero, Array(sc.rho))
            @test all(iszero, Array(sc.efield))
            @test sc.min_bounds == (0.0, 0.0, 0.0)
            @test prepared.workspace === sc._workspace
            @test sc._workspace.free_cache.spectra !== nothing
            @test sc._workspace.free_cache.key === nothing
            @test_throws DimensionMismatch prepare!(sc, x, x, x, to_backend(zeros(3)))
            @test_throws ArgumentError prepare!(sc, x, x, x, to_backend(zeros(Float32, 4)))
            @test_throws ArgumentError prepare!(sc, x, x, x, x; mask=to_backend(zeros(Int, 4)))
            @test_throws DimensionMismatch prepare!(sc, x, x, x, x; mask=to_backend(fill(true, 3)))
            @test_throws ArgumentError prepare!(sc, x, x, x, x)
            empty_input = to_backend(Float64[])
            @test prepare!(sc, empty_input, empty_input, empty_input, empty_input) isa SpaceCharge.PreparedFFTIGF
            sc32 = FFTIGF((4, 5, 6); T=Float32, backend, fft_flags=FFTW.ESTIMATE)
            normal = to_backend(Float32[-1, 1])
            wide = to_backend(Float32[-3e38, 3e38])
            tiny = to_backend(Float32[0, nextfloat(0f0)])
            @test_throws ArgumentError prepare!(sc32, wide, normal, normal, normal)
            @test_throws ArgumentError prepare!(sc32, tiny, normal, normal, normal)
            # Validation must not replace the geometry of a previously solved field.
            coords = to_backend([-0.001, 0.002])
            charges = to_backend([1e-12, -2e-12])
            solve!(sc, coords, coords, coords, charges, 1.0)
            old_bounds, old_field = (sc.min_bounds, sc.max_bounds, sc.delta), Array(sc.efield)
            shifted = coords .+ 1
            prepare!(sc, shifted, shifted, shifted, charges)
            @test (sc.min_bounds, sc.max_bounds, sc.delta) == old_bounds
            @test Array(sc.efield) == old_field
            if !(backend isa CPU)
                @test_throws ArgumentError prepare!(sc, zeros(4), x, x, x)
                @test_throws ArgumentError prepare!(sc, x, x, x, x; mask=fill(true, 4))
            end
        end

        @testset "Charge refresh reuses a fixed Green spectrum" begin
            sc = FFTIGF((4, 5, 6); backend, fft_flags=FFTW.ESTIMATE)
            x = to_backend([-0.001, 0.002])
            q = to_backend([1e-12, -2e-12])
            prepared = prepare!(sc, x, x, x, q)
            solve_prepared!(prepared, x, x, x, q, 2.0)
            first_field = Array(sc.efield)
            cache = prepared.workspace.free_cache
            key, spectra = cache.key, cache.spectra
            spectrum_values = map(Array, spectra)
            q .*= 2
            solve_prepared!(prepared, x, x, x, q, 2.0)
            @test Array(sc.efield) ≈ 2 .* first_field rtol=5e-13
            @test cache.key == key
            @test cache.spectra === spectra
            @test map(Array, spectra) == spectrum_values
            again = prepare!(sc, x, x, x, q)
            @test again.workspace === prepared.workspace
            @test cache.key == key
        end

        for T in (Float32, Float64), field_method in (:direct, :potential_fd),
            at_cathode in (false, true), cache_green in (false, true),
            pruned_fft in (false, true)
            opts = (; T, backend, field_method, at_cathode, cache_green,
                    pruned_fft, fft_flags=FFTW.ESTIMATE)
            checked = FFTIGF((4, 5, 6); opts...)
            sc = FFTIGF((4, 5, 6); opts...)
            x = to_backend(T[-0.002, 0.001, 0.003, NaN])
            y = to_backend(T[0.001, -0.003, 0.002, NaN])
            z = to_backend(T[0.002, 0.003, 0.005, NaN])
            q = to_backend(T[1e-12, -2e-12, 3e-12, NaN])
            mask = to_backend(Bool[true, true, true, false])
            expected = ntuple(_ -> similar(x), 3)
            actual = ntuple(_ -> similar(x), 3)
            prepared = prepare!(sc, x, y, z, q; mask)
            workspace = prepared.workspace
            free_spectra = workspace.free_cache.spectra
            image_spectra = workspace.image_cache.spectra
            @test (free_spectra !== nothing) == cache_green
            @test (image_spectra !== nothing) == (cache_green && at_cathode)
            rtol = T === Float32 ? 5e-5 : 5e-13

            # Change gamma, domain widths, cathode offset, mask, and charges.
            # NaNs in excluded entries must never participate in either path.
            for step in 1:3
                gamma = T(step + 0.25)
                if step == 2
                    x .*= T(1.5)
                    y .*= T(0.75)
                    z .+= T(0.001)
                    q .*= T(-0.5)
                    copyto!(mask, Bool[true, false, true, false])
                elseif step == 3
                    copyto!(mask, Bool[true, true, true, false])
                end
                bounds = step == 3 ? (T.((-0.01, -0.01, 0)), T.((0.01, 0.01, 0.02))) : nothing
                solve!(checked, x, y, z, q, gamma; mask, bounds)
                @test solve_prepared!(prepared, x, y, z, q, gamma; mask, bounds) === nothing
                field!(expected..., checked, x, y, z; mask)
                @test field_prepared!(actual..., prepared, x, y, z; mask) === actual
                @test sc.min_bounds == checked.min_bounds
                @test sc.max_bounds == checked.max_bounds
                @test sc.gamma == gamma
                @test Array(sc.rho) ≈ Array(checked.rho) rtol=rtol
                for k in 1:3
                    @test Array(actual[k]) ≈ Array(expected[k]) rtol=rtol
                    @test Array(actual[k])[4] == 0
                    @test all(isfinite, Array(actual[k]))
                end
                @test sc._workspace === workspace
                @test workspace.free_cache.spectra === free_spectra
                @test workspace.image_cache.spectra === image_spectra
            end

            # A single included particle exercises the degenerate-bounds fit.
            copyto!(mask, Bool[true, false, false, false])
            for coord in (x, y, z)
                copyto!(coord, T[0, NaN, NaN, NaN])
            end
            solve!(checked, x, y, z, q, T(1); mask)
            solve_prepared!(prepared, x, y, z, q, T(1); mask)
            @test sc.delta == checked.delta
            @test Array(sc.efield) ≈ Array(checked.efield) rtol=rtol
        end

        if backend isa CPU
            @testset "Reprepare after resource changes" begin
                sc = FFTIGF((4, 5, 6); fft_flags=FFTW.ESTIMATE)
                x, q = [-0.001, 0.002], [1e-12, -2e-12]
                original = prepare!(sc, x, x, x, q)
                sc.rho, sc.efield = copy(sc.rho), copy(sc.efield)
                replaced = prepare!(sc, x, x, x, q)
                @test replaced.workspace !== original.workspace
                @test replaced.workspace.owner === sc.rho
                @test replaced.workspace.field_owner === sc.efield
                old_threads = FFTW.get_num_threads()
                try
                    FFTW.set_num_threads(old_threads == 1 ? 2 : 1)
                    rethreaded = prepare!(sc, x, x, x, q)
                    @test rethreaded.workspace !== replaced.workspace
                    @test rethreaded.workspace.fft_threads == FFTW.get_num_threads()
                    solve_prepared!(rethreaded, x, x, x, q, 1.0)
                    reference = FFTIGF((4, 5, 6); fft_flags=FFTW.ESTIMATE)
                    solve!(reference, x, x, x, q, 1.0)
                    @test sc.efield ≈ reference.efield rtol=5e-13
                finally
                    FFTW.set_num_threads(old_threads)
                end
            end

            @testset "Prepared threaded deposition" begin
                n = 40_000
                x = collect(range(-0.001, 0.001; length=n))
                y, z = sin.(x .* 2000) .* 0.001, cos.(x .* 1000) .* 0.001
                q = fill(1e-15, n)
                checked = FFTIGF((4, 5, 6); fft_flags=FFTW.ESTIMATE)
                sc = FFTIGF((4, 5, 6); fft_flags=FFTW.ESTIMATE)
                prepared = prepare!(sc, x, y, z, q)
                if Threads.nthreads() > 1
                    @test prepared.deposit_chunks > 1
                    @test prepared.deposit_workspace !== nothing
                end
                scratch = prepared.deposit_workspace
                for gamma in (1.0, 2.0)
                    solve!(checked, x, y, z, q, gamma)
                    solve_prepared!(prepared, x, y, z, q, gamma)
                    @test sc.rho ≈ checked.rho rtol=5e-14
                    @test sc.efield ≈ checked.efield rtol=5e-13
                    @test prepared.deposit_workspace === scratch
                end
            end
        end
    end
end
