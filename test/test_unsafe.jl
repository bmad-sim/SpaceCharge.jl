function run_unsafe_tests(backend=CPU())
    to_backend(x) = copyto!(KernelAbstractions.allocate(backend, eltype(x), size(x)...), x)
    @testset "Unchecked FFTIGF ($backend)" begin
        for T in (Float32, Float64), field_method in (:direct, :potential_fd)
            opts = (; T, backend, field_method, fft_flags=FFTW.ESTIMATE)
            checked = FFTIGF((4, 5, 6); opts...)
            unchecked = FFTIGF((4, 5, 6); opts...)
            x = to_backend(T[-0.002, 0.001, 0.003, NaN])
            y = to_backend(T[0.001, -0.003, 0.002, NaN])
            z = to_backend(T[0.002, 0.003, 0.005, NaN])
            q = to_backend(T[1e-12, -2e-12, 3e-12, NaN])
            mask = to_backend(Bool[true, true, true, false])
            expected = ntuple(_ -> similar(x), 3)
            actual = ntuple(_ -> similar(x), 3)
            rtol = T === Float32 ? 5e-5 : 5e-13
            workspace = nothing
            explicit_bounds = (T.((-0.01, -0.01, 0)), T.((0.01, 0.01, 0.02)))
            for (gamma, bounds) in ((T(1), nothing), (T(2), explicit_bounds))
                solve!(checked, x, y, z, q, gamma; mask, bounds)
                @test unsafe_solve!(unchecked, x, y, z, q, gamma; mask, bounds) === nothing
                field!(expected..., checked, x, y, z; mask)
                @test unsafe_field!(actual..., unchecked, x, y, z; mask) === actual
                @test unchecked.delta == checked.delta
                for k in 1:3
                    @test Array(actual[k]) ≈ Array(expected[k]) rtol=rtol
                    @test iszero(Array(actual[k])[4])
                end
                if workspace === nothing
                    workspace = unchecked._workspace
                else
                    @test unchecked._workspace === workspace
                end
            end
        end
    end
end
