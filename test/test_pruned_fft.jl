using Random, LinearAlgebra

function run_pruned_fft_tests()
    @testset "Pruned CPU FFTs" begin
        rng = MersenneTwister(43107)
        @testset "Transforms $T $n" for T in (Float32, Float64), n in ((2,3,4), (5,7,3), (6,4,8))
            mesh = reference_test_mesh(T, n, 1.0)
            w = SpaceCharge._get_workspace(mesh; fft_flags=FFTW.ESTIMATE, pruned_fft=true)
            tol = T == Float32 ? 3e-6 : 2e-14
            # Repeated calls with changed signed sources expose stale padding.
            for image in (false, true, false)
                mesh.rho .= randn(rng, T, n...)
                SpaceCharge._charge_transform!(mesh, w, w.pruned, image)
                got = copy(w.rhohat)
                SpaceCharge._charge_transform!(mesh, w, nothing, image)
                @test got ≈ w.rhohat rtol=tol
            end
            # A generic real input, with no Green-kernel parity assumptions.
            dense = randn(rng, T, (2 .* n)...)
            w.product .= rfft(dense)
            SpaceCharge._field_inverse!(mesh, w, w.pruned, 1, inv(T(length(dense))), false)
            nx,ny,nz = n
            @test mesh.efield[:,:,:,1] ≈ dense[nx:2nx-1,ny:2ny-1,nz:2nz-1] rtol=tol
        end
        @testset "Full-path equivalence $T $n gamma=$gamma cathode=$cathode" for
            T in (Float32,Float64), n in ((4,6,8),(5,7,3),(7,4,6)),
            gamma in (1.,10.,100.), cathode in (false,true)
            full = reference_test_mesh(T,n,gamma)
            pruned = reference_test_mesh(T,n,gamma)
            # Shifted kernels retain the legacy boundary planes. Cancellation
            # during their convolution amplifies Float32 FFT rounding; on FFTW's
            # native eight-thread plans the full/pruned difference is ~5e-6.
            tol = T == Float32 ? (cathode ? 2e-5 : 3e-6) : 2e-13
            for state in (:initial,:changed,:refresh,:shift,:zero)
                if state == :changed
                    full.rho .*= T(-.37)
                    full.rho[end,1,1] += T(1e-11)
                elseif state == :shift
                    shift = T(.271) * full.delta[3]
                    full.min_bounds = full.min_bounds .+ (zero(T),zero(T),shift)
                    full.max_bounds = full.max_bounds .+ (zero(T),zero(T),shift)
                elseif state == :zero
                    fill!(full.rho,zero(T))
                end
                pruned.rho .= full.rho
                pruned.min_bounds = full.min_bounds
                pruned.max_bounds = full.max_bounds
                a = solve!(full; at_cathode=cathode,cache_green=state!=:refresh,pruned_fft=false)
                b = solve!(pruned; at_cathode=cathode,cache_green=state!=:refresh,pruned_fft=true)
                @test cathode ? b === pruned.efield : b === nothing
                for c in 1:3
                    @test pruned.efield[:,:,:,c] ≈ full.efield[:,:,:,c] rtol=tol
                end
                state in (:initial,:shift) && test_field_against_reference(pruned;at_cathode=cathode)
            end
        end
        @testset "Workspace mode and planning changes" begin
            mesh = reference_test_mesh(Float64,(4,5,6),10.)
            solve!(mesh;pruned_fft=true)
            expected = copy(mesh.efield)
            first = mesh._workspace
            solve!(mesh;pruned_fft=false)
            @test mesh._workspace !== first
            @test mesh._workspace.pruned === nothing
            @test mesh.efield ≈ expected rtol=2e-13
            solve!(mesh;pruned_fft=true,fft_flags=FFTW.ESTIMATE)
            @test mesh._workspace.pruned !== nothing
            @test mesh.efield ≈ expected rtol=2e-13
            mesh.rho = copy(mesh.rho)
            before = mesh._workspace
            solve!(mesh;pruned_fft=true,fft_flags=FFTW.ESTIMATE)
            @test mesh._workspace !== before
            @test mesh.efield ≈ expected rtol=2e-13
            previous_threads = FFTW.get_num_threads()
            try
                FFTW.set_num_threads(previous_threads == 1 ? 2 : 1)
                before = mesh._workspace
                solve!(mesh;pruned_fft=true,fft_flags=FFTW.ESTIMATE)
                @test mesh._workspace !== before
                @test mesh._workspace.fft_threads == FFTW.get_num_threads()
                @test mesh.efield ≈ expected rtol=2e-13
            finally
                FFTW.set_num_threads(previous_threads)
            end
        end
    end
end
