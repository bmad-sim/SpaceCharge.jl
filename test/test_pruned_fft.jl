using Test, SpaceCharge, FFTW

function run_pruned_fft_tests()
    @testset "Public transform and cache options give the same field" begin
        for T in (Float32,Float64), method in (:direct,:potential_fd), cathode in (false,true)
            n = (5,4,6) # Mixed even/odd dimensions exercise the transform crop.
            bounds = ((-.02,-.03,.01),(.02,.03,.04))
            x,y,z,q = T[-.01,.013],T[.007,-.012],T[.015,.028],T[1e-12,-.7e-12]
            reference = FFTIGF(n; T, field_method=method, at_cathode=cathode,
                              fft_flags=FFTW.ESTIMATE)
            solve!(reference,x,y,z,q,2;bounds)
            # Cache and FFT strategy are constructor options, not mutable plan state.
            for options in ((pruned_fft=false,cache_green=true),
                            (pruned_fft=true,cache_green=false))
                sc = FFTIGF(n; T, field_method=method, at_cathode=cathode,
                            fft_flags=FFTW.ESTIMATE, options...)
                solve!(sc,x,y,z,q,2;bounds)
                tol = T == Float32 ? (method == :direct && !cathode ? 3e-6 : 2e-5) : 2e-13
                for c in 1:3
                    @test sc.efield[:,:,:,c] ≈ reference.efield[:,:,:,c] rtol=tol
                end
            end
        end
    end
end
