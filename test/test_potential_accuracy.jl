using Test, SpaceCharge, FFTW, KernelAbstractions
using LinearAlgebra: norm
using AtomicAndPhysicalConstants: EPS_0, E_CHARGE

function test_array(backend, values)
    result = KernelAbstractions.zeros(backend, eltype(values), size(values)...)
    copyto!(result, values)
    return result
end

# Fixed physical probes are grid nodes at every resolution. A single nodal
# charge avoids sampling/deposition noise; its shrinking integrated source cell
# and the potential derivative converge to the analytic point field.
function test_potential_point_convergence(backend)
    @testset "Point-charge convergence, including faces" begin
        probes = [(0.008, 0.0, 0.0), (0.016, 0.0, 0.0),
                  (0.008, 0.008, 0.008), (-0.016, 0.008, -0.004)]
        charge = 1e-12
        coefficient = charge / (4pi * EPS_0 * E_CHARGE)
        expected = [coefficient .* collect(p) ./ norm(p)^3 for p in probes]
        errors = zeros(3, length(probes))
        x, y, z = ntuple(d -> test_array(backend, getindex.(probes, d)), 3)
        outputs = ntuple(_ -> similar(x), 3)
        for (level, n) in enumerate((17, 33, 65))
            sc = FFTIGF((n,n,n); backend, field_method=:potential_fd,
                        fft_flags=FFTW.ESTIMATE)
            source = test_array(backend,[0.])
            q = test_array(backend,[charge])
            solve!(sc,source,source,source,q,1;bounds=((-.016,-.016,-.016),(.016,.016,.016)))
            field!(outputs..., sc, x, y, z)
            measured = Array.(outputs)
            for p in eachindex(probes)
                value = [measured[d][p] for d in 1:3]
                errors[level,p] = norm(value - expected[p]) / norm(expected[p])
            end
        end
        # Halving h reduces the leading O(h^2) error by about four. Bounds allow
        # pre-asymptotic one-sided boundary corrections and the coarsest r=4h
        # probe. One-sided derivatives approach the asymptotic rate from above.
        for p in eachindex(probes)
            @test 2.8 < errors[1,p] / errors[2,p] < 5.5
            @test 3.4 < errors[2,p] / errors[3,p] < 4.6
            @test errors[3,p] < 0.005
        end
        @test maximum(errors[3,[2,4]]) < 0.003
    end
end

function test_potential_gamma_identity(backend)
    @testset "Rest/lab gamma identity" begin
        for T in (Float32, Float64), at_cathode in (false, true)
            # A power-of-two stretch makes both geometries exactly equivalent
            # in floating-point arithmetic, isolating the gamma transformations.
            g = T(8)
            bounds = (T.((-.012,-.016,.002)), T.((.020,.016,.010)))
            rest_bounds = map(b -> (b[1], b[2], g*b[3]), bounds)
            x, y, z, q = map(v -> test_array(backend, T.(v)),
                ([-.009,.011,.017],[-.013,.006,.012],[.003,.005,.009],
                 [1e-12,-.4e-12,.7e-12]))
            lab = FFTIGF((7,6,5); T, backend, at_cathode,
                         field_method=:potential_fd, fft_flags=FFTW.ESTIMATE)
            rest = FFTIGF((7,6,5); T, backend, at_cathode,
                          field_method=:potential_fd, fft_flags=FFTW.ESTIMATE)
            solve!(lab, x,y,z,q,g; bounds)
            solve!(rest, x,y,g.*z,q,1; bounds=rest_bounds)
            tolerance = T == Float32 ? 5e-6 : 3e-13
            output_lab, output_rest = ntuple(_ -> similar(x),3), ntuple(_ -> similar(x),3)
            field!(output_lab...,lab,x,y,z)
            field!(output_rest...,rest,x,y,g.*z)
            for d in 1:3
                @test Array(output_lab[d]) ≈ (d < 3 ? g : one(T)) .* Array(output_rest[d]) rtol=tolerance
            end
        end
    end
end

function test_potential_cathode_plane(backend)
    @testset "Cathode plane tangential cancellation" begin
        for T in (Float32,Float64)
            bounds = ((-.020,-.030,0.),(.020,.030,.040))
            x,y,z,q = map(v -> test_array(backend,T.(v)),
                ([-.006,.013],[.007,-.012],[.015,.028],[1e-12,-.4e-12]))
            free = FFTIGF((7,8,9); T,backend,field_method=:potential_fd,fft_flags=FFTW.ESTIMATE)
            cathode = FFTIGF((7,8,9); T,backend,field_method=:potential_fd,
                             at_cathode=true,fft_flags=FFTW.ESTIMATE)
            solve!(free,x,y,z,q,2;bounds)
            solve!(cathode,x,y,z,q,2;bounds)
            # Shifted primitives do not share the free kernel's enforced
            # reflection. Float32 potential cancellation is amplified by the
            # one-sided stencil: |3|+|-4|+|1| over 2h gives 4/h, versus 1/h for
            # centered differences. The Float32 allowance also covers CPU/CUDA
            # primitive and FFT roundoff; retain the strict Float64 test.
            tolerance = T == Float32 ? 4e-3 : 2e-11
            field = Array(cathode.efield)
            free_field = Array(free.efield)
            scale = maximum(abs,free_field[:,:,1,:])
            @test maximum(abs,field[:,:,1,1:2]) < tolerance*scale
        end
    end
    @testset "Cathode normal field converges to analytic image field" begin
        # At the cathode the one-sided normal derivative is not constrained to
        # twice the discrete free-space derivative. Test the physical limit of
        # a nodal charge plus its negative image instead of enforcing doubling.
        charge = 1e-12
        source_z, probe_x = .016, .008
        coefficient = charge / (4pi * EPS_0 * E_CHARGE)
        expected = -2coefficient * source_z / (probe_x^2 + source_z^2)^1.5
        errors = Float64[]
        for n in (9,17,33)
            sc = FFTIGF((n,n,n);backend,field_method=:potential_fd,
                        at_cathode=true,fft_flags=FFTW.ESTIMATE)
            source = test_array(backend,[0.])
            z = test_array(backend,[source_z]); q = test_array(backend,[charge])
            solve!(sc,source,source,z,q,1;bounds=((-.016,-.016,0.),(.016,.016,.032)))
            center = (n+1) ÷ 2
            field = Array(sc.efield)
            observed = field[center+(n-1)÷4,center,1,3]
            push!(errors,abs(observed-expected)/abs(expected))
        end
        @test 3.0 < errors[1]/errors[2] < 6.0
        @test 3.3 < errors[2]/errors[3] < 5.0
        @test errors[3] < .01
    end
end

function run_potential_accuracy_tests(backend=CPU())
    @testset "Potential-FD accuracy and invariants on $backend" begin
        test_potential_point_convergence(backend)
        test_potential_gamma_identity(backend)
        test_potential_cathode_plane(backend)
    end
end
