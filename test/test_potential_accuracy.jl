using Test, SpaceCharge, FFTW, KernelAbstractions
using LinearAlgebra: norm
using AtomicAndPhysicalConstants: EPS_0, E_CHARGE

function potential_test_array(backend, values)
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
        x, y, z = ntuple(d -> potential_test_array(backend, getindex.(probes, d)), 3)
        outputs = ntuple(_ -> similar(x), 3)
        for (level, n) in enumerate((17, 33, 65))
            sc = FFTIGF((n,n,n); backend, field_method=:potential_fd,
                        fft_flags=FFTW.ESTIMATE)
            SpaceCharge._set_grid_bounds!(sc, ((-.016,-.016,-.016),(.016,.016,.016)))
            rho = zeros(n,n,n)
            center = (n+1) ÷ 2
            rho[center,center,center] = charge
            copyto!(sc.rho, rho)
            solve!(sc, 1)
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
            x, y, z, q = map(v -> potential_test_array(backend, T.(v)),
                ([-.009,.011,.017],[-.013,.006,.012],[.003,.005,.009],
                 [1e-12,-.4e-12,.7e-12]))
            lab = FFTIGF((7,6,5); T, backend, at_cathode,
                         field_method=:potential_fd, fft_flags=FFTW.ESTIMATE)
            rest = FFTIGF((7,6,5); T, backend, at_cathode,
                          field_method=:potential_fd, fft_flags=FFTW.ESTIMATE)
            solve!(lab, x,y,z,q,g; bounds)
            solve!(rest, x,y,g.*z,q,1; bounds=rest_bounds)
            @test Array(lab.rho) == Array(rest.rho)
            a, b = Array(lab.efield), Array(rest.efield)
            tolerance = T == Float32 ? 5e-6 : 3e-13
            for d in 1:3
                @test a[:,:,:,d] ≈ (d < 3 ? g : one(T)) .* b[:,:,:,d] rtol=tolerance
            end
            output_lab, output_rest = ntuple(_ -> similar(x),3), ntuple(_ -> similar(x),3)
            field!(output_lab...,lab,x,y,z)
            field!(output_rest...,rest,x,y,g.*z)
            for d in 1:3
                @test Array(output_lab[d]) ≈ (d < 3 ? g : one(T)) .* Array(output_rest[d]) rtol=tolerance
            end
        end
    end
end

function potential_test_cell_average(center, h)
    value = 0.0
    for signs in Iterators.product((-1,1),(-1,1),(-1,1))
        x,y,z = center .+ signs .* h ./ 2
        r = sqrt(x*x+y*y+z*z)
        # These nodal-cell corners never lie on a coordinate plane. Keep the
        # analytic primitive here independent of all production Green builders.
        primitive = x*y*log(z+r) + x*z*log(y+r) + y*z*log(x+r) -
            (x*x*atan(y*z/(x*r)) + y*y*atan(x*z/(y*r)) + z*z*atan(x*y/(z*r))) / 2
        value += prod(signs) * primitive
    end
    return value / prod(h)
end

function test_potential_boundary_symmetry(backend)
    @testset "One-sided boundary self-force and interior symmetry" begin
        for T in (Float32, Float64), n in ((5,6,7), (2,3,4))
            sc = FFTIGF(n; T, backend, field_method=:potential_fd,
                        fft_flags=FFTW.ESTIMATE)
            SpaceCharge._set_grid_bounds!(sc, ((-.012,-.016,-.008),(.020,.016,.008)))
            middle = ntuple(d -> (n[d]+1)÷2,3)
            faces = [ntuple(k -> k == d ? side : middle[k],3)
                     for d in 1:3 for side in (1,n[d])]
            corners = [(i,j,k) for i in (1,n[1]) for j in (1,n[2]) for k in (1,n[3])]
            rho = zeros(T,n)
            h = Float64.(sc.delta) .* (1.,1.,4.)
            coefficient = 1e-12 / (4pi * EPS_0 * E_CHARGE)
            own_potential = potential_test_cell_average((0.,0.,0.),h)
            tolerance = T == Float32 ? 3e-5 : 3e-12
            # A one-sided derivative of an even self potential is nonzero.
            # Assemble its expected value from the exact cell integral at
            # offsets 0,h,2h, independently of the FFT and production stencil.
            # A two-node dimension instead uses the same secant at both faces.
            for index in unique(vcat(faces,corners))
                fill!(rho,zero(T)); rho[index...] = T(1e-12)
                copyto!(sc.rho,rho); solve!(sc,4)
                field = Array(sc.efield)
                scale = maximum(abs,field)
                for d in 1:3
                    observed = field[index...,d]
                    if index[d] in (1,n[d])
                        offset = ntuple(k -> k == d ? h[d] : 0.,3)
                        neighbor = potential_test_cell_average(offset,h)
                        derivative = if n[d] == 2
                            (own_potential-neighbor) / sc.delta[d]
                        else
                            second_neighbor = potential_test_cell_average(2 .* offset,h)
                            (3own_potential-4neighbor+second_neighbor) / (2sc.delta[d])
                        end
                        direction = index[d] == 1 ? 1 : -1
                        expected = direction * coefficient * (d == 3 ? .25 : 4.) * derivative
                        @test observed ≈ expected rtol=tolerance atol=128eps(T)*scale
                        @test direction * observed > 0
                    else
                        @test abs(observed) < 128eps(T)*scale
                    end
                end
            end

            # Centered interior derivatives retain zero self-force and odd pair
            # interactions. Boundary charges intentionally do not obey these
            # invariants with one-sided stencils; do not require global balance.
            all(n .>= 3) || continue
            fill!(rho,zero(T)); rho[middle...] = T(1e-12)
            copyto!(sc.rho,rho); solve!(sc,4)
            field = Array(sc.efield)
            @test maximum(abs,field[middle...,:]) < 128eps(T)*maximum(abs,field)
            fill!(rho,zero(T))
            interior = [(2,2,2),(n[1]-1,n[2]-1,n[3]-1),middle,(2,n[2]-1,3)]
            for (k,index) in enumerate(unique(interior))
                rho[index...] += T(1e-12 * (-1)^k * (1 + k/10))
            end
            copyto!(sc.rho,rho); solve!(sc,4)
            field = Array(sc.efield)
            for d in 1:3
                force = rho .* field[:,:,:,d]
                scale = sum(abs,rho) * maximum(abs,field[:,:,:,d])
                @test abs(sum(force)) < 128eps(T)*scale
            end
            # Check a pair explicitly, including its own-cell contributions.
            fill!(rho,zero(T))
            left, right = (2,2,2), n .- 1
            rho[left...] = T(1e-12); rho[right...] = T(-.7e-12)
            copyto!(sc.rho,rho); solve!(sc,4)
            field = Array(sc.efield)
            for d in 1:3
                fleft = rho[left...] * field[left...,d]
                fright = rho[right...] * field[right...,d]
                # The error budget includes FFT/subtraction noise relative to
                # the larger near-source field, not only this distant pair.
                scale = sum(abs,rho) * maximum(abs,field[:,:,:,d])
                @test abs(fleft + fright) < 128eps(T)*scale
            end
        end
    end
end

function test_potential_cathode_plane(backend)
    @testset "Cathode plane tangential cancellation" begin
        for T in (Float32,Float64)
            bounds = ((-.020,-.030,0.),(.020,.030,.040))
            x,y,z,q = map(v -> potential_test_array(backend,T.(v)),
                ([-.006,.013],[.007,-.012],[.015,.028],[1e-12,-.4e-12]))
            free = FFTIGF((7,8,9); T,backend,field_method=:potential_fd,fft_flags=FFTW.ESTIMATE)
            cathode = FFTIGF((7,8,9); T,backend,field_method=:potential_fd,
                             at_cathode=true,fft_flags=FFTW.ESTIMATE)
            solve!(free,x,y,z,q,2;bounds)
            solve!(cathode,x,y,z,q,2;bounds)
            deposited = Array(cathode.rho)
            # Shifted primitives do not share the free kernel's enforced
            # reflection. Float32 potential cancellation is amplified by the
            # one-sided stencil: |3|+|-4|+|1| over 2h gives 4/h, versus 1/h for
            # centered differences. The Float32 allowance also covers CPU/CUDA
            # primitive and FFT roundoff; retain the strict Float64 test.
            tolerance = T == Float32 ? 4e-3 : 2e-11
            for _ in 1:2
                solve!(cathode,2)
                field = Array(cathode.efield)
                free_field = Array(free.efield)
                scale = maximum(abs,free_field[:,:,1,:])
                @test maximum(abs,field[:,:,1,1:2]) < tolerance*scale
                @test Array(cathode.rho) == deposited
            end
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
            SpaceCharge._set_grid_bounds!(sc, ((-.016,-.016,0.),(.016,.016,.032)))
            center = (n+1) ÷ 2
            rho = zeros(n,n,n); rho[center,center,center] = charge
            copyto!(sc.rho,rho); solve!(sc,1)
            field = Array(sc.efield)
            observed = field[center+(n-1)÷4,center,1,3]
            push!(errors,abs(observed-expected)/abs(expected))
        end
        @test 3.0 < errors[1]/errors[2] < 6.0
        @test 3.3 < errors[2]/errors[3] < 5.0
        @test errors[3] < .01
    end
end

function test_potential_shared_particle_api(backend)
    @testset "Masked and staged APIs preserve deposition and geometry" begin
        for T in (Float32,Float64), at_cathode in (false,true), fitted in (false,true)
            x,y,z,q = map(v -> potential_test_array(backend,T.(v)),
                ([-.006,.007,NaN],[-.005,.004,NaN],[.002,.010,NaN],
                 [1e-12,-.7e-12,NaN]))
            mask = potential_test_array(backend,Bool[true,true,false])
            bounds = fitted ? nothing : ((-.012,-.010,0.),(.012,.010,.016))
            direct = FFTIGF((7,6,5);T,backend,at_cathode,fft_flags=FFTW.ESTIMATE)
            fd = FFTIGF((7,6,5);T,backend,at_cathode,
                        field_method=:potential_fd,fft_flags=FFTW.ESTIMATE)
            staged = FFTIGF((7,6,5);T,backend,at_cathode,
                            field_method=:potential_fd,fft_flags=FFTW.ESTIMATE)
            @test solve!(direct,x,y,z,q,4;mask,bounds) === nothing
            @test solve!(fd,x,y,z,q,4;mask,bounds) === nothing
            @test (fd.min_bounds,fd.max_bounds,fd.delta) ==
                  (direct.min_bounds,direct.max_bounds,direct.delta)
            @test Array(fd.rho) == Array(direct.rho)
            solve!(staged,x,y,z,q,1;mask,bounds)
            @test deposit!(staged,x,y,z,q;mask) === staged
            @test solve!(staged,4) === nothing
            @test (staged.min_bounds,staged.max_bounds,staged.delta) ==
                  (fd.min_bounds,fd.max_bounds,fd.delta)
            @test Array(staged.rho) == Array(fd.rho)
            tolerance = T == Float32 ? 5e-6 : 3e-13
            @test Array(staged.efield) ≈ Array(fd.efield) rtol=tolerance
            output, staged_output = ntuple(_ -> similar(x),3), ntuple(_ -> similar(x),3)
            @test field!(output...,fd,x,y,z;mask) === output
            field!(staged_output...,staged,x,y,z;mask)
            for d in 1:3
                a,b = Array(output[d]),Array(staged_output[d])
                @test all(isfinite,a)
                @test a[3] == 0
                @test a ≈ b rtol=tolerance
            end
            rho = Array(staged.rho)
            deposit!(staged,x,y,z,q;mask,clear=false)
            @test Array(staged.rho) ≈ 2rho rtol=8eps(T)
            solve!(staged,4)
            @test Array(staged.efield) ≈ 2Array(fd.efield) rtol=tolerance
        end
    end
end

function run_potential_accuracy_tests(backend=CPU())
    @testset "Potential-FD accuracy and invariants on $backend" begin
        test_potential_point_convergence(backend)
        test_potential_gamma_identity(backend)
        test_potential_boundary_symmetry(backend)
        test_potential_cathode_plane(backend)
        test_potential_shared_particle_api(backend)
    end
end
