using SpaceCharge, Test, FFTW
using AtomicAndPhysicalConstants: EPS_0, E_CHARGE

# Independent scalar eight-corner integral and direct O(N^2) sum. No production
# Green builders, FFT layout, or differencing kernels are used by this reference.
function reference_potential_primitive(x, y, z)
    r = sqrt(x*x + y*y + z*z)
    iszero(r) && return 0.0
    value = 0.0
    for (a, b, c) in ((x,y,z), (y,z,x), (z,x,y))
        !iszero(a) && (value -= a*a/2 * atan(b*c/(a*r)))
        !iszero(b*c) && (value += b*c*log(a+r))
    end
    return value
end

function reference_potential_grid(rho, delta, gamma; offset=(0.,0.,0.))
    n = size(rho)
    h = Float64.(delta) .* (1., 1., Float64(gamma))
    shift = Float64.(offset) .* (1., 1., Float64(gamma))
    kernel = zeros(2 .* n .- 1)
    for I in CartesianIndices(kernel)
        center = (Tuple(I) .- n) .* h .+ shift
        for signs in Iterators.product((-1,1), (-1,1), (-1,1))
            corner = center .+ signs .* h ./ 2
            kernel[I] += prod(signs) * reference_potential_primitive(corner...)
        end
        kernel[I] /= prod(h)
    end
    phi = zeros(n)
    for P in CartesianIndices(phi), Q in CartesianIndices(rho)
        K = CartesianIndex(Tuple(P) .- Tuple(Q) .+ n)
        phi[P] += Float64(rho[Q]) * kernel[K]
    end
    return phi .* inv(4pi * EPS_0 * E_CHARGE)
end

function reference_potential_field(mesh; cathode=mesh.options.at_cathode)
    phi = reference_potential_grid(Array(mesh.rho), mesh.delta, mesh.gamma)
    if cathode
        dz = 2mesh.min_bounds[3] + (mesh.max_bounds[3] - mesh.min_bounds[3])
        phi .+= reference_potential_grid(-reverse(Array(mesh.rho); dims=3),
            mesh.delta, mesh.gamma; offset=(0.,0.,dz))
    end
    result = zeros(mesh.grid_size..., 3)
    for d in 1:3
        n = mesh.grid_size[d]
        # Independent dense negative-derivative matrix on physical nodes.
        D = zeros(n,n)
        if n == 2
            D[:,1] .= 1; D[:,2] .= -1
            D ./= Float64(mesh.delta[d])
        else
            D[1,1:3] .= (3.,-4.,1.)
            D[n,n-2:n] .= (-1.,4.,-3.)
            for i in 2:n-1
                D[i,i-1] = 1; D[i,i+1] = -1
            end
            D ./= 2Float64(mesh.delta[d])
        end
        scale = d == 3 ? inv(Float64(mesh.gamma)) : Float64(mesh.gamma)
        for I in CartesianIndices(mesh.rho)
            result[I,d] = scale * sum(D[I[d],j] *
                phi[CartesianIndex(Base.setindex(Tuple(I),j,d))] for j in 1:n)
        end
    end
    return phi, result
end

function check_potential_reference(sc; cathode=sc.options.at_cathode)
    phi, expected = reference_potential_field(sc; cathode)
    observed = Array(sc.efield)
    tol = eltype(sc.rho) == Float32 ? 8e-4 : 3e-11
    @test all(isfinite, observed)
    w = sc._workspace
    observed_phi = if cathode
        Array(w.potential)
    else
        # Free-space fields consume the inverse crop without a normalized copy.
        # Retain the independent scalar-potential check at the same tolerance.
        nx, ny, nz = sc.grid_size
        raw = w.pruned === nothing ? view(w.result, nx:2nx-1, ny:2ny-1, nz:2nz-1) :
                                     view(w.pruned.result, nx:2nx-1, :, :)
        T = eltype(sc.rho)
        factor = T(inv(4pi * EPS_0 * E_CHARGE)) / T(length(w.pad))
        factor .* Array(raw)
    end
    @test observed_phi ≈ phi rtol=tol
    for c in 1:3
        @test observed[:,:,:,c] ≈ expected[:,:,:,c] rtol=tol
    end
end

function potential_reference_mesh(T, n, gamma; backend=CPU(), kwargs...)
    sc = FFTIGF(n; T, backend, field_method=:potential_fd,
                fft_flags=FFTW.ESTIMATE, kwargs...)
    SpaceCharge._set_grid_bounds!(sc, ((-.023,-.017,.011/gamma),(.029,.021,.039/gamma)))
    rho = [T(1e-12*(sin(.7i+.3j-.2k)+.17cos(i+2j+k)))
           for i in 1:n[1], j in 1:n[2], k in 1:n[3]]
    rho[1,end,1] += T(7e-12)
    rho[end,1,end] -= T(3e-12)
    copyto!(sc.rho, rho)
    return sc
end

function run_potential_reference_tests(backend=CPU())
    @testset "Potential FD independent reference $backend" begin
        @testset "$T n=$n gamma=$gamma cathode=$cathode" for
            T in (Float32,Float64), n in ((2,3,4),(5,4,3)),
            gamma in (1.,10.,100.), cathode in (false,true)
            sc = potential_reference_mesh(T, n, gamma; backend, at_cathode=cathode)
            solve!(sc, gamma)
            check_potential_reference(sc)
            @test length(sc._workspace.free_cache.spectra) == 1
            @test cathode ? size(sc._workspace.potential) == n : sc._workspace.potential === nothing
            @test size(sc._workspace.pad) == 2 .* n
            @test (sc._workspace.image_cache.spectra !== nothing) == cathode
        end
    end
    @testset "Potential images across the cathode and aligned corners $backend" begin
        for T in (Float32,Float64), zlo in (-5.,-2.,-1.,.25)
            sc = potential_reference_mesh(T, (3,4,5), 2.; backend, at_cathode=true)
            # Exact binary spacing puts a shifted longitudinal cell corner at
            # zero for zlo=.25. Also cover negative, symmetric, and crossing
            # domains; a zero image offset correctly shares the free spectrum.
            h = T(1/64)
            SpaceCharge._set_grid_bounds!(sc, ((-h,-2h,T(zlo)*h),(h,h,T(zlo+4)*h)))
            solve!(sc,2)
            check_potential_reference(sc)
            @test (sc._workspace.image_cache.spectra === nothing) == (zlo == -2.)
        end
    end
    @testset "Potential scratch is lazy and preserves plans $backend" begin
        for T in (Float32, Float64)
            sc = potential_reference_mesh(T, (3,4,5), 2.; backend)
            solve!(sc, 2)
            first_workspace = sc._workspace
            @test first_workspace.potential === nothing
            for cathode in (true, false, true)
                SpaceCharge._solve_mesh!(sc; at_cathode=cathode)
                w = sc._workspace
                @test w.forward === first_workspace.forward
                @test w.pruned === first_workspace.pruned
                @test w.free_cache === first_workspace.free_cache
                @test w.free_cache.spectra === first_workspace.free_cache.spectra
                check_potential_reference(sc; cathode)
            end
        end
    end
    @testset "Potential FD cache and transform reuse $backend" begin
        for T in (Float32,Float64), cathode in (false,true)
            sc = potential_reference_mesh(T, (3,4,5), 2.; backend, at_cathode=cathode)
            solve!(sc, 2)
            w = sc._workspace
            spectra = w.free_cache.spectra
            for state in (:repeat,:charge,:gamma,:spacing,:translation,:uncached,:recached,:zero)
                if state == :charge
                    sc.rho .*= T(-.37)
                elseif state == :spacing
                    sc.delta = sc.delta .* T.((1.03,.91,1.07))
                    sc.max_bounds = sc.min_bounds .+ sc.delta .* (sc.grid_size .- 1)
                elseif state == :translation
                    shift = (zero(T),zero(T),T(.271)*sc.delta[3])
                    sc.min_bounds = sc.min_bounds .+ shift
                    sc.max_bounds = sc.max_bounds .+ shift
                elseif state == :zero
                    fill!(sc.rho,zero(T))
                end
                SpaceCharge._solve_mesh!(sc; gamma=T(state == :gamma ? 3. : sc.gamma),
                                          cache_green=state != :uncached)
                @test sc._workspace === w
                @test w.free_cache.spectra === spectra
                check_potential_reference(sc)
                if backend isa CPU
                    expected = copy(sc.efield)
                    SpaceCharge._solve_mesh!(sc; pruned_fft=false)
                    @test sc._workspace.pruned === nothing
                    @test sc.efield ≈ expected rtol=(T == Float32 ? 2e-5 : 2e-13)
                    SpaceCharge._solve_mesh!(sc; pruned_fft=true)
                    w = sc._workspace
                    spectra = w.free_cache.spectra
                end
            end
            before = sc._workspace
            sc.rho = copy(sc.rho)
            solve!(sc, sc.gamma)
            @test sc._workspace !== before
            check_potential_reference(sc)
        end
    end
end

function run_potential_constructor_tests()
    @testset "Anisotropic potential primitive against BigFloat" begin
        # The nonzero Float32 transverse terms disappear in r's rounding, so
        # evaluating log(a+r) directly would produce -Inf instead of this limit.
        a, b, c = -1f0, 1f-4, 1f-4
        r = sqrt(a*a+b*b+c*c)
        @test r + a == 0
        aa, bb, cc = BigFloat.((a,b,c))
        rr = sqrt(aa*aa+bb*bb+cc*cc)
        expected_log = bb*cc*log(aa+rr)
        observed_log = SpaceCharge._potential_log_term(a,b,c,r)
        @test isfinite(observed_log)
        @test observed_log ≈ Float32(expected_log) rtol=4eps(Float32)
        for xyz in ((a,b,c),(b,a,c),(b,c,a))
            expected = reference_potential_primitive(BigFloat.(xyz)...)
            observed = SpaceCharge.potential_green_function(xyz...)
            @test isfinite(observed)
            @test observed ≈ Float32(expected) rtol=5e-5
        end
    end
    @testset "Field method selection" begin
        @test_throws ArgumentError FFTIGF((4,4,4); field_method=:unknown)
        for T in (Float32,Float64)
            default = FFTIGF((4,5,6); T, fft_flags=FFTW.ESTIMATE)
            direct = FFTIGF((4,5,6); T, field_method=:direct, fft_flags=FFTW.ESTIMATE)
            x = T[-.01,.02]; q = T[1e-12,-3e-12]
            solve!(default,x,x,x,q,2)
            solve!(direct,x,x,x,q,2)
            @test default.efield == direct.efield
            @test direct._workspace.potential === nothing
            @test length(direct._workspace.free_cache.spectra) == 3
            potential = FFTIGF((4,5,6); T, field_method=:potential_fd, fft_flags=FFTW.ESTIMATE)
            @test_throws MethodError solve!(potential,x,x,x,q)
            @test_throws MethodError solve!(potential)
            @test (@inferred solve!(potential,x,x,x,q,2)) === nothing
            @test (@inferred solve!(potential,2)) === nothing
        end
    end
end
