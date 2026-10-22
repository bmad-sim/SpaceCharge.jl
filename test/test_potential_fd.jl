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

function check_potential_reference(sc)
    _, expected = reference_potential_field(sc)
    observed = Array(sc.efield)
    tol = eltype(sc.rho) == Float32 ? 8e-4 : 3e-11
    for c in 1:3
        @test observed[:,:,:,c] ≈ expected[:,:,:,c] rtol=tol
    end
end

function run_potential_reference_tests(backend=CPU())
    @testset "Potential FD independent reference $backend" begin
        # Keep both two-node and ordinary stencils, and stationary/boosted fields.
        @testset "$T n=$n gamma=$gamma cathode=$cathode" for
            T in (Float32,Float64), cathode in (false,true),
            (n, gamma) in (((2,3,4),1.), ((3,2,2),10.), ((5,4,3),100.))
            sc = reference_test_mesh(T, n, gamma; backend, at_cathode=cathode, field_method=:potential_fd)
            check_potential_reference(sc)
        end
    end
    @testset "Potential images across the cathode and aligned corners $backend" begin
        for T in (Float32,Float64), zlo in (-5.,-2.,-1.,.25)
            # Binary geometry includes negative, symmetric, crossing, and
            # cell-corner-aligned image domains.
            h = T(1/64)
            bounds = ((-h,-2h,T(zlo)*h),(h,h,T(zlo+4)*h))
            sc = FFTIGF((3,4,5); T, backend, at_cathode=true,
                        field_method=:potential_fd, fft_flags=FFTW.ESTIMATE)
            x,y,z,q = map(a->test_array(backend,T.(a)),
                ([0,h/2],[0,-h],[bounds[1][3]+h,bounds[2][3]-h],[1e-12,-.7e-12]))
            solve!(sc,x,y,z,q,2;bounds)
            check_potential_reference(sc)
        end
    end
end

function run_potential_anisotropy_test()
    @testset "Anisotropic Float32 field at an axial probe" begin
        # At this aspect ratio, transverse terms disappear in rounded radii.
        # The physical field remains finite despite cancellation in log terms.
        sc = FFTIGF((3,3,3); T=Float32, field_method=:potential_fd, fft_flags=FFTW.ESTIMATE)
        x = Float32[0]; q = Float32[1e-12]
        solve!(sc,x,x,x,q,1;bounds=((-1.,-1e-4,-1e-4),(1.,1e-4,1e-4)))
        out = ntuple(_->similar(x),3)
        field!(out...,sc,Float32[1],x,x)
        _, expected = reference_potential_field(sc)
        @test out[1][1] ≈ expected[3,2,2,1] rtol=8e-4
        scale = maximum(abs,sc.efield)
        @test abs(out[2][1]) < 128eps(Float32)*scale
        @test abs(out[3][1]) < 128eps(Float32)*scale
    end
end
