using SpaceCharge
using Test
using FFTW
using AtomicAndPhysicalConstants: EPS_0, E_CHARGE

# Independent baseline from commit 91ac2ee. Keep the primitive, eight-corner
# stencil, complex FFTs, and extraction convention here instead of calling the
# optimized production Green-function builder. Evaluate in Float64 even when the
# production mesh is Float32, so this comparison also checks rounding error.
function reference_field_primitive(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    return x * atan((y * z) / (r * x)) - z * log(r + y) +
           y * log((r - z) / (r + z)) / 2
end

function reference_integrated_field(rho, delta, gamma; offset=(0.0, 0.0, 0.0))
    nx, ny, nz = size(rho)
    dims = (2nx, 2ny, 2nz)
    dx, dy, dz = Float64(delta[1]), Float64(delta[2]), Float64(delta[3]) * Float64(gamma)
    g = Float64(gamma)
    origin = ((1 - dims[1]) * dx / 2 + Float64(offset[1]),
              (1 - dims[2]) * dy / 2 + Float64(offset[2]),
              (1 - dims[3]) * dz / 2 + Float64(offset[3]) * g)
    padded_rho = zeros(ComplexF64, dims)
    padded_rho[1:nx, 1:ny, 1:nz] .= rho
    rho_hat = fft(padded_rho)
    result = zeros(Float64, nx, ny, nz, 3)
    coulomb_factor = inv(4pi * EPS_0 * E_CHARGE)

    for component in 1:3
        primitive = zeros(ComplexF64, dims)
        factor = (component < 3 ? g : 1.0) / (dx * dy * dz)
        for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
            x = (i - 1) * dx + origin[1]
            y = (j - 1) * dy + origin[2]
            z = (k - 1) * dz + origin[3]
            arguments = component == 1 ? (x, y, z) :
                        component == 2 ? (y, z, x) : (z, x, y)
            primitive[i, j, k] = reference_field_primitive(arguments...) * factor
        end
        # The old solver leaves the last plane in each direction undifferenced.
        # Those planes do not contribute to the extracted physical convolution,
        # but preserving them makes this a faithful baseline implementation.
        integrated = copy(primitive)
        for k in 1:dims[3]-1, j in 1:dims[2]-1, i in 1:dims[1]-1
            integrated[i, j, k] = primitive[i+1, j+1, k+1] - primitive[i, j+1, k+1] -
                primitive[i+1, j, k+1] - primitive[i+1, j+1, k] - primitive[i, j, k] +
                primitive[i, j, k+1] + primitive[i, j+1, k] + primitive[i+1, j, k]
        end
        convolution = ifft(rho_hat .* fft(integrated))
        result[:, :, :, component] .= coulomb_factor .* real.(
            convolution[nx:2nx-1, ny:2ny-1, nz:2nz-1])
    end
    return result
end

function reference_mesh_field(mesh; at_cathode=false)
    field = reference_integrated_field(mesh.rho, mesh.delta, mesh.gamma)
    if at_cathode
        # Form the offset in the mesh's arithmetic, as the public solver does.
        offset_z = 2 * mesh.min_bounds[3] + (mesh.max_bounds[3] - mesh.min_bounds[3])
        field .+= reference_integrated_field(-reverse(mesh.rho; dims=3), mesh.delta,
            mesh.gamma; offset=(0.0, 0.0, offset_z))
    end
    return field
end

function reference_test_mesh(T, n, gamma; backend=CPU(), kwargs...)
    bounds = ((-.023,-.017,.011/gamma),(.029,.021,.039/gamma))
    axes = ntuple(d->T.(range(bounds[1][d],bounds[2][d];length=n[d])),3)
    points = collect(Iterators.product(axes...))
    x,y,z = ntuple(d->vec(getindex.(points,d)),3)
    q = [T(1e-12*(sin(.7i+.3j-.2k)+.17cos(i+2j+k)))
         for i in 1:n[1], j in 1:n[2], k in 1:n[3]]
    q[1,end,1] += T(7e-12); q[end,1,end] -= T(3e-12)
    particles = map((x,y,z,vec(q))) do a
        dest = KernelAbstractions.zeros(backend,T,length(a))
        copyto!(dest,a)
    end
    mesh = FFTIGF(n; T, backend, fft_flags=FFTW.ESTIMATE, kwargs...)
    solve!(mesh,particles...,gamma;bounds)
    return mesh
end

function test_field_against_reference(mesh; at_cathode=false)
    expected = reference_mesh_field(mesh; at_cathode=at_cathode)
    # Float32 includes cancellation in the primitive and corner differences;
    # compare each component so a boosted transverse field cannot hide Ez errors.
    tolerance = eltype(mesh.rho) == Float32 ? 3e-4 : 2e-10
    for component in 1:3
        @test mesh.efield[:, :, :, component] ≈ expected[:, :, :, component] rtol=tolerance
    end
end

function run_solver_reference_tests()
    @testset "Independent integrated-field reference" begin
        # Pair grid shapes with stationary/boosted cases; cache tests cover gamma changes.
        @testset "$T, grid=$grid_size, gamma=$gamma, cathode=$at_cathode" for
            T in (Float32, Float64), at_cathode in (false, true),
            (grid_size, gamma) in (((4, 5, 6), 1.0), ((7, 4, 5), 100.0))
            mesh = reference_test_mesh(T, grid_size, gamma; at_cathode)
            test_field_against_reference(mesh; at_cathode=at_cathode)
        end
    end
end

function run_solver_reuse_tests(backend=CPU())
    @testset "Repeated solves follow changed physical inputs on $backend" begin
        for method in (:direct,:potential_fd), (T,cathode) in ((Float32,true),(Float64,false))
            sc = FFTIGF((5,6,4); T, backend, field_method=method,
                        at_cathode=cathode, fft_flags=FFTW.ESTIMATE)
            x,y,z,q = map(a->test_array(backend,T.(a)),
                ([-.01,.013],[.007,-.012],[.015,.028],[1e-12,-.7e-12]))
            bounds = ((-.02,-.03,.01),(.02,.03,.04)); gamma = 2.
            for state in (:initial,:repeat,:charge,:gamma,:spacing,:translation,:zero)
                if state == :charge
                    q .*= T(-.37)
                elseif state == :gamma
                    gamma = 3.
                elseif state == :spacing
                    bounds = ((-.03,-.04,.005),(.03,.04,.045))
                elseif state == :translation
                    z .+= T(.002)
                    bounds = map(b->(b[1],b[2],b[3]+.002),bounds)
                elseif state == :zero
                    fill!(q,zero(T))
                end
                solve!(sc,x,y,z,q,gamma;bounds)
                if method == :direct
                    snapshot = (rho=Array(sc.rho),efield=Array(sc.efield),delta=sc.delta,
                                gamma=sc.gamma,min_bounds=sc.min_bounds,max_bounds=sc.max_bounds)
                    test_field_against_reference(snapshot;at_cathode=cathode)
                else
                    check_potential_reference(sc)
                end
            end
        end
    end
end
