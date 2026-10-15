using SpaceCharge
using Test
using FFTW

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
    coulomb_factor = 299792458.0^2 * 1.0e-7

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

function reference_test_mesh(T, grid_size, gamma)
    # Contract z with gamma to retain moderate cell aspect ratios in the rest
    # frame. This tests boosted normalization without the known ill-conditioning
    # of the old primitive's Float32 log((r-z)/(r+z)) at extreme aspect ratios.
    mesh = FFTIGF(grid_size; bounds=((-0.023, -0.017, 0.011 / gamma),
        (0.029, 0.021, 0.039 / gamma)), T=T).mesh
    mesh.gamma = T(gamma)
    for k in 1:grid_size[3], j in 1:grid_size[2], i in 1:grid_size[1]
        mesh.rho[i, j, k] = T(1e-12 * (sin(0.7i + 0.3j - 0.2k) + 0.17cos(i + 2j + k)))
    end
    # Unequal signed sources at opposite boundaries expose flips and shifts.
    mesh.rho[1, end, 2] += T(7e-12)
    mesh.rho[end, 2, end] -= T(3e-12)
    return mesh
end

function test_field_against_reference(mesh; at_cathode=false)
    expected = reference_mesh_field(mesh; at_cathode=at_cathode)
    @test all(isfinite, mesh.efield)
    @test all(isfinite, expected)
    # Float32 includes cancellation in the primitive and corner differences;
    # compare each component so a boosted transverse field cannot hide Ez errors.
    tolerance = eltype(mesh.rho) == Float32 ? 3e-4 : 2e-10
    for component in 1:3
        @test mesh.efield[:, :, :, component] ≈ expected[:, :, :, component] rtol=tolerance
    end
end

function run_solver_reference_tests()
    @testset "Independent integrated-field reference" begin
        @testset "$T, grid=$grid_size, gamma=$gamma, cathode=$at_cathode" for
            T in (Float32, Float64), grid_size in ((4, 5, 6), (7, 4, 5)),
            gamma in (1.0, 10.0, 100.0), at_cathode in (false, true)
            mesh = reference_test_mesh(T, grid_size, gamma)
            SpaceCharge._solve_mesh!(mesh; at_cathode=at_cathode)
            test_field_against_reference(mesh; at_cathode=at_cathode)
        end
    end
end

# Enable this test function once the public cache_green option is available.
function run_solver_cache_reference_tests()
    @testset "Green cache invalidation against independent reference" begin
        @testset "$T, cathode=$at_cathode" for T in (Float32, Float64), at_cathode in (false, true)
            mesh = reference_test_mesh(T, (5, 6, 4), 10.0)
            for update in (:initial, :reuse, :charge_values, :charge_array, :delta,
                           :gamma, :cathode_offset, :uncached, :recached, :zero)
                @testset "$update" begin
                    if update == :charge_values
                        mesh.rho .*= T(-0.7)
                        mesh.rho[2, 3, 1] += T(5e-12)
                    elseif update == :charge_array
                        mesh.rho = reverse(copy(mesh.rho); dims=2)
                    elseif update == :delta
                        mesh.delta = mesh.delta .* T.((1.03, 0.91, 1.07))
                        mesh.max_bounds = mesh.min_bounds .+ mesh.delta .* (mesh.grid_size .- 1)
                    elseif update == :gamma
                        mesh.gamma = T(11.3)
                    elseif update == :cathode_offset
                        translation = (zero(T), zero(T), T(0.17) * mesh.delta[3])
                        mesh.min_bounds = mesh.min_bounds .+ translation
                        mesh.max_bounds = mesh.max_bounds .+ translation
                    elseif update == :zero
                        fill!(mesh.rho, zero(T))
                    end
                    SpaceCharge._solve_mesh!(mesh; at_cathode=at_cathode, cache_green=update != :uncached)
                    test_field_against_reference(mesh; at_cathode=at_cathode)
                end
            end
        end
    end
end
