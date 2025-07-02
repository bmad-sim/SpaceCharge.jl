using KernelAbstractions
using AbstractFFTs
using FFTW
using LinearAlgebra # For mul!

"""
    struct RectangularPipe <: BoundaryCondition

Represents rectangular pipe boundary conditions.
"""
struct RectangularPipe <: BoundaryCondition end

"""
    rfun(x, y, z)

Placeholder for the Fortran `rfun` function, which computes the Green's function
for a rectangular pipe. This will involve a series summation.
"""
function rfun(u, v, w, gam, a, b, hz, i_val, j_val)
    # Translated from Fortran `rfun` function.
    # This function computes the Green's function for a rectangular pipe using a series summation.

    ainv = 1.0 / a
    binv = 1.0 / b
    piainv = pi * ainv
    pibinv = pi * binv

    res = 0.0
    for m in 1:5
        for n in 1:5
            kapmn = sqrt((m * piainv)^2 + (n * pibinv)^2)
            
            zfun = (exp(-kapmn * abs(gam * w - hz)) - 2.0 * exp(-kapmn * abs(gam * w)) + exp(-kapmn * abs(gam * w + hz))) / (hz^2 * kapmn^2)
            if w == 0.0
                zfun = zfun + 2.0 / (hz * kapmn)
            end
            term = (i_val^m) * (j_val^n) * cos(m * u * piainv) * cos(n * v * pibinv) * zfun / kapmn
            res = res + term
        end
    end
    res = res * 2.0 * pi * ainv * binv

    return res
end

# Helper function to calculate dimensions for cgrn arrays
function calculate_rectpipe_grn_dims(grid_size::NTuple{3, Int}, npad::NTuple{3, Int})
    nx, ny, nz = grid_size
    ipad, jpad, kpad = npad

    # Assuming 1-based indexing for Julia arrays
    rilo, rihi = 1, nx
    rjlo, rjhi = 1, ny
    rklo, rkhi = 1, nz

    cilo, cihi = 1, nx
    cjlo, cjhi = 1, ny
    cklo, ckhi = 1, nz

    # Calculate padded dimensions
    iperiod = 2 * nx + ipad
    jperiod = 2 * ny + jpad
    kperiod = 2 * nz + kpad

    # Calculate lower and upper bounds for each cgrn array
    # cgrn1
    g1ilo = cilo - rihi
    g1ihi = cihi - rilo
    g1jlo = cjlo - rjhi
    g1jhi = cjhi - rjlo
    g1klo = cklo - rkhi
    g1khi = ckhi - rklo

    # cgrn2
    g2ilo = cilo - rihi
    g2ihi = cihi - rilo
    g2jlo = cjlo + rjlo
    g2jhi = cjhi + rjhi
    g2klo = cklo - rkhi
    g2khi = ckhi - rklo

    # cgrn3
    g3ilo = cilo + rilo
    g3ihi = cihi + rihi
    g3jlo = cjlo - rjhi
    g3jhi = cjhi - rjlo
    g3klo = cklo - rkhi
    g3khi = ckhi - rklo

    # cgrn4
    g4ilo = cilo + rilo
    g4ihi = cihi + rihi
    g4jlo = cjlo + rjlo
    g4jhi = cjhi + rjhi
    g4klo = cklo - rkhi
    g4khi = ckhi - rklo

    # Return dimensions as tuples (lower_bound, upper_bound) for each axis
    return (
        (g1ilo, g1ihi + ipad, g1jlo, g1jhi + jpad, g1klo, g1khi + kpad),
        (g2ilo, g2ihi + ipad, g2jlo, g2jhi + jpad, g2klo, g2khi + kpad),
        (g3ilo, g3ihi + ipad, g3jlo, g3jhi + jpad, g3klo, g3khi + kpad),
        (g4ilo, g4ihi + ipad, g4jlo, g4jhi + jpad, g4klo, g4khi + kpad),
        (iperiod, jperiod, kperiod)
    )
end

"""
    solve!(mesh::Mesh3D, ::RectangularPipe)

Solves the space charge problem for rectangular pipe boundary conditions.

# Arguments
- `mesh`: A `Mesh3D` object containing the charge density and where the fields will be stored.
"""
function solve!(mesh::Mesh3D, ::RectangularPipe)
    T = eltype(mesh.rho)
    grid_size = mesh.grid_size
    delta = mesh.delta
    min_bounds = mesh.min_bounds
    gam = mesh.gamma

    # Define padding (npad in Fortran code)
    # For now, let's use 0 padding as in the Fortran osc_alloc_rectpipe_arrays example
    npad = (0, 0, 0)

    # Calculate dimensions for Green's function arrays
    (g1_dims, g2_dims, g3_dims, g4_dims, periods) = calculate_rectpipe_grn_dims(grid_size, npad)
    iperiod, jperiod, kperiod = periods

    # Allocate Green's function arrays
    cgrn1 = zeros(Complex{T}, iperiod, jperiod, kperiod)
    cgrn2 = zeros(Complex{T}, iperiod, jperiod, kperiod)
    cgrn3 = zeros(Complex{T}, iperiod, jperiod, kperiod)
    cgrn4 = zeros(Complex{T}, iperiod, jperiod, kperiod)

    # Calculate Green's functions
    osc_getgrnpipe_julia!(cgrn1, cgrn2, cgrn3, cgrn4, gam, mesh.max_bounds[1], mesh.max_bounds[2], delta, min_bounds, npad)

    # Pad the rho array
    padded_rho_size = (iperiod, jperiod, kperiod)
    padded_rho = zeros(Complex{T}, padded_rho_size)
    padded_rho[1:grid_size[1], 1:grid_size[2], 1:grid_size[3]] = mesh.rho

    # Perform forward FFT on padded_rho
    fft_plan_rho = plan_fft(padded_rho)
    fft_rho = fft_plan_rho * padded_rho

    # Perform convolution-correlation
    # The fftconvcorr3d_julia! function expects the output array 'con' to be the size of the original mesh.rho
    # and it will handle the unpadding internally.
    potential_field = zeros(T, grid_size)
    fftconvcorr3d_julia!(fft_rho, cgrn1, cgrn2, cgrn3, cgrn4, potential_field, iperiod, jperiod, kperiod)

    # Calculate electric field from potential (similar to free_space.jl)
    dx, dy, dz = delta
    for k in 1:grid_size[3]
        km1 = max(1, k - 1)
        kp1 = min(grid_size[3], k + 1)
        zfac = (km1 == k || kp1 == k) ? 2.0 : 1.0 # Fortran's merge equivalent

        for j in 1:grid_size[2]
            jm1 = max(1, j - 1)
            jp1 = min(grid_size[2], j + 1)
            yfac = (jm1 == j || jp1 == j) ? 2.0 : 1.0

            for i in 1:grid_size[1]
                im1 = max(1, i - 1)
                ip1 = min(grid_size[1], i + 1)
                xfac = (im1 == i || ip1 == i) ? 2.0 : 1.0

                mesh.efield[i, j, k, 1] = -(potential_field[ip1, j, k] - potential_field[im1, j, k]) / (2.0 * dx) * gam * xfac
                mesh.efield[i, j, k, 2] = -(potential_field[i, jp1, k] - potential_field[i, jm1, k]) / (2.0 * dy) * gam * yfac
                mesh.efield[i, j, k, 3] = -(potential_field[i, j, kp1] - potential_field[i, j, km1]) / (2.0 * dz) / gam * zfac
            end
        end
    end

    # Calculate magnetic field (similar to free_space.jl)
    beta0 = sqrt(1 - 1 / gam^2)
    clight = 299792458.0
    for k in 1:grid_size[3]
        for j in 1:grid_size[2]
            for i in 1:grid_size[1]
                mesh.bfield[i, j, k, 1] = -mesh.efield[i, j, k, 2] * beta0 / clight
                mesh.bfield[i, j, k, 2] = mesh.efield[i, j, k, 1] * beta0 / clight
                mesh.bfield[i, j, k, 3] = 0.0
            end
        end
    end

    return nothing
end

# Julia translation of osc_getgrnpipe
# This function will calculate the Green's functions for the rectangular pipe boundary conditions.
# It will need to be called before solve! for RectangularPipe.
function osc_getgrnpipe_julia!(
    cgrn1::AbstractArray{Complex{T}, 3},
    cgrn2::AbstractArray{Complex{T}, 3},
    cgrn3::AbstractArray{Complex{T}, 3},
    cgrn4::AbstractArray{Complex{T}, 3},
    gam::T,
    a::T,
    b::T,
    delta::NTuple{3, T},
    umin::NTuple{3, T},
    npad::NTuple{3, Int},
) where T

    hx, hy, hz = delta
    xmin, ymin, zmin = umin

    iperiod, jperiod, kperiod = size(cgrn1)
    ipad, jpad, kpad = npad

    # This puts the Green function where it's needed so the convolution ends up in the correct location in the array
    ishift = iperiod ÷ 2 - (ipad + 1) ÷ 2
    jshift = jperiod ÷ 2 - (jpad + 1) ÷ 2
    kshift = kperiod ÷ 2 - (kpad + 1) ÷ 2

    # 1: cgrn1
    for k_idx in 1:kperiod
        k_val = k_idx - 1 # 0-indexed for calculation
        kp = mod(k_val + kshift, kperiod) # Fortran's mod is different for negative numbers
        w = kp * hz

        for j_idx in 1:jperiod
            j_val = j_idx - 1
            jp = mod(j_val + jshift, jperiod)
            v = jp * hy

            for i_idx in 1:iperiod
                i_val = i_idx - 1
                ip = mod(i_val + ishift, iperiod)
                u = ip * hx

                cgrn1[i_idx, j_idx, k_idx] = rfun(u, v, w, gam, a, b, hz, 1, 1)
            end
        end
    end
    fft_plan = plan_fft!(cgrn1)
    mul!(cgrn1, fft_plan, cgrn1) # In-place FFT

    # 2: cgrn2
    for k_idx in 1:kperiod
        k_val = k_idx - 1
        kp = mod(k_val + kshift, kperiod)
        w = kp * hz

        for j_idx in 1:jperiod
            j_val = j_idx - 1
            # v = 2.0 * ymin + j_val * hy # Original Fortran line
            v = 2.0 * ymin + (j_idx - 1) * hy # Julia 1-indexed
            
            for i_idx in 1:iperiod
                i_val = i_idx - 1
                ip = mod(i_val + ishift, iperiod)
                u = ip * hx

                cgrn2[i_idx, j_idx, k_idx] = rfun(u, v, w, gam, a, b, hz, 1, -1)
            end
        end
    end
    fft_plan = plan_fft!(cgrn2)
    mul!(cgrn2, fft_plan, cgrn2) # In-place FFT

    # 3: cgrn3
    for k_idx in 1:kperiod
        k_val = k_idx - 1
        kp = mod(k_val + kshift, kperiod)
        w = kp * hz

        for j_idx in 1:jperiod
            j_val = j_idx - 1
            jp = mod(j_val + jshift, jperiod)
            v = jp * hy

            for i_idx in 1:iperiod
                i_val = i_idx - 1
                # u = 2.0 * xmin + i_val * hx # Original Fortran line
                u = 2.0 * xmin + (i_idx - 1) * hx # Julia 1-indexed

                cgrn3[i_idx, j_idx, k_idx] = rfun(u, v, w, gam, a, b, hz, -1, 1)
            end
        end
    end
    fft_plan = plan_fft!(cgrn3)
    mul!(cgrn3, fft_plan, cgrn3) # In-place FFT

    # 4: cgrn4
    for k_idx in 1:kperiod
        k_val = k_idx - 1
        kp = mod(k_val + kshift, kperiod)
        w = kp * hz

        for j_idx in 1:jperiod
            j_val = j_idx - 1
            # v = 2.0 * ymin + j_val * hy # Original Fortran line
            v = 2.0 * ymin + (j_idx - 1) * hy # Julia 1-indexed

            for i_idx in 1:iperiod
                i_val = i_idx - 1
                # u = 2.0 * xmin + i_val * hx # Original Fortran line
                u = 2.0 * xmin + (i_idx - 1) * hx # Julia 1-indexed

                cgrn4[i_idx, j_idx, k_idx] = rfun(u, v, w, gam, a, b, hz, -1, -1)
            end
        end
    end
    fft_plan = plan_fft!(cgrn4)
    mul!(cgrn4, fft_plan, cgrn4) # In-place FFT

    return nothing
end

# Julia translation of fftconvcorr3d
function fftconvcorr3d_julia!(
    crho::AbstractArray{Complex{T}, 3},
    qgrn1::AbstractArray{Complex{T}, 3},
    qgrn2::AbstractArray{Complex{T}, 3},
    qgrn3::AbstractArray{Complex{T}, 3},
    qgrn4::AbstractArray{Complex{T}, 3},
    con::AbstractArray{T, 3},
    iperiod::Int,
    jperiod::Int,
    kperiod::Int,
) where T

    fpei = T(299792458.0^2 * 1.0e-7) # 1/(4 pi eps0)
    qtot = T(1.0) # Assuming unit charge for now

    # Allocate temporary complex arrays
    ccon = similar(crho)
    ctmp = similar(crho)

    # 1st term:
    @. ccon = crho * qgrn1
    fft_plan = plan_ifft!(ccon)
    mul!(ccon, fft_plan, ccon) # In-place IFFT

    # 2nd term:
    @. ctmp = crho * qgrn2
    fft_plan = plan_ifft!(ctmp)
    mul!(ctmp, fft_plan, ctmp) # In-place IFFT
    @. ccon = ccon - ctmp

    # 3rd term:
    @. ctmp = crho * qgrn3
    fft_plan = plan_ifft!(ctmp)
    mul!(ctmp, fft_plan, ctmp) # In-place IFFT
    @. ccon = ccon - ctmp

    # 4th term:
    @. ctmp = crho * qgrn4
    fft_plan = plan_ifft!(ctmp)
    mul!(ctmp, fft_plan, ctmp) # In-place IFFT
    @. ccon = ccon + ctmp

    # Normalize:
    factr = T(1.0) / (T(iperiod) * T(jperiod) * T(kperiod)) * fpei * qtot

    # Store final result in original size (not double size) real array:
    # Assuming con is already correctly sized to the original mesh.rho size
    # and ccon is double-sized. We need to copy the relevant part.
    # This part needs careful indexing based on how the padding is handled.
    # For now, a direct copy of the unpadded region.
    con .= real.(ccon[1:size(con,1), 1:size(con,2), 1:size(con,3)]) .* factr

    return nothing
end
