using KernelAbstractions
using StaticArrays

# Placeholder for translated Fortran functions
# These will need to be replaced with actual translations from the Fortran code
# and carefully checked for numerical stability (e.g., handling division by zero, log of non-positive numbers).

"""
    lafun2(x, y, z)

Placeholder for the Fortran `lafun2` function.
This function is expected to compute a component of the Green's function.
"""
function lafun2(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    res = -0.5 * z^2 * atan(x * y / (z * r)) - 0.5 * y^2 * atan(x * z / (y * r)) - 0.5 * x^2 * atan(y * z / (x * r)) +
          y * z * log(x + r) + x * z * log(y + r) + x * y * log(z + r)
    return res
end

"""
    xlafun(x, y, z)

Translated from Fortran `xlafun` function.
"""
function xlafun(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    res = z - x * atan(z / x) + x * atan(y * z / (x * r))
    if y + r != 0
        res = res - z * log(y + r)
    end
    if z + r != 0
        res = res - y * log(z + r)
    end
    return res
end

"""
    ylafun(x, y, z)

Translated from Fortran `ylafun` function.
"""
function ylafun(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    res = x - y * atan(x / y) + y * atan(x * z / (y * r))
    if x + r != 0
        res = res - z * log(x + r)
    end
    if z + r != 0
        res = res - x * log(z + r)
    end
    return res
end

"""
    zlafun(x, y, z)

Translated from Fortran `zlafun` function.
"""
function zlafun(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    res = y - z * atan(y / z) + z * atan(x * y / (z * r))
    if x + r != 0
        res = res - y * log(x + r)
    end
    if y + r != 0
        res = res - x * log(y + r)
    end
    return res
end

"""
    igfcoulombfun(u, v, w, gam, dx, dy, dz)

Translated from Fortran `igfcoulombfun` function.
"""
function igfcoulombfun(u, v, w, gam, dx, dy, dz)
    x1 = u - 0.5 * dx
    x2 = u + 0.5 * dx
    y1 = v - 0.5 * dy
    y2 = v + 0.5 * dy
    z1 = (w - 0.5 * dz) * gam
    z2 = (w + 0.5 * dz) * gam

    res = lafun2(x2, y2, z2) - lafun2(x1, y2, z2) - lafun2(x2, y1, z2) - lafun2(x2, y2, z1) -
          lafun2(x1, y1, z1) + lafun2(x1, y1, z2) + lafun2(x1, y2, z1) + lafun2(x2, y1, z1)
    res = res / (dx * dy * dz * gam)
    return res
end

"""
    igfexfun(u, v, w, gam, dx, dy, dz)

Translated from Fortran `igfexfun` function.
"""
function igfexfun(u, v, w, gam, dx, dy, dz)
    x1 = u - 0.5 * dx
    x2 = u + 0.5 * dx
    y1 = v - 0.5 * dy
    y2 = v + 0.5 * dy
    z1 = (w - 0.5 * dz) * gam
    z2 = (w + 0.5 * dz) * gam

    if x1 < 0.0 && x2 > 0.0 && y1 < 0.0 && y2 > 0.0 && z1 < 0.0 && z2 > 0.0
        res = 0.0
    else
        res = xlafun(x2, y2, z2) - xlafun(x1, y2, z2) - xlafun(x2, y1, z2) - xlafun(x2, y2, z1) -
              xlafun(x1, y1, z1) + xlafun(x1, y1, z2) + xlafun(x1, y2, z1) + xlafun(x2, y1, z1)
    end
    res = res / (dx * dy * dz)
    return res
end

"""
    igfeyfun(u, v, w, gam, dx, dy, dz)

Translated from Fortran `igfeyfun` function.
"""
function igfeyfun(u, v, w, gam, dx, dy, dz)
    x1 = u - 0.5 * dx
    x2 = u + 0.5 * dx
    y1 = v - 0.5 * dy
    y2 = v + 0.5 * dy
    z1 = (w - 0.5 * dz) * gam
    z2 = (w + 0.5 * dz) * gam

    if x1 < 0.0 && x2 > 0.0 && y1 < 0.0 && y2 > 0.0 && z1 < 0.0 && z2 > 0.0
        res = 0.0
    else
        res = ylafun(x2, y2, z2) - ylafun(x1, y2, z2) - ylafun(x2, y1, z2) - ylafun(x2, y2, z1) -
              ylafun(x1, y1, z1) + ylafun(x1, y1, z2) + ylafun(x1, y2, z1) + ylafun(x2, y1, z1)
    end
    res = res / (dx * dy * dz)
    return res
end

"""
    igfezfun(u, v, w, gam, dx, dy, dz)

Translated from Fortran `igfezfun` function.
"""
function igfezfun(u, v, w, gam, dx, dy, dz)
    x1 = u - 0.5 * dx
    x2 = u + 0.5 * dx
    y1 = v - 0.5 * dy
    y2 = v + 0.5 * dy
    z1 = (w - 0.5 * dz) * gam
    z2 = (w + 0.5 * dz) * gam

    if x1 < 0.0 && x2 > 0.0 && y1 < 0.0 && y2 > 0.0 && z1 < 0.0 && z2 > 0.0
        res = 0.0
    else
        res = zlafun(x2, y2, z2) - zlafun(x1, y2, z2) - zlafun(x2, y1, z2) - zlafun(x2, y2, z1) -
              zlafun(x1, y1, z1) + zlafun(x1, y1, z2) + zlafun(x1, y2, z1) + zlafun(x2, y1, z1)
    end
    res = res / (dx * dy * dz * gam)
    return res
end

"""
    generate_igf_kernel!(green_function_array, grid_size, delta, component_idx)

A kernel to compute the Integrated Green's Function (IGF) for a given component.

This kernel will call the translated `lafun2`/`xlafun2` functions and perform
the 8-point differencing to get the integrated value for each cell.

# Arguments
- `green_function_array`: The array to store the computed Green's function.
- `grid_size`: A tuple (nx, ny, nz) representing the size of the grid.
- `delta`: A tuple (dx, dy, dz) representing the grid spacing.
- `component_idx`: An integer (1, 2, or 3) indicating which component (x, y, or z) of the field the Green's function is for.
"""
@kernel function generate_igf_kernel!(
    green_function_array,
    grid_size,
    delta,
    component_idx,
    gam,
)
    idx = @index(Global, NTuple)
    i, j, k = idx[1], idx[2], idx[3]

    # Calculate the physical coordinates of the cell center
    # These coordinates are relative to the center of the (0,0,0) cell in the padded grid.
    # The Green's function is typically centered at the origin.
    x_center = (i - (grid_size[1] / 2 + 1)) * delta[1]
    y_center = (j - (grid_size[2] / 2 + 1)) * delta[2]
    z_center = (k - (grid_size[3] / 2 + 1)) * delta[3]

    # Perform 8-point differencing for the integrated Green's function
    # This is a placeholder and needs to be verified against the Fortran logic.
    # The actual implementation will depend on how lafun2 and xlafun2 are used
    # to construct the integrated Green's function.

    # For now, a dummy calculation based on the component_idx
    if component_idx == 0 # Coulomb potential
        green_function_array[i, j, k] = igfcoulombfun(x_center, y_center, z_center, gam, delta[1], delta[2], delta[3])
    elseif component_idx == 1 # Ex component
        green_function_array[i, j, k] = igfexfun(x_center, y_center, z_center, gam, delta[1], delta[2], delta[3])
    elseif component_idx == 2 # Ey component
        green_function_array[i, j, k] = igfeyfun(x_center, y_center, z_center, gam, delta[1], delta[2], delta[3])
    elseif component_idx == 3 # Ez component
        green_function_array[i, j, k] = igfezfun(x_center, y_center, z_center, gam, delta[1], delta[2], delta[3])
    else
        green_function_array[i, j, k] = 0.0 # Should not happen with correct component_idx
    end
end

"""
    rfun(u, v, w, gam, a, b, hz, i_sign, j_sign)

Translated from Fortran `rfun` function.
"""
function rfun(u, v, w, gam, a, b, hz, i_sign::Int, j_sign::Int)
    res = 0.0
    ainv = 1.0 / a
    binv = 1.0 / b
    piainv = pi * ainv
    pibinv = pi * binv

    for m = 1:5
        for n = 1:5
            kapmn = sqrt((m * piainv)^2 + (n * pibinv)^2)
            zfun = (exp(-kapmn * abs(gam * w - hz)) - 2.0 * exp(-kapmn * abs(gam * w)) + exp(-kapmn * abs(gam * w + hz))) / (hz^2 * kapmn^2)
            if w == 0.0
                zfun += 2.0 / (hz * kapmn)
            end
            term = (i_sign^m) * (j_sign^n) * cos(m * u * piainv) * cos(n * v * pibinv) * zfun / kapmn
            res += term
        end
    end
    res = res * 2.0 * pi * ainv * binv
    return res
end