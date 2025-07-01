using KernelAbstractions
using StaticArrays

@inline function lafun2(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    if r == 0.0
        return 0.0
    end
    return -0.5 * z^2 * atan(x * y / (z * r)) - 0.5 * y^2 * atan(x * z / (y * r)) -
           0.5 * x^2 * atan(y * z / (x * r)) + y * z * log(x + r) +
           x * z * log(y + r) + x * y * log(z + r)
end

@inline function xlafun2(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    if r == 0.0
        return 0.0
    end
    return x * atan((y * z) / (r * x)) - z * log(r + y) + y * log((r - z) / (r + z)) / 2
end


function osc_get_cgrn_freespace!(
    cgrn,
    delta,
    gamma,
    icomp;
    offset = (0.0, 0.0, 0.0),
)
    dx, dy, dz = delta[1], delta[2], delta[3] * gamma
    isize, jsize, ksize = size(cgrn)

    factor = if (icomp == 1) || (icomp == 2)
        gamma / (dx * dy * dz)
    else
        1.0 / (dx * dy * dz)
    end

    umin = (0.5 - isize / 2) * dx + offset[1]
    vmin = (0.5 - jsize / 2) * dy + offset[2]
    wmin = (0.5 - ksize / 2) * dz + offset[3] * gamma

    # Temporary array to store point-wise Green's function values
    temp_g = similar(cgrn, Float64)

    # Use KernelAbstractions for parallel computation
    backend = get_backend(cgrn)
    kernel! = get_green_kernel!(backend)
    kernel!(temp_g, delta, gamma, icomp, offset, ndrange = size(cgrn))

    # 8-point differencing for integrated Green's function
    @views cgrn[1:isize-1, 1:jsize-1, 1:ksize-1] .=
        temp_g[2:isize, 2:jsize, 2:ksize] .-
        temp_g[1:isize-1, 2:jsize, 2:ksize] .-
        temp_g[2:isize, 1:jsize-1, 2:ksize] .-
        temp_g[2:isize, 2:jsize, 1:ksize-1] .-
        temp_g[1:isize-1, 1:jsize-1, 1:ksize-1] .+
        temp_g[1:isize-1, 1:jsize-1, 2:ksize] .+
        temp_g[1:isize-1, 2:jsize, 1:ksize-1] .+
        temp_g[2:isize, 1:jsize-1, 1:ksize-1]
end

@kernel function get_green_kernel!(temp_g, delta, gamma, icomp, offset)
    i, j, k = @index(Global, NTuple)
    isize, jsize, ksize = @ndrange

    dx, dy, dz = delta[1], delta[2], delta[3] * gamma

    umin = (0.5 - isize / 2) * dx + offset[1]
    vmin = (0.5 - jsize / 2) * dy + offset[2]
    wmin = (0.5 - ksize / 2) * dz + offset[3] * gamma

    u = (i - 1) * dx + umin
    v = (j - 1) * dy + vmin
    w = (k - 1) * dz + wmin

    gval = if icomp == 0
        lafun2(u, v, w)
    elseif icomp == 1
        xlafun2(u, v, w)
    elseif icomp == 2
        xlafun2(v, w, u)
    elseif icomp == 3
        xlafun2(w, u, v)
    else
        0.0
    end

    factor = if (icomp == 1) || (icomp == 2)
        gamma / (delta[1] * delta[2] * delta[3] * gamma)
    else
        1.0 / (delta[1] * delta[2] * delta[3] * gamma)
    end

    temp_g[i, j, k] = gval * factor
end
