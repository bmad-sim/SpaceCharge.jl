using KernelAbstractions

"""
    potential_green_function(x, y, z)

The indefinite integral:
∫∫∫ 1/r dx dy dz = 
        -z^2*atan(x*y/(z*r))/2 - y^2*atan(x*z/(y*r))/2 -x^2*atan(y*z/(x*r))/2 
        +y*z*log(x+r) + x*z*log(y+r) + x*y*log(z+r)
        
This corresponds to the scalar potential.
"""
@inline function potential_green_function(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    if r == zero(x)
        return zero(x)
    end
    return -0.5 * z^2 * atan(x * y / (z * r)) - 0.5 * y^2 * atan(x * z / (y * r)) -
           0.5 * x^2 * atan(y * z / (x * r)) + y * z * log(x + r) +
           x * z * log(y + r) + x * y * log(z + r)
end

"""
    field_green_function(x, y, z)

The indefinite integral:
∫∫∫ x/r^3 dx dy dz = x*atan((y*z)/(r*x)) -z*log(r+y) + y*log((r-z)/(r+z))/2

This corresponds to the electric field component Ex.
Other components can be computed by permuting the arguments.
"""
@inline function field_green_function(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)

    if r == zero(x)
        return zero(x)
    end

    # Term 1: x * atan(y*z / (r*x))
    # When x == 0, this is 0 * atan(...) = 0
    term1 = if x == zero(x)
        zero(x)
    else
        x * atan((y * z) / (r * x))
    end

    # Term 2: -z * log(r + y)
    # When r + y == 0 (i.e., y = -r), z must also be 0 (since r = |x|), so z * log(...) = 0
    term2 = if (r + y) <= zero(x)
        zero(x)
    else
        -z * log(r + y)
    end

    # Term 3: y * log((r - z) / (r + z)) / 2
    # When r + z == 0 (i.e., z = -r), y must also be 0, so y * log(...) = 0
    # When r - z == 0 (i.e., z = r), y must also be 0, so y * log(...) = 0
    term3 = if (r + z) <= zero(x) || (r - z) <= zero(x)
        zero(x)
    else
        y * log((r - z) / (r + z)) / 2
    end

    return term1 + term2 + term3
end


function get_green_function!(
    cgrn::A,
    delta::NTuple{3, T},
    gamma::T,
    icomp::Int;
    offset::NTuple{3, T} = (zero(T), zero(T), zero(T)),
    temp::Union{Nothing, A} = nothing
) where {T<:AbstractFloat, A<:AbstractArray}
    isize, jsize, ksize = size(cgrn)

    # Populate cgrn with pointwise Green's function
    backend = get_backend(cgrn)
    kernel! = get_green_kernel!(backend)
    kernel!(cgrn, delta, gamma, icomp, offset, ndrange = size(cgrn))

    # Apply 8-point differencing to compute integrated Green's function
    # Use a temporary array to avoid read-write race condition
    diff_temp = temp === nothing ? similar(cgrn) : temp
    backend = get_backend(cgrn)
    differencing_kernel! = apply_8point_differencing!(backend)
    differencing_kernel!(diff_temp, cgrn, ndrange = (isize-1, jsize-1, ksize-1))

    # Copy differenced values back to cgrn
    cgrn_view = @view cgrn[1:isize-1, 1:jsize-1, 1:ksize-1]
    temp_view = @view diff_temp[1:isize-1, 1:jsize-1, 1:ksize-1]
    cgrn_view .= temp_view
end

@kernel function get_green_kernel!(cgrn, delta, gamma, icomp, offset)
    i, j, k = @index(Global, NTuple)
    isize, jsize, ksize = @ndrange

    dx, dy, dz = delta[1], delta[2], delta[3] * gamma

    # Apply normalization factor
    factor = if (icomp == 1) || (icomp == 2)
        gamma / (dx * dy * dz)  # transverse fields are enhanced by gamma
    else
        one(dx) / (dx * dy * dz)
    end

    umin = (0.5 - isize / 2) * dx + offset[1]
    vmin = (0.5 - jsize / 2) * dy + offset[2]
    wmin = (0.5 - ksize / 2) * dz + offset[3] * gamma

    u = (i - 1) * dx + umin
    v = (j - 1) * dy + vmin
    w = (k - 1) * dz + wmin

    gval = if icomp == 1
        field_green_function(u, v, w) * factor
    elseif icomp == 2
        field_green_function(v, w, u) * factor
    elseif icomp == 3
        field_green_function(w, u, v) * factor
    else
        zero(dx)
    end

    cgrn[i, j, k] = Complex{typeof(gval)}(gval, zero(gval))
end

@kernel function apply_8point_differencing!(out, cgrn)
    i, j, k = @index(Global, NTuple)

    # Compute 8-point finite difference stencil
    # Writes to separate output array to avoid read-write race condition
    out[i, j, k] = cgrn[i+1, j+1, k+1] - cgrn[i, j+1, k+1] -
                   cgrn[i+1, j, k+1] - cgrn[i+1, j+1, k] -
                   cgrn[i, j, k] + cgrn[i, j, k+1] +
                   cgrn[i, j+1, k] + cgrn[i+1, j, k]
end
