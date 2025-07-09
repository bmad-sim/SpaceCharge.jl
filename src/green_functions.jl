using KernelAbstractions
using StaticArrays

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
    if r == 0.0
        return 0.0
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
    if r == 0.0
        return 0.0
    end
    return x * atan((y * z) / (r * x)) - z * log(r + y) + y * log((r - z) / (r + z)) / 2
end


function get_green_function!(
    cgrn::A,
    delta::NTuple{3, T},
    gamma::T,
    icomp::Int;
    offset::NTuple{3, T} = (0.0, 0.0, 0.0)
) where {T<:AbstractFloat, A<:AbstractArray}
    isize, jsize, ksize = size(cgrn)

    # Populate cgrn with pointwise Green's function
    backend = get_backend(cgrn)
    kernel! = get_green_kernel!(backend)
    kernel!(cgrn, delta, gamma, icomp, offset, ndrange = size(cgrn))

    # Apply 8-point differencing to compute integrated Green's function
    # Use optimized kernel to avoid memory allocations
    backend = get_backend(cgrn)
    differencing_kernel! = apply_8point_differencing!(backend)
    differencing_kernel!(cgrn, ndrange = (isize-1, jsize-1, ksize-1))
end

@kernel function get_green_kernel!(cgrn, delta, gamma, icomp, offset)
    i, j, k = @index(Global, NTuple)
    isize, jsize, ksize = @ndrange

    dx, dy, dz = delta[1], delta[2], delta[3] * gamma

    # Apply normalization factor
    factor = if (icomp == 1) || (icomp == 2)
        gamma / (dx * dy * dz)  # transverse fields are enhanced by gamma
    else
        1.0 / (dx * dy * dz)
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
        0.0
    end

    cgrn[i, j, k] = complex(gval, 0.0)
end

@kernel function apply_8point_differencing!(cgrn)
    i, j, k = @index(Global, NTuple)
    
    # Compute 8-point finite difference stencil in-place
    # This avoids all temporary array allocations
    cgrn[i, j, k] = cgrn[i+1, j+1, k+1] - cgrn[i, j+1, k+1] - 
                    cgrn[i+1, j, k+1] - cgrn[i+1, j+1, k] - 
                    cgrn[i, j, k] + cgrn[i, j, k+1] + 
                    cgrn[i, j+1, k] + cgrn[i+1, j, k]
end
