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
    return _potential_atan_term(x, y, z, r) + _potential_atan_term(y, x, z, r) +
           _potential_atan_term(z, x, y, r) + _potential_log_term(x, y, z, r) +
           _potential_log_term(y, x, z, r) + _potential_log_term(z, x, y, r)
end

# Continuous limits of the same primitive on coordinate planes. Shifted image
# cell corners can lie exactly on them; avoid 0*log(0) and 0*atan(0/0).
@inline _potential_atan_term(a, b, c, r) =
    iszero(a) ? zero(r) : -a^2 / 2 * atan(b * c / (a * r))
# Rationalizing r+a for negative a avoids cancellation to log(0) when the
# transverse coordinates are small. This is the same analytic primitive.
@inline _potential_log_term(a, b, c, r) =
    iszero(b) || iszero(c) ? zero(r) : b * c * log(a < 0 ? (b*b + c*c) / (r-a) : a+r)

"""
    field_green_function(x, y, z)

The indefinite integral:
∫∫∫ x/r^3 dx dy dz = x*atan((y*z)/(r*x)) -z*log(r+y) + y*log((r-z)/(r+z))/2

This corresponds to the electric field component Ex.
Other components can be computed by permuting the arguments.

Assumes x, y, z are all non-zero (no singularity handling) to avoid branching on GPU.
"""
@inline function field_green_function(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    return x * atan((y * z) / (r * x)) - z * log(r + y) + y * log((r - z) / (r + z)) / 2
end


function get_green_function!(
    cgrn::A,
    delta::NTuple{3, T},
    gamma::T,
    icomp::Int;
    offset::NTuple{3, T} = (zero(T), zero(T), zero(T)),
    temp::A
) where {T<:AbstractFloat, A<:AbstractArray}
    isize, jsize, ksize = size(cgrn)

    # Populate cgrn with pointwise Green's function
    backend = get_backend(cgrn)
    kernel! = get_green_kernel!(backend)
    kernel!(cgrn, delta, gamma, icomp, offset, ndrange = size(cgrn))

    # Apply 8-point differencing to compute integrated Green's function
    # Use a temporary array to avoid read-write race condition
    differencing_kernel! = apply_8point_differencing!(backend)
    differencing_kernel!(temp, cgrn, ndrange = (isize-1, jsize-1, ksize-1))

    # Copy differenced values back to cgrn
    cgrn_view = @view cgrn[1:isize-1, 1:jsize-1, 1:ksize-1]
    temp_view = @view temp[1:isize-1, 1:jsize-1, 1:ksize-1]
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

    umin = (1 - isize) * dx / 2 + offset[1]
    vmin = (1 - jsize) * dy / 2 + offset[2]
    wmin = (1 - ksize) * dz / 2 + offset[3] * gamma

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

    cgrn[i, j, k] = gval
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

@kernel function _parity_corners!(corners, delta, gamma, comp)
    i, j, k = @index(Global, NTuple)
    dx, dy, dz = delta[1], delta[2], delta[3] * gamma
    T = typeof(dx)
    half = one(dx) / 2
    # n+1 corners at -delta/2 through (n-1/2)*delta, enclosing n cells
    # centered on nonnegative offsets 0, delta, ..., (n-1)*delta.
    x = (T(i - 1) - half) * dx
    y = (T(j - 1) - half) * dy
    z = (T(k - 1) - half) * dz
    factor = (comp == 1 || comp == 2 ? gamma : one(dx)) / (dx * dy * dz)
    value = if comp == 1
        field_green_function(x, y, z)
    elseif comp == 2
        field_green_function(y, z, x)
    else
        field_green_function(z, x, y)
    end
    corners[i, j, k] = value * factor
end

@kernel function _parity_reflect!(out, octant, n, comp)
    i, j, k = @index(Global, NTuple)
    # In the source layout, zero displacement is at (nx, ny, nz).
    # The final plane in any dimension is padding; see parity_green! below.
    if i == 2n[1] || j == 2n[2] || k == 2n[3]
        out[i, j, k] = zero(eltype(out))
    else
        a, b, c = i - n[1], j - n[2], k - n[3]
        d = comp == 1 ? a : (comp == 2 ? b : c)
        value = octant[abs(a)+1, abs(b)+1, abs(c)+1]
        # Enforce the exact physical symmetry, removing cancellation residue
        # at the own-component center plane. Other axes have even parity.
        out[i, j, k] = d == 0 ? zero(value) : (d < 0 ? -value : value)
    end
end

"""
    _symmetric_green!(out, mesh, comp, work)

Fill a real or complex doubled-grid kernel for component `comp` (1:3), using
only (nx+1)*(ny+1)*(nz+1) antiderivative evaluations and reflection of the nx*ny*nz
integrated positive octant. Compatible with the source kernel's spatial layout.
This changes floating-point roundoff (including explicitly zero own-component
center planes), but does not change the integrated-E discretization.

The final doubled-grid plane in each dimension is explicitly zero, whereas the
source leaves undifferenced antiderivative values there. These entries do not
contribute to the source solver's retained convolution: rho occupies indices
1:n, output indices n:2n-1, hence consumed kernel indices are 1:2n-1 in each
dimension. This equivalence applies to that crop only; the entire FFT output is
not expected to match. Nonzero offsets violate the reflection symmetry and are
not supported by this interface.

Kernel launches use the array backend. Callers must synchronize after the whole
operation when timing GPU execution, as with other KernelAbstractions kernels.
"""
function _symmetric_green!(out, mesh, comp::Integer, work)
    n = mesh.grid_size
    1 <= comp <= 3 || throw(ArgumentError("comp must be 1, 2, or 3"))
    size(out) == (2n[1], 2n[2], 2n[3]) || throw(DimensionMismatch("out must be doubled mesh dimensions"))
    size(work.corners) == (n[1]+1, n[2]+1, n[3]+1) || throw(DimensionMismatch("corner workspace does not match mesh"))
    size(work.octant) == n || throw(DimensionMismatch("octant workspace does not match mesh"))
    backend = get_backend(out)
    _parity_corners!(backend)(work.corners, mesh.delta, mesh.gamma, Int(comp);
                             ndrange = size(work.corners))
    apply_8point_differencing!(backend)(work.octant, work.corners; ndrange = n)
    _parity_reflect!(backend)(out, work.octant, n, Int(comp); ndrange = size(out))
    return out
end
