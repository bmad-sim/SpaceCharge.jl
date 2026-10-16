# Exact cell-averaged rest-frame potential, with the same charge-per-node
# normalization and longitudinal stretching as the integrated-field method.
@kernel function _potential_corners!(corners, h, origin)
    i, j, k = @index(Global, NTuple)
    corners[i, j, k] = potential_green_function(
        (i-1)*h[1] + origin[1], (j-1)*h[2] + origin[2], (k-1)*h[3] + origin[3])
end

@kernel function _potential_reflect!(out, octant, n, factor)
    i, j, k = @index(Global, NTuple)
    # All axes have even parity. The final planes cannot enter the physical crop.
    out[i, j, k] = i == 2n[1] || j == 2n[2] || k == 2n[3] ? zero(eltype(out)) :
        factor * octant[abs(i-n[1])+1, abs(j-n[2])+1, abs(k-n[3])+1]
end

@kernel function _normalize_potential!(out, factor)
    i, j, k = @index(Global, NTuple)
    nx, ny, nz = @ndrange
    out[i, j, k] = i == nx || j == ny || k == nz ? zero(eltype(out)) :
        factor * out[i, j, k]
end

function _potential_green!(mesh, w, offset)
    h = (mesh.delta[1], mesh.delta[2], mesh.delta[3] * mesh.gamma)
    factor = inv(prod(h))
    backend = get_backend(mesh.rho)
    if all(iszero, offset)
        _potential_corners!(backend)(w.corners, h, .-h ./ 2; ndrange=size(w.corners))
        apply_8point_differencing!(backend)(w.octant, w.corners; ndrange=mesh.grid_size)
        _potential_reflect!(backend)(w.pad, w.octant, mesh.grid_size, factor; ndrange=size(w.pad))
    else
        shift = (offset[1], offset[2], offset[3] * mesh.gamma)
        origin = (1 .- size(w.pad)) .* h ./ 2 .+ shift
        _potential_corners!(backend)(w.result, h, origin; ndrange=size(w.result))
        apply_8point_differencing!(backend)(w.pad, w.result; ndrange=size(w.pad) .- 1)
        _normalize_potential!(backend)(w.pad, factor; ndrange=size(w.pad))
    end
    return nothing
end

@inline function _potential_difference(phi, I, d, factor)
    i, n = I[d], size(phi, d)
    step = CartesianIndex(ntuple(k -> k == d ? 1 : 0, 3))
    @inbounds if n == 2
        # Both endpoints use the secant; scale below includes the common 1/(2h).
        return i == 1 ? 2*(factor*phi[I]-factor*phi[I+step]) :
                        2*(factor*phi[I-step]-factor*phi[I])
    elseif i == 1
        return 3*(factor*phi[I]) - 4*(factor*phi[I+step]) + factor*phi[I+2step]
    elseif i == n
        return -(factor*phi[I-2step]) + 4*(factor*phi[I-step]) - 3*(factor*phi[I])
    else
        return factor*phi[I-step] - factor*phi[I+step]
    end
end

@kernel function _potential_field!(field, @Const(phi), scale, factor)
    I = @index(Global, Cartesian)
    # Centered inside, second-order one-sided at physical faces. The adjacent
    # doubled-grid values are not a valid physical halo and must not be read.
    # Normalize each value before subtraction, as in the staged calculation.
    field[I, 1] = scale[1] * _potential_difference(phi, I, 1, factor)
    field[I, 2] = scale[2] * _potential_difference(phi, I, 2, factor)
    field[I, 3] = scale[3] * _potential_difference(phi, I, 3, factor)
end

function _field_from_potential!(mesh, phi, factor=one(eltype(phi)))
    dx, dy, dz = mesh.delta
    gamma = mesh.gamma
    # Phi is the rest-frame potential in volts: E_perp=gamma*E'_perp,
    # E_z=E'_z, and d/dz'=1/gamma*d/dz for the lab-frame grid spacing.
    scale = (gamma / (2dx), gamma / (2dy), inv(2gamma * dz))
    _potential_field!(get_backend(mesh.rho))(mesh.efield, phi, scale, factor;
                                           ndrange=mesh.grid_size)
    return nothing
end
