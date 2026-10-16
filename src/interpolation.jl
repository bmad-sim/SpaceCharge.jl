using KernelAbstractions

# Read-only gather into caller-owned outputs.
@kernel function interpolate_kernel!(
    Ex,
    Ey,
    Ez,
    efield,
    particles_x,
    particles_y,
    particles_z,
    min_bounds,
    inv_delta,
    active,
)
    i = @index(Global, Linear)

    if !_active(active, i)
        Ex[i] = Ey[i] = Ez[i] = zero(eltype(Ex))
    else
        @inbounds x, y, z = particles_x[i], particles_y[i], particles_z[i]
        ix, dx = _cell(x, min_bounds[1], inv_delta[1], size(efield, 1))
        iy, dy = _cell(y, min_bounds[2], inv_delta[2], size(efield, 2))
        iz, dz = _cell(z, min_bounds[3], inv_delta[3], size(efield, 3))

        @inbounds begin
            # --- 4. Calculate the trilinear weighting factors ---
            w_000 = (1 - dx) * (1 - dy) * (1 - dz)
            w_100 = dx * (1 - dy) * (1 - dz)
            w_010 = (1 - dx) * dy * (1 - dz)
            w_110 = dx * dy * (1 - dz)
            w_001 = (1 - dx) * (1 - dy) * dz
            w_101 = dx * (1 - dy) * dz
            w_011 = (1 - dx) * dy * dz
            w_111 = dx * dy * dz

            # --- 5. Interpolate the fields ---
            Ex[i] = (
                efield[ix + 1, iy + 1, iz + 1, 1] * w_000 +
                efield[ix + 2, iy + 1, iz + 1, 1] * w_100 +
                efield[ix + 1, iy + 2, iz + 1, 1] * w_010 +
                efield[ix + 2, iy + 2, iz + 1, 1] * w_110 +
                efield[ix + 1, iy + 1, iz + 2, 1] * w_001 +
                efield[ix + 2, iy + 1, iz + 2, 1] * w_101 +
                efield[ix + 1, iy + 2, iz + 2, 1] * w_011 +
                efield[ix + 2, iy + 2, iz + 2, 1] * w_111
            )
            Ey[i] = (
                efield[ix + 1, iy + 1, iz + 1, 2] * w_000 +
                efield[ix + 2, iy + 1, iz + 1, 2] * w_100 +
                efield[ix + 1, iy + 2, iz + 1, 2] * w_010 +
                efield[ix + 2, iy + 2, iz + 1, 2] * w_110 +
                efield[ix + 1, iy + 1, iz + 2, 2] * w_001 +
                efield[ix + 2, iy + 1, iz + 2, 2] * w_101 +
                efield[ix + 1, iy + 2, iz + 2, 2] * w_011 +
                efield[ix + 2, iy + 2, iz + 2, 2] * w_111
            )
            Ez[i] = (
                efield[ix + 1, iy + 1, iz + 1, 3] * w_000 +
                efield[ix + 2, iy + 1, iz + 1, 3] * w_100 +
                efield[ix + 1, iy + 2, iz + 1, 3] * w_010 +
                efield[ix + 2, iy + 2, iz + 1, 3] * w_110 +
                efield[ix + 1, iy + 1, iz + 2, 3] * w_001 +
                efield[ix + 2, iy + 1, iz + 2, 3] * w_101 +
                efield[ix + 1, iy + 2, iz + 2, 3] * w_011 +
                efield[ix + 2, iy + 2, iz + 2, 3] * w_111
            )
        end
    end
end
