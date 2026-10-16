# Internal FFT stage; public solve! entry points have already validated gamma.
function _solve_mesh!(mesh::FFTIGF; gamma=mesh.gamma,
                      at_cathode::Bool=mesh.options.at_cathode,
                      fft_flags=mesh.options.fft_flags,
                      cache_green::Bool=mesh.options.cache_green,
                      pruned_fft::Bool=mesh.options.pruned_fft)
    mesh.gamma = gamma
    w = _get_workspace(mesh; fft_flags, pruned_fft)
    # Real charge initializes the direct field or scalar potential, as selected.
    T = eltype(mesh.rho)
    _solve_freespace!(mesh, w, (zero(T), zero(T), zero(T)); cache_green)

    if at_cathode
        # Image charge at -z: reverse and negate directly into the FFT input.
        # Add the image field or potential to the corresponding real-charge result.
        offset_z = 2 * mesh.min_bounds[3] + (mesh.max_bounds[3] - mesh.min_bounds[3])
        offset = (zero(T), zero(T), offset_z)
        _solve_freespace!(mesh, w, offset; cache_green, image=true)
    end
    if mesh.options.field_method == :potential_fd
        # Differentiate once, after combining real and image potentials.
        _field_from_potential!(mesh, w)
    end
    return nothing
end

function _green_spectrum!(dest, mesh, w, comp, offset)
    if comp == 0
        _potential_green!(mesh, w, offset)
    elseif all(iszero, offset)
        _symmetric_green!(w.pad, mesh, comp, w)
    else
        get_green_function!(w.pad, mesh.delta, mesh.gamma, comp;
                            offset, temp=w.result)
    end
    mul!(dest, w.forward, w.pad)
    return dest
end

function _cached_spectra!(mesh, w, offset)
    cache = all(iszero, offset) ? w.free_cache : w.image_cache
    key = (mesh.delta, mesh.gamma, offset)
    if cache.key != key
        # Invalidate before rebuilding: a failed transform must never leave a
        # partially overwritten cache that could be mistaken for a valid one.
        cache.key = nothing
        if cache.spectra === nothing
            cache.spectra = map(_ -> similar(w.product), w.components)
        end
        for (i, comp) in enumerate(w.components)
            _green_spectrum!(cache.spectra[i], mesh, w, comp, offset)
        end
        cache.key = key
    end
    return cache.spectra
end

# Convolve real or image charge with the kernels selected by w.components:
# direct field components (1, 2, 3) or scalar potential (0,).
# This function barrier specializes the plans and buffers stored in the mesh.
function _solve_freespace!(mesh::FFTIGF{T}, w, offset; cache_green=true, image=false) where T
    _charge_transform!(mesh, w, w.pruned, image)
    greens = cache_green ? _cached_spectra!(mesh, w, offset) : nothing
    # EPS_0 is in 1/(eV*m); EPS_0 * E_CHARGE is the SI permittivity in F/m.
    factor = T(inv(4pi * EPS_0 * E_CHARGE)) / T(length(w.pad))
    for (i, comp) in enumerate(w.components)
        if greens === nothing
            _green_spectrum!(w.product, mesh, w, comp, offset)
            w.product .*= w.rhohat
        else
            @. w.product = greens[i] * w.rhohat
        end
        if comp == 0
            # Real charge initializes potential; image charge accumulates into it.
            potential = _convolution_inverse!(mesh, w, w.pruned)
            if image
                @. w.potential += factor * potential
            else
                @. w.potential = factor * potential
            end
        else
            _field_inverse!(mesh, w, w.pruned, comp, factor, image)
        end
    end
    return nothing
end
