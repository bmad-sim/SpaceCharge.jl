# Internal FFT stage; public solve! entry points have already validated gamma.
function _solve_mesh!(mesh::FFTIGF; gamma=mesh.gamma,
                      at_cathode::Bool=mesh.options.at_cathode,
                      fft_flags=mesh.options.fft_flags,
                      cache_green::Bool=mesh.options.cache_green,
                      pruned_fft::Bool=mesh.options.pruned_fft)
    mesh.gamma = gamma
    w = _get_workspace(mesh; fft_flags, pruned_fft)
    # Real charge field
    T = eltype(mesh.rho)
    _solve_freespace!(mesh, w, (zero(T), zero(T), zero(T)); cache_green)

    if at_cathode
        # Image charge at -z: reverse and negate directly into the FFT input.
        # Accumulate each component into the real field without an image mesh.
        offset_z = 2 * mesh.min_bounds[3] + (mesh.max_bounds[3] - mesh.min_bounds[3])
        offset = (zero(T), zero(T), offset_z)
        _solve_freespace!(mesh, w, offset; cache_green, image=true)
    end
    return nothing
end

function _green_spectrum!(dest, mesh, w, comp, offset)
    if all(iszero, offset)
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
            cache.spectra = ntuple(_ -> similar(w.product), 3)
        end
        for comp in 1:3
            _green_spectrum!(cache.spectra[comp], mesh, w, comp, offset)
        end
        cache.key = key
    end
    return cache.spectra
end

# Function barrier specializes the plans and buffers stored in the mesh.
function _solve_freespace!(mesh::FFTIGF{T}, w, offset; cache_green=true, image=false) where T
    _charge_transform!(mesh, w, w.pruned, image)
    greens = cache_green ? _cached_spectra!(mesh, w, offset) : nothing
    factor = T(FPEI) / T(length(w.pad))
    for comp in 1:3
        if greens === nothing
            _green_spectrum!(w.product, mesh, w, comp, offset)
            w.product .*= w.rhohat
        else
            @. w.product = greens[comp] * w.rhohat
        end
        _field_inverse!(mesh, w, w.pruned, comp, factor, image)
    end
    return nothing
end
