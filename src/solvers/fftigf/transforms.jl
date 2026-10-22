# Exact separable CPU transforms. The charge occupies 1:n in each direction;
# SpaceCharge retains n:2n-1 of the convolution, NOT the first octant.
# Kernel transforms remain full: shifted image kernels need no symmetry here.
function _pruned_plans(n, rhohat::Array{Complex{T},3}, product; flags) where T
    nx, ny, nz = n
    xpad = zeros(T, 2nx, ny, nz)
    result = similar(xpad)
    # Explicit input/output layouts allow FFTW to write/read the strided views
    # directly. A public plan_rfft(xpad, 1) assumes a packed output instead.
    f1 = FFTW.rFFTWPlan{T,FFTW.FORWARD,false,3}(
        xpad, view(rhohat, :, 1:ny, 1:nz), 1, flags, FFTW.NO_TIMELIMIT)
    f2 = plan_fft!(view(rhohat, :, :, 1:nz), 2; flags)
    f3 = plan_fft!(rhohat, 3; flags)
    b3 = plan_bfft!(product, 3; flags)
    b2 = plan_bfft!(view(product, :, :, nz:2nz-1), 2; flags)
    b1 = FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,false,3}(
        view(product, :, ny:2ny-1, nz:2nz-1), result, 1, flags, FFTW.NO_TIMELIMIT)
    # MEASURE may overwrite its buffers. Charge padding is reset on every call.
    return (; xpad, result, f1, f2, f3, b3, b2, b1)
end

@kernel function _prepare_charge!(pad, @Const(rho), image)
    i, j, k = @index(Global, NTuple)
    nx, ny, nz = size(rho)
    # Write each input element once, including the image reversal and padding.
    pad[i, j, k] = if i <= nx && j <= ny && k <= nz
        image ? -rho[i, j, nz-k+1] : rho[i, j, k]
    else
        zero(eltype(pad))
    end
end

function _charge_transform!(mesh, w, ::Nothing, image)
    _prepare_charge!(get_backend(mesh.rho))(w.pad, mesh.rho, image;
                                           ndrange=size(w.pad))
    mul!(w.rhohat, w.forward, w.pad)
    return nothing
end

function _charge_transform!(mesh, w, p, image)
    nx, ny, nz = mesh.grid_size
    _prepare_charge!(get_backend(mesh.rho))(p.xpad, mesh.rho, image;
                                           ndrange=size(p.xpad))
    c = w.rhohat
    mul!(view(c, :, 1:ny, 1:nz), p.f1, p.xpad)
    @views fill!(c[:, ny+1:2ny, 1:nz], zero(eltype(c)))
    p.f2 * view(c, :, :, 1:nz)
    @views fill!(c[:, :, nz+1:2nz], zero(eltype(c)))
    p.f3 * c
    return nothing
end

function _convolution_inverse!(mesh, w, ::Nothing)
    nx, ny, nz = mesh.grid_size
    mul!(w.result, w.backward, w.product)
    return view(w.result, nx:2nx-1, ny:2ny-1, nz:2nz-1)
end

function _convolution_inverse!(mesh, w, p)
    nx, ny, nz = mesh.grid_size
    p.b3 * w.product
    p.b2 * view(w.product, :, :, nz:2nz-1)
    mul!(p.result, p.b1, view(w.product, :, ny:2ny-1, nz:2nz-1))
    # y/z have already been cropped; x is still the complete real inverse.
    return view(p.result, nx:2nx-1, :, :)
end

function _field_inverse!(mesh, w, p, comp, factor, image)
    _store_field!(mesh, _convolution_inverse!(mesh, w, p), comp, factor, image)
end

@kernel function _store_field_kernel!(out, @Const(field), comp, factor, image)
    I = @index(Global, Cartesian)
    @inbounds if image
        out[I, comp] += factor * field[I]
    else
        out[I, comp] = factor * field[I]
    end
end

function _store_field!(mesh, field, comp, factor, image)
    _store_field_kernel!(get_backend(mesh.efield))(mesh.efield, field, comp, factor, image;
                                                  ndrange=size(field))
    return nothing
end
