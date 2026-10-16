# Each solver owns at most one free-space and one shifted-image spectrum set.
# The exact key includes every physical parameter used by the kernel builder;
# array backend, precision, and shape are fixed by the owning FFT workspace.
mutable struct GreenCache{T,A,N}
    key::Union{Nothing,Tuple{NTuple{3,T},T,NTuple{3,T}}}
    spectra::Union{Nothing,NTuple{N,A}}
end

"""
    _get_workspace(mesh; fft_flags=FFTW.MEASURE)

Lazily allocate real FFT buffers and plans. CPU planning effort is configurable;
GPU plans use the backend default. Replacing mesh arrays rebuilds the workspace.
"""
function _get_workspace(mesh::FFTIGF{T}; fft_flags=FFTW.MEASURE, pruned_fft::Bool=true,
                        at_cathode::Bool=mesh.options.at_cathode) where T
    n = mesh.grid_size
    size(mesh.rho) == n || throw(DimensionMismatch("rho does not match grid_size"))
    size(mesh.efield) == (n..., 3) || throw(DimensionMismatch("efield does not match grid_size"))
    fft_threads = get_backend(mesh.rho) isa CPU ? FFTW.get_num_threads() : 0
    use_pruned = pruned_fft && get_backend(mesh.rho) isa CPU
    potential_fd = mesh.options.field_method == :potential_fd
    # 0 selects the scalar potential kernel; 1:3 select direct field components.
    components = potential_fd ? (0,) : (1, 2, 3)
    dims = 2 .* n
    w = mesh._workspace
    if w === nothing || w.owner !== mesh.rho || w.field_owner !== mesh.efield ||
       size(w.pad) != dims || w.flags != fft_flags || w.fft_threads != fft_threads ||
       (w.pruned !== nothing) != use_pruned || w.components != components
        pad = similar(mesh.rho, dims)
        result = similar(pad)
        rhohat = similar(mesh.rho, Complex{T}, (n[1]+1, dims[2], dims[3]))
        product = similar(rhohat)
        corners = similar(mesh.rho, n .+ 1)
        octant = similar(mesh.rho)
        potential = potential_fd && at_cathode ? similar(mesh.rho) : nothing
        free_cache = GreenCache{T,typeof(product),length(components)}(nothing, nothing)
        image_cache = GreenCache{T,typeof(product),length(components)}(nothing, nothing)
        if get_backend(mesh.rho) isa CPU
            forward = plan_rfft(pad; flags=fft_flags)
            backward = use_pruned ? nothing : plan_brfft(product, dims[1]; flags=fft_flags)
        else
            forward = plan_rfft(pad)
            backward = plan_brfft(product, dims[1])
        end
        pruned = use_pruned ? _pruned_plans(n, rhohat, product; flags=fft_flags) : nothing
        mesh._workspace = (;owner=mesh.rho, field_owner=mesh.efield,
                            flags=fft_flags, fft_threads, pad, result, rhohat, product, forward, backward,
                            corners, octant, potential, components,
                            free_cache, image_cache, pruned)
    elseif potential_fd && at_cathode && w.potential === nothing
        # Real potential must survive the image inverse. Adding this scratch
        # must not discard existing FFT plans or Green spectra.
        mesh._workspace = merge(w, (;potential=similar(mesh.rho)))
    end
    return mesh._workspace
end
