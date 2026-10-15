using AbstractFFTs

# Each mesh owns at most one free-space and one shifted-image spectrum set.
# The exact key includes every physical parameter used by the kernel builder;
# array backend, precision, and shape are fixed by the owning FFT workspace.
mutable struct GreenCache{T,A}
    key::Union{Nothing,Tuple{NTuple{3,T},T,NTuple{3,T}}}
    spectra::Union{Nothing,NTuple{3,A}}
end

mutable struct Mesh3D{T <: AbstractFloat, A <: AbstractArray{T}, B <: AbstractArray{T}}
    # Grid dimensions
    grid_size::NTuple{3, Int}
    # Physical domain
    min_bounds::NTuple{3, T}
    max_bounds::NTuple{3, T}
    delta::NTuple{3, T}
    # Physics parameters
    gamma::T
    # Data arrays (CPU or GPU)
    rho::A
    efield::B
    # Optimization workspace (lazy initialization)
    _workspace::Union{Nothing, NamedTuple}
    # CPU deposition uses independent bounded scratch; it needs no FFT plans.
    _deposit_workspace::Union{Nothing,Vector{Array{T,3}}}
end

"""
    _get_workspace(mesh; fft_flags=FFTW.MEASURE)

Lazily allocate real FFT buffers and plans. CPU planning effort is configurable;
GPU plans use the backend default. Replacing mesh arrays rebuilds the workspace.
"""
function _get_workspace(mesh::Mesh3D{T}; fft_flags=FFTW.MEASURE, pruned_fft::Bool=true) where T
    n = mesh.grid_size
    size(mesh.rho) == n || throw(DimensionMismatch("rho does not match grid_size"))
    size(mesh.efield) == (n..., 3) || throw(DimensionMismatch("efield does not match grid_size"))
    fft_threads = get_backend(mesh.rho) isa CPU ? FFTW.get_num_threads() : 0
    use_pruned = pruned_fft && get_backend(mesh.rho) isa CPU
    w = mesh._workspace
    if w === nothing || w.owner !== mesh.rho || w.field_owner !== mesh.efield ||
       size(w.pad) != 2 .* n || w.flags != fft_flags || w.fft_threads != fft_threads ||
       (w.pruned !== nothing) != use_pruned
        dims = 2 .* n
        pad = similar(mesh.rho, dims)
        result = similar(pad)
        rhohat = similar(mesh.rho, Complex{T}, (n[1]+1, dims[2], dims[3]))
        product = similar(rhohat)
        corners = similar(mesh.rho, n .+ 1)
        octant = similar(mesh.rho)
        free_cache = GreenCache{T,typeof(product)}(nothing, nothing)
        image_cache = GreenCache{T,typeof(product)}(nothing, nothing)
        if get_backend(mesh.rho) isa CPU
            forward = plan_rfft(pad; flags=fft_flags)
            backward = plan_brfft(product, dims[1]; flags=fft_flags)
        else
            forward = plan_rfft(pad)
            backward = plan_brfft(product, dims[1])
        end
        pruned = use_pruned ? _pruned_plans(n, rhohat, product; flags=fft_flags) : nothing
        mesh._workspace = (;owner=mesh.rho, field_owner=mesh.efield,
                            flags=fft_flags, fft_threads, pad, result, rhohat, product, forward, backward,
                            corners, octant, free_cache, image_cache, pruned)
    end
    return mesh._workspace
end
# Internal fixed-shape storage. Physical geometry is owned by FFTIGF.
function Mesh3D(n, lo::NTuple{3,T}, hi::NTuple{3,T}, backend) where T
    rho = KernelAbstractions.zeros(backend, T, n...)
    efield = KernelAbstractions.zeros(backend, T, n..., 3)
    Mesh3D{T,typeof(rho),typeof(efield)}(n, lo, hi, (hi .- lo) ./ (n .- 1),
                                      one(T), rho, efield, nothing, nothing)
end

@inline _inside(x, y, z, lo, hi) =
    (lo[1] <= x) & (x <= hi[1]) & (lo[2] <= y) & (y <= hi[2]) &
    (lo[3] <= z) & (z <= hi[3])

# Closed upper faces belong to the last cell, with weight 1 on its upper node.
# Clamp only after physical-domain validation, including rounding of normalization.
@inline function _cell(x, lo, inv_delta, n)
    u = (x - lo) * inv_delta
    i = floor(Int, u)
    # Domain validation already guarantees u >= 0. Only the upper cell needs
    # special treatment; avoid clamping every interior coordinate twice.
    return i < n - 1 ? (i, u - i) : (n - 2, one(x))
end
