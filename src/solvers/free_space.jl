using KernelAbstractions
using AbstractFFTs
using FFTW

"""
    solve!(mesh::Mesh3D; at_cathode::Bool = false)

Solves the space charge problem for free-space boundary conditions.

# Arguments
- `mesh`: A `Mesh3D` object containing the charge density and where the fields will be stored.
- `at_cathode`: A boolean indicating whether to model a cathode at z=0.
"""
function solve!(mesh::Mesh3D; at_cathode::Bool = false)
    # Real charge field
    T = eltype(mesh.rho)
    solve_freespace!(mesh, offset = (zero(T), zero(T), zero(T)))

    if at_cathode
        # Image charge field
        image_mesh = deepcopy(mesh)
        image_mesh.rho .= -image_mesh.rho[:, :, end:-1:1]

        # Image charge offset for cathode at z=0: 
        # Real charge at z, image charge at -z, distance = 2z
        offset_z = 2 * mesh.min_bounds[3] + (mesh.max_bounds[3] - mesh.min_bounds[3])
        offset = (zero(T), zero(T), offset_z)

        solve_freespace!(image_mesh, offset = offset)

        # Superposition of fields
        mesh.efield .+= image_mesh.efield
    end
end

"""
    solve_freespace!(mesh::Mesh3D{T, A, B}; offset::NTuple{3, T} = (zero(T), zero(T), zero(T))) where {T<:AbstractFloat, A<:AbstractArray, B<:AbstractArray}

Optimized free-space solver for computing electric field from charge density.
Uses pre-allocated workspace arrays, cached in-place FFT plans, and CPU multi-threading for 
optimal performance on both CPU and GPU.
"""
function solve_freespace!(mesh::Mesh3D{T, A, B}; offset::NTuple{3, T} = (zero(T), zero(T), zero(T))) where {T<:AbstractFloat, A<:AbstractArray, B<:AbstractArray}
    nx, ny, nz = mesh.grid_size

    # Set FFTW to use all available threads for optimal CPU performance
    FFTW.set_num_threads(Threads.nthreads())

    # Get pre-allocated workspace (lazy initialization)
    workspace = _get_workspace(mesh)
    crho = workspace.crho
    cgrn = workspace.cgrn
    temp_result = workspace.temp_result
    fft_plan_inplace = workspace.fft_plan_inplace
    ifft_plan_inplace = workspace.ifft_plan_inplace

    # Clear and setup charge density array
    fill!(crho, 0.0)
    crho[1:nx, 1:ny, 1:nz] .= mesh.rho

    # In-place FFT of charge density
    fft_plan_inplace * crho

    # Normalization factor: 1/(4 pi eps0)
    factr = FPEI

    for icomp = 1:3
        # Get Green's function (reuse cgrn array)
        get_green_function!(
            cgrn,
            mesh.delta,
            mesh.gamma,
            icomp,
            offset = offset,
        )

        # In-place FFT of Green's function
        fft_plan_inplace * cgrn

        # Convolution (vectorized operation)
        @. temp_result = crho * cgrn

        # In-place inverse FFT
        ifft_plan_inplace * temp_result

        # Extract field using optimized slicing
        ishift, jshift, kshift = nx - 1, ny - 1, nz - 1
        @views @. mesh.efield[:, :, :, icomp] = factr * real(temp_result[1+ishift:nx+ishift, 1+jshift:ny+jshift, 1+kshift:nz+kshift])
    end
end
