using KernelAbstractions
using AbstractFFTs

"""
    abstract type BoundaryCondition end

Abstract type for defining different boundary conditions for space charge solvers.
"""
abstract type BoundaryCondition end

"""
    struct FreeSpace <: BoundaryCondition

Represents free-space boundary conditions.
"""
struct FreeSpace <: BoundaryCondition end

"""
    solve!(mesh::Mesh3D, ::FreeSpace; at_cathode::Bool = false)

Solves the space charge problem for free-space boundary conditions.

# Arguments
- `mesh`: A `Mesh3D` object containing the charge density and where the fields will be stored.
- `at_cathode`: A boolean indicating whether to model a cathode at z=0.
"""
function solve!(mesh::Mesh3D, ::FreeSpace; at_cathode::Bool = false)
    # Real charge field
    osc_freespace_solver2!(mesh, offset = (0.0, 0.0, 0.0))

    if at_cathode
        # Image charge field
        image_mesh = deepcopy(mesh)
        image_mesh.rho .= -image_mesh.rho[:, :, end:-1:1]

        offset_z =
            2 * mesh.min_bounds[3] + (mesh.max_bounds[3] - mesh.min_bounds[3])
        offset = (0.0, 0.0, offset_z)

        osc_freespace_solver2!(image_mesh, offset = offset)

        # Superposition of fields
        mesh.efield .+= image_mesh.efield

        # B-field from image charges has opposite sign
        beta = sqrt(1 - 1 / mesh.gamma^2)
        clight = 299792458.0
        mesh.bfield[:, :, :, 1] .-= (beta / clight) * image_mesh.efield[:, :, :, 2]
        mesh.bfield[:, :, :, 2] .+= (beta / clight) * image_mesh.efield[:, :, :, 1]
    end
end

"""
    osc_freespace_solver2!(mesh; offset = (0.0, 0.0, 0.0))

Julia implementation of the Fortran `osc_freespace_solver2` subroutine.
"""
function osc_freespace_solver2!(mesh; offset = (0.0, 0.0, 0.0))
    nx, ny, nz = mesh.grid_size
    nx2, ny2, nz2 = 2nx, 2ny, 2nz

    # Allocate complex arrays
    crho = similar(mesh.rho, ComplexF64, nx2, ny2, nz2)
    fill!(crho, 0.0)
    cgrn = similar(crho)

    # Copy rho to padded array
    crho[1:nx, 1:ny, 1:nz] .= mesh.rho

    # FFT of charge density
    fft_plan = plan_fft(crho)
    fft_rho = fft_plan * crho

    # Loop over phi, Ex, Ey, Ez
    for icomp = 0:3
        # Get Green's function
        osc_get_cgrn_freespace!(
            cgrn,
            mesh.delta,
            mesh.gamma,
            icomp,
            offset = offset,
        )

        # FFT of Green's function (no fftshift)
        fft_grn = fft_plan * cgrn

        # Convolution
        conv_fft = fft_rho .* fft_grn

        # Inverse FFT
        ifft_plan = plan_ifft(conv_fft)
        result = ifft_plan * conv_fft

        # Normalization factor
        dV = mesh.delta[1] * mesh.delta[2] * mesh.delta[3]
        factr = (299792458.0^2 * 1.00000000055e-7) / dV

        # Extract field/potential using manual slicing to match Fortran
        ishift, jshift, kshift = nx - 1, ny - 1, nz - 1
        if icomp == 0
            if isdefined(mesh, :phi)
                mesh.phi .= factr * real.(result[1+ishift:nx+ishift, 1+jshift:ny+jshift, 1+kshift:nz+kshift])
            end
        else
            mesh.efield[:, :, :, icomp] .= factr * real.(result[1+ishift:nx+ishift, 1+jshift:ny+jshift, 1+kshift:nz+kshift])
        end
    end

    # Calculate B-field
    if isdefined(mesh, :bfield)
        beta = sqrt(1 - 1 / mesh.gamma^2)
        clight = 299792458.0
        mesh.bfield[:, :, :, 1] = -(beta / clight) * mesh.efield[:, :, :, 2]
        mesh.bfield[:, :, :, 2] = (beta / clight) * mesh.efield[:, :, :, 1]
        mesh.bfield[:, :, :, 3] .= 0.0
    end
end



    


