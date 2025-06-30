using KernelAbstractions
using AbstractFFTs
using FFTW

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
    solve!(mesh::Mesh3D, ::FreeSpace)

Solves the space charge problem for free-space boundary conditions.

# Arguments
- `mesh`: A `Mesh3D` object containing the charge density and where the fields will be stored.
"""
function solve!(mesh::Mesh3D, ::FreeSpace; at_cathode::Bool=false)
    # Calculate fields for the real charge distribution
    _calculate_fields_from_rho!(
        mesh.rho,
        mesh.grid_size,
        mesh.min_bounds,
        mesh.max_bounds,
        mesh.delta,
        mesh.efield,
        mesh.bfield,
        mesh.gamma
    )

    if at_cathode
        # Create a temporary array for the image charge distribution
        image_rho = zeros(eltype(mesh.rho), mesh.grid_size)

        # Populate image_rho by z-flipping and negating the real rho
        # Assuming cathode is at z_min (mesh.min_bounds[3])
        # The image charge at z' for a real charge at z is z' = 2*z_cathode - z
        # For a grid, this means flipping the z-index.
        for k in 1:mesh.grid_size[3]
            # Calculate the corresponding image z-index
            # If z_cathode is at the first plane (k=1), then image of k=1 is k=1, image of k=2 is k=0 (out of bounds)
            # This needs to be carefully mapped. A simple flip is (grid_size[3] - k + 1)
            image_k = mesh.grid_size[3] - k + 1
            if 1 <= image_k <= mesh.grid_size[3]
                image_rho[:, :, image_k] = -mesh.rho[:, :, k]
            end
        end

        # Create temporary arrays for image fields
        image_efield = zeros(eltype(mesh.efield), size(mesh.efield))
        image_bfield = zeros(eltype(mesh.bfield), size(mesh.bfield))

        # Calculate fields for the image charge distribution
        _calculate_fields_from_rho!(
            image_rho,
            mesh.grid_size,
            mesh.min_bounds,
            mesh.max_bounds,
            mesh.delta,
            image_efield,
            image_bfield,
            mesh.gamma
        )

        # Add the real and image fields together
        # Be careful with signs for B-field (image charges move in opposite direction)
        mesh.efield .+= image_efield
        mesh.bfield .-= image_bfield # B-field from image charge is subtracted
    end

    return nothing
end

"""
    _calculate_fields_from_rho!(rho_array, grid_size, min_bounds, max_bounds, delta, efield_out, bfield_out, gam)

Helper function to calculate electric and magnetic fields from a given charge density array.
This function performs the core FFT-based convolution for field calculation.

# Arguments
- `rho_array`: The charge density array.
- `grid_size`: The size of the grid.
- `min_bounds`: Minimum bounds of the physical domain.
- `max_bounds`: Maximum bounds of the physical domain.
- `delta`: Grid spacing.
- `efield_out`: Output array for the electric field.
- `bfield_out`: Output array for the magnetic field.
- `gam`: Relativistic gamma factor.
"""
function _calculate_fields_from_rho!(
    rho_array,
    grid_size,
    min_bounds,
    max_bounds,
    delta,
    efield_out,
    bfield_out,
    gam,
)
    # --- 1. Pad the rho array ---
    padded_rho_size = 2 .* grid_size
    padded_rho = zeros(Complex{eltype(rho_array)}, padded_rho_size)
    padded_rho[1:grid_size[1], 1:grid_size[2], 1:grid_size[3]] = rho_array

    # --- 2. Create an FFT plan ---
    backend = get_backend(rho_array)
    if backend isa CPU
        fft_plan = plan_fft(padded_rho)
    elseif backend isa CUDABackend
        fft_plan = plan_fft(padded_rho)
    else
        error("Unsupported backend for FFT: ", typeof(backend))
    end

    # --- 3. Perform forward FFT on padded_rho ---
    fft_rho = fft_plan * padded_rho

    # --- 4. Loop through components (Ex, Ey, Ez) and calculate Green's function ---
    green_function_real = zeros(eltype(rho_array), padded_rho_size)
    green_function_fft = similar(fft_rho)

    for comp_idx in 1:3
        kernel! = SpaceCharge.generate_igf_kernel!(backend)
        kernel!(
            green_function_real,
            padded_rho_size,
            delta,
            comp_idx,
            gam,
            ndrange=padded_rho_size,
        )

        green_function_fft = fft_plan * green_function_real

        fft_field_comp = fft_rho .* green_function_fft

        field_comp_real = inv(fft_plan) * fft_field_comp

        efield_out[1:grid_size[1], 1:grid_size[2], 1:grid_size[3], comp_idx] = real.(field_comp_real[1:grid_size[1], 1:grid_size[2], 1:grid_size[3]])
    end

    # --- 5. Calculate B-field from Ex and Ey ---
    beta0 = sqrt(1 - 1 / gam^2)
    clight = 299792458.0
    bfield_out[1:grid_size[1], 1:grid_size[2], 1:grid_size[3], 1] = -efield_out[1:grid_size[1], 1:grid_size[2], 1:grid_size[3], 2] * beta0 / clight
    bfield_out[1:grid_size[1], 1:grid_size[2], 1:grid_size[3], 2] = efield_out[1:grid_size[1], 1:grid_size[2], 1:grid_size[3], 1] * beta0 / clight
    bfield_out[1:grid_size[1], 1:grid_size[2], 1:grid_size[3], 3] .= 0.0

    return nothing
end
    # --- 1. Pad the rho array ---
    padded_rho_size = 2 .* grid_size
    padded_rho = zeros(Complex{eltype(rho_array)}, padded_rho_size)
    padded_rho[1:grid_size[1], 1:grid_size[2], 1:grid_size[3]] = rho_array

    # --- 2. Create an FFT plan ---
    backend = get_backend(rho_array)
    if backend isa CPU
        fft_plan = plan_fft(padded_rho)
    elseif backend isa CUDABackend
        fft_plan = plan_fft(padded_rho)
    else
        error("Unsupported backend for FFT: ", typeof(backend))
    end

    # --- 3. Perform forward FFT on padded_rho ---
    fft_rho = fft_plan * padded_rho

    # --- 4. Loop through components (Ex, Ey, Ez) and calculate Green's function ---
    green_function_real = zeros(eltype(rho_array), padded_rho_size)
    green_function_fft = similar(fft_rho)

    for comp_idx in 1:3
        kernel! = SpaceCharge.generate_igf_kernel!(backend)
        kernel!(
            green_function_real,
            padded_rho_size,
            delta,
            comp_idx,
            mesh.gamma,
            ndrange=padded_rho_size,
        )

        green_function_fft = fft_plan * green_function_real

        fft_field_comp = fft_rho .* green_function_fft

        field_comp_real = inv(fft_plan) * fft_field_comp

        efield_out[1:grid_size[1], 1:grid_size[2], 1:grid_size[3], comp_idx] = real.(field_comp_real[1:grid_size[1], 1:grid_size[2], 1:grid_size[3]])
    end

    # --- 5. Calculate B-field from Ex and Ey ---
    bfield_out[1:grid_size[1], 1:grid_size[2], 1:grid_size[3], 1] = -efield_out[1:grid_size[1], 1:grid_size[2], 1:grid_size[3], 2]
    bfield_out[1:grid_size[1], 1:grid_size[2], 1:grid_size[3], 2] = efield_out[1:grid_size[1], 1:grid_size[2], 1:grid_size[3], 1]
    bfield_out[1:grid_size[1], 1:grid_size[2], 1:grid_size[3], 3] .= 0.0

    return nothing
end

function solve!(mesh::Mesh3D, ::FreeSpace; at_cathode::Bool=false)
    # Calculate fields for the real charge distribution
    _calculate_fields_from_rho!(
        mesh.rho,
        mesh.grid_size,
        mesh.min_bounds,
        mesh.max_bounds,
        mesh.delta,
        mesh.efield,
        mesh.bfield,
    )

    if at_cathode
        # Create a temporary array for the image charge distribution
        image_rho = zeros(eltype(mesh.rho), mesh.grid_size)

        # Populate image_rho by z-flipping and negating the real rho
        # Assuming cathode is at z_min (mesh.min_bounds[3])
        # The image charge at z' for a real charge at z is z' = 2*z_cathode - z
        # For a grid, this means flipping the z-index.
        for k in 1:mesh.grid_size[3]
            # Calculate the corresponding image z-index
            # If z_cathode is at the first plane (k=1), then image of k=1 is k=1, image of k=2 is k=0 (out of bounds)
            # This needs to be carefully mapped. A simple flip is (grid_size[3] - k + 1)
            image_k = mesh.grid_size[3] - k + 1
            if 1 <= image_k <= mesh.grid_size[3]
                image_rho[:, :, image_k] = -mesh.rho[:, :, k]
            end
        end

        # Create temporary arrays for image fields
        image_efield = zeros(eltype(mesh.efield), size(mesh.efield))
        image_bfield = zeros(eltype(mesh.bfield), size(mesh.bfield))

        # Calculate fields for the image charge distribution
        _calculate_fields_from_rho!(
            image_rho,
            mesh.grid_size,
            mesh.min_bounds,
            mesh.max_bounds,
            mesh.delta,
            image_efield,
            image_bfield,
            mesh.gamma
        )

        # Add the real and image fields together
        # Be careful with signs for B-field (image charges move in opposite direction)
        mesh.efield .+= image_efield
        mesh.bfield .-= image_bfield # B-field from image charge is subtracted
    end

    return nothing
end
