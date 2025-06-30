
using FFTW
using StaticArrays

"""
    calculate_field(charge_density, dx, dy, dz)

Calculates the electric field from a given charge density distribution on a 3D
grid using FFT-based convolution with the Green's function.

# Arguments
- `charge_density::Array{Float64, 3}`: A 3D array representing the charge density on the grid.
- `dx::Real`: Grid spacing in the x-direction.
- `dy::Real`: Grid spacing in the y-direction.
- `dz::Real`: Grid spacing in the z-direction.

# Returns
- `Array{SVector{3, Float64}, 3}`: A 3D array where each element is a 3D vector representing the electric field at that grid point.
"""
function calculate_field(charge_density::Array{Float64, 3}, dx::Real, dy::Real, dz::Real)
    nx, ny, nz = size(charge_density)

    # 1. Get the Green's function
    green = green_function(nx, ny, nz, dx, dy, dz)

    # 2. Perform 3D FFT of charge density and Green's function
    rho_k = fft(charge_density)
    green_k = fft(green)

    # 3. Multiply in Fourier space to get the potential
    potential_k = rho_k .* green_k

    # 4. Calculate electric field components in Fourier space
    kx = fftfreq(nx, 2.0 * pi / dx)
    ky = fftfreq(ny, 2.0 * pi / dy)
    kz = fftfreq(nz, 2.0 * pi / dz)

    Ex_k = -im .* kx .* potential_k
    Ey_k = -im .* ky .* potential_k
    Ez_k = -im .* kz .* potential_k

    # 5. Perform inverse 3D FFT to get the electric field in real space
    Ex = real(ifft(Ex_k))
    Ey = real(ifft(Ey_k))
    Ez = real(ifft(Ez_k))

    # 6. Combine into an array of 3D vectors
    electric_field = [SVector(Ex[i,j,k], Ey[i,j,k], Ez[i,j,k]) for i in 1:nx, j in 1:ny, k in 1:nz]

    return electric_field
end
