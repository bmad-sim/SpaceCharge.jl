
using Plots
plotly()

"""
    plot_field(electric_field, dx, dy, dz)

Generates an interactive 3D quiver plot of the electric vector field.

# Arguments
- `electric_field::Array{SVector{3, Float64}, 3}`: A 3D array of electric field vectors.
- `dx::Real`: Grid spacing in the x-direction.
- `dy::Real`: Grid spacing in the y-direction.
- `dz::Real`: Grid spacing in the z-direction.
"""
function plot_field(electric_field::Array{SVector{3, Float64}, 3}, dx::Real, dy::Real, dz::Real)
    nx, ny, nz = size(electric_field)
    x = [i * dx for i in 1:nx]
    y = [j * dy for j in 1:ny]
    z = [k * dz for k in 1:nz]

    u = [field[1] for field in electric_field]
    v = [field[2] for field in electric_field]
    w = [field[3] for field in electric_field]

    quiver(x, y, z, quiver=(u, v, w), title="Electric Vector Field")
end
