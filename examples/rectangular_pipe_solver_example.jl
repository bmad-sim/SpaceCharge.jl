using SpaceCharge

# RectangularPipe solver example: single point charge, using RectangularPipe boundary

# Define grid size
grid_size = (32, 32, 32)

# Define particle positions and charges (single point charge at origin)
particles_x = [0.0]
particles_y = [0.0]
particles_z = [0.0]
particles_q = [1.0e-9]

# Create a Mesh3D object with automatic bounds (default: CPU)
mesh = Mesh3D(grid_size, particles_x, particles_y, particles_z)

# Deposit particle charges onto the grid
deposit!(mesh, particles_x, particles_y, particles_z, particles_q)

# Solve for electric and magnetic fields (RectangularPipe boundary)
solve!(mesh, RectangularPipe())

# Print field at the center of the mesh
i = div(grid_size[1], 2)
j = div(grid_size[2], 2)
k = div(grid_size[3], 2)
println("E-field at center: ", mesh.efield[i, j, k, :])
println("B-field at center: ", mesh.bfield[i, j, k, :]) 