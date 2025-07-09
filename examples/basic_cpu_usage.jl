using SpaceCharge

# Basic CPU usage example: single point charge, default settings

# Define grid size
grid_size = (32, 32, 32)

# Define particle positions and charges (single point charge at origin)
particles_x = [0.0]
particles_y = [0.0]
particles_z = [0.0]
particles_q = [1.0e-9] # 1 nC charge

# Create a Mesh3D object with automatic bounds (default: CPU)
mesh = Mesh3D(grid_size, particles_x, particles_y, particles_z)

# Deposit particle charges onto the grid
deposit!(mesh, particles_x, particles_y, particles_z, particles_q)

# Solve for electric field
solve!(mesh)

# Interpolate fields back to particle positions
Ex, Ey, Ez = interpolate_field(mesh, particles_x, particles_y, particles_z)

println("Electric Field at particle position: Ex=$(Ex[1]), Ey=$(Ey[1]), Ez=$(Ez[1])") 