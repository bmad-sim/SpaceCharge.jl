using SpaceCharge

# FreeSpace solver with cathode example: single point charge above z=0, using at_cathode=true

# Define grid size
grid_size = (32, 32, 32)

# Define particle positions and charges (single point charge above cathode at z=0)
particles_x = [0.0]
particles_y = [0.0]
particles_z = [0.01]
particles_q = [1.0e-9]

# Create a Mesh3D object with automatic bounds (default: CPU)
mesh = Mesh3D(grid_size, particles_x, particles_y, particles_z)

# Deposit particle charges onto the grid
deposit!(mesh, particles_x, particles_y, particles_z, particles_q)

# Solve for electric field (at_cathode=true)
solve!(mesh; at_cathode=true)

# Interpolate fields back to particle position
Ex, Ey, Ez = interpolate_field(mesh, particles_x, particles_y, particles_z)

println("E-field at particle: Ex=$(Ex[1]), Ey=$(Ey[1]), Ez=$(Ez[1])") 