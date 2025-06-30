using SpaceCharge

# Define grid parameters
grid_size = (32, 32, 32)
min_bounds = (-0.05, -0.05, -0.05) # meters
max_bounds = (0.05, 0.05, 0.05)   # meters

# Create a Mesh3D object
mesh = SpaceCharge.Mesh3D(grid_size, min_bounds, max_bounds)

# Define some particles (e.g., a single point charge at the center)
particles_x = [0.0]
particles_y = [0.0]
particles_z = [0.0]
particles_q = [1.0e-9] # 1 nC charge

# Deposit particle charges onto the grid
SpaceCharge.deposit!(mesh, particles_x, particles_y, particles_z, particles_q)

# Solve for the electric and magnetic fields using FreeSpace boundary conditions
SpaceCharge.solve!(mesh, SpaceCharge.FreeSpace())

# Interpolate fields back to particle positions (example for the first particle)
Ex, Ey, Ez, Bx, By, Bz = SpaceCharge.interpolate_field(mesh, particles_x, particles_y, particles_z)

println("Electric Field at particle position: Ex=$(Ex[1]), Ey=$(Ey[1]), Ez=$(Ez[1])")
println("Magnetic Field at particle position: Bx=$(Bx[1]), By=$(By[1]), Bz=$(Bz[1])")

# You can now access the calculated fields on the mesh:
# mesh.rho   # Charge density
# mesh.efield # Electric field (Ex, Ey, Ez components)
# mesh.bfield # Magnetic field (Bx, By, Bz components)
