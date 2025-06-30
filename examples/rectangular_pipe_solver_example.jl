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

# Solve for the electric and magnetic fields using RectangularPipe boundary conditions
SpaceCharge.solve!(mesh, SpaceCharge.RectangularPipe())

println("Rectangular Pipe Solver Example:")
println("  Total charge deposited: $(sum(mesh.rho))")
println("  Electric field (first element): $(mesh.efield[1,1,1,:])")
println("  Magnetic field (first element): $(mesh.bfield[1,1,1,:])")
