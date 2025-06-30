using SpaceCharge

# Define grid parameters
grid_size = (32, 32, 32)
min_bounds = (-0.05, -0.05, 0.0) # Cathode at z=0
max_bounds = (0.05, 0.05, 0.1)   # meters

# Create a Mesh3D object
mesh = SpaceCharge.Mesh3D(grid_size, min_bounds, max_bounds)

# Define some particles (e.g., a single point charge near the cathode)
particles_x = [0.0]
particles_y = [0.0]
particles_z = [0.01]
particles_q = [1.0e-9] # 1 nC charge

# Deposit particle charges onto the grid
SpaceCharge.deposit!(mesh, particles_x, particles_y, particles_z, particles_q)

# Solve for the electric and magnetic fields using FreeSpace boundary conditions with cathode
SpaceCharge.solve!(mesh, SpaceCharge.FreeSpace(); at_cathode=true)

println("Cathode Solver Example:")
println("  Total charge deposited: $(sum(mesh.rho))")
println("  Electric field (first element): $(mesh.efield[1,1,1,:])")
println("  Magnetic field (first element): $(mesh.bfield[1,1,1,:])")
