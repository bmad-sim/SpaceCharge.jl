using SpaceCharge

# 1. Set up grid and particle parameters
nx, ny, nz = 32, 32, 32
dx, dy, dz = 0.1, 0.1, 0.1
num_particles = 1_000_000

# 2. Generate particles with an isotropic Gaussian distribution
mean_pos = [1.6, 1.6, 1.6]  # Centered in the grid
cov_matrix = [0.5 0 0; 0 0.5 0; 0 0 0.5]  # Uncorrelated, isotropic
particles = generate_particles(num_particles, mean_pos, cov_matrix)

# 3. Deposit charge on the grid using CIC
charge_density = deposit_charge_cic(particles, nx, ny, nz, dx, dy, dz)

# 4. Calculate the electric field
electric_field = calculate_field(charge_density, dx, dy, dz)

println("Electric field calculation complete.")

# 5. Plot the electric field
println("Generating plot...")
plot = plot_field(electric_field, dx, dy, dz)
display(plot)

println("Plot displayed. You can interact with it in the plot pane.")