using SpaceCharge
using CUDA

# GPU usage example: single point charge, using array_type=CuArray

# Define grid size
grid_size = (64, 64, 64)

# Define particle positions and charges on the GPU
particles_x_gpu = CuArray([0.0])
particles_y_gpu = CuArray([0.0])
particles_z_gpu = CuArray([0.0])
particles_q_gpu = CuArray([1.0e-9])

# Create a Mesh3D object with automatic bounds on the GPU (array_type=CuArray)
mesh_gpu = Mesh3D(grid_size, particles_x_gpu, particles_y_gpu, particles_z_gpu; array_type=CuArray)

# Deposit particle charges onto the grid (on GPU)
deposit!(mesh_gpu, particles_x_gpu, particles_y_gpu, particles_z_gpu, particles_q_gpu)

# Solve for electric field (on GPU)
solve!(mesh_gpu)

# Interpolate fields back to particle positions (on GPU)
Ex_gpu, Ey_gpu, Ez_gpu = interpolate_field(mesh_gpu, particles_x_gpu, particles_y_gpu, particles_z_gpu)

println("GPU Electric Field at particle position: Ex=$(Ex_gpu[1]), Ey=$(Ey_gpu[1]), Ez=$(Ez_gpu[1])") 