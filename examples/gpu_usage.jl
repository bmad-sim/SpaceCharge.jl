using SpaceCharge
using CUDA # Make sure CUDA.jl is installed and functional

# Define grid parameters
grid_size = (64, 64, 64)
min_bounds = (-0.05, -0.05, -0.05)
max_bounds = (0.05, 0.05, 0.05)

# Create a Mesh3D object on the GPU
mesh_gpu = SpaceCharge.Mesh3D(grid_size, min_bounds, max_bounds; array_type=CuArray)

# Define particles on the GPU
particles_x_gpu = CuArray([0.0])
particles_y_gpu = CuArray([0.0])
particles_z_gpu = CuArray([0.0])
particles_q_gpu = CuArray([1.0e-9])

# Deposit and solve on the GPU (functions automatically dispatch to GPU kernels)
SpaceCharge.deposit!(mesh_gpu, particles_x_gpu, particles_y_gpu, particles_z_gpu, particles_q_gpu)
SpaceCharge.solve!(mesh_gpu, SpaceCharge.FreeSpace())

# Interpolate fields back to particle positions on the GPU
Ex_gpu, Ey_gpu, Ez_gpu, Bx_gpu, By_gpu, Bz_gpu = SpaceCharge.interpolate_field(mesh_gpu, particles_x_gpu, particles_y_gpu, particles_z_gpu)

println("GPU Electric Field at particle position: Ex=\$(Ex_gpu[1]), Ey=\$(Ey_gpu[1]), Ez=\$(Ez_gpu[1])")
