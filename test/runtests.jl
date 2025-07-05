using SpaceCharge
using Test
using CUDA
using KernelAbstractions
using FFTW

# Include individual test modules
include("test_mesh.jl")
include("test_deposition.jl")
include("test_solvers.jl")
include("test_interpolation.jl")
include("test_gpu.jl")

# Qualitative tests
@testset "SpaceCharge.jl" begin
    # Run all test modules
    run_mesh_tests()
    run_deposition_tests()
    run_solver_tests()
    run_interpolation_tests()
    
    # GPU tests only if CUDA is available
    if CUDA.functional()
        run_gpu_tests()
    end
end

# Quantitative tests
include("analytical_test.jl")
