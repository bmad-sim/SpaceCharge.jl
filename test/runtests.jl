using SpaceCharge
using Test
using KernelAbstractions
using FFTW

# Conditionally load CUDA for GPU tests
# Note: CUDA is not a required test dependency. To run GPU tests, install CUDA manually:
#   using Pkg; Pkg.add("CUDA")
const CUDA_AVAILABLE = try
    using CUDA
    CUDA.functional()
catch
    false
end

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
    if CUDA_AVAILABLE
        run_gpu_tests()
    else
        @info "CUDA not available, skipping GPU tests"
    end
end

# Quantitative tests
include("analytical_test.jl")
