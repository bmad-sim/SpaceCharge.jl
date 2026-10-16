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
include("test_threaded_deposition.jl")
include("test_solvers.jl")
include("test_solver_reference.jl")
include("test_pruned_fft.jl")
include("test_interpolation.jl")
include("test_potential_fd.jl")
include("test_potential_accuracy.jl")
CUDA_AVAILABLE && include("test_gpu.jl")

# Qualitative tests
@testset "SpaceCharge.jl" begin
    # Run all test modules
    run_mesh_tests()
    run_deposition_tests()
    run_threaded_deposition_tests()
    run_solver_tests()
    run_solver_reference_tests()
    run_pruned_fft_tests()
    run_solver_cache_reference_tests()
    run_interpolation_tests()
    run_potential_constructor_tests()
    run_potential_reference_tests()
    run_potential_accuracy_tests()
    
    # GPU tests only if CUDA is available
    if CUDA_AVAILABLE
        run_gpu_tests()
        run_potential_reference_tests(CUDABackend())
        run_potential_accuracy_tests(CUDABackend())
    else
        @info "CUDA not available, skipping GPU tests"
    end
end

# Quantitative tests
include("analytical_test.jl")
