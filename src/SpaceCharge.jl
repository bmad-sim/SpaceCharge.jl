module SpaceCharge

using KernelAbstractions
using Adapt
using CUDA
using FFTW

include("utils.jl")
include("mesh.jl")
include("deposition.jl")
include("interpolation.jl")
include("green_functions.jl")
include("solvers/free_space.jl")

function __init__()
    FFTW.set_num_threads(Threads.nthreads())
end

export Mesh3D, deposit!, clear_mesh!, interpolate_field, solve!

end # module SpaceCharge
