module SpaceCharge

using KernelAbstractions
using Adapt
using CUDA

include("mesh.jl")
include("deposition.jl")
include("interpolation.jl")
include("green_functions.jl")
include("solvers/free_space.jl")
include("utils.jl")

export Mesh3D, deposit!, clear_mesh!, interpolate_field, solve!

end # module SpaceCharge
