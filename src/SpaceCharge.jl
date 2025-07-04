module SpaceCharge

using KernelAbstractions
using Adapt
using CUDA

include("mesh.jl")
include("deposition.jl")
include("interpolation.jl")
include("green_functions.jl")
include("solvers/free_space.jl")
include("solvers/rectangular_pipe.jl")
include("utils.jl")

export Mesh3D, deposit!, clear_mesh!, interpolate_field, solve!, FreeSpace, RectangularPipe, analytical_efield, BoundaryCondition

end # module SpaceCharge
