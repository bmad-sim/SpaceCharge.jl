module SpaceCharge

using KernelAbstractions
using Adapt

include("mesh.jl")
include("deposition.jl")
include("interpolation.jl")
include("green_functions.jl")
include("solvers/free_space.jl")
include("solvers/rectangular_pipe.jl")
include("utils.jl")

export Mesh3D, deposit!, interpolate_field, generate_igf_kernel!, solve!, FreeSpace, RectangularPipe

end # module OpenSpaceCharge
