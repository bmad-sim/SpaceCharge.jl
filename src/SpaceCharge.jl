module SpaceCharge

using KernelAbstractions
using Adapt

include("mesh.jl")
include("deposition.jl")
include("interpolation.jl")
include("green_functions.jl")

export Mesh3D, deposit!, interpolate_field, generate_igf_kernel!

end # module OpenSpaceCharge
