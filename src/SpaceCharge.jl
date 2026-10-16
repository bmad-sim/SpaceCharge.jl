module SpaceCharge

using KernelAbstractions
using FFTW
using LinearAlgebra: mul!

include("utils.jl")
include("interface.jl")
include("solvers/fftigf.jl")
include("fft/workspace.jl")
include("deposition.jl")
include("interpolation.jl")
include("green_functions.jl")
include("fft/transforms.jl")
include("solvers/free_space.jl")

export FFTIGF, solve!, field!, deposit!

end # module SpaceCharge
