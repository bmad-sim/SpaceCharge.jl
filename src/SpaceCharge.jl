module SpaceCharge

using KernelAbstractions
using Atomix
using AbstractFFTs
using FFTW
using LinearAlgebra: mul!

using AtomicAndPhysicalConstants: EPS_0, E_CHARGE
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
