module SpaceCharge

using KernelAbstractions
using Atomix
using AbstractFFTs
using FFTW
using LinearAlgebra: mul!
using AtomicAndPhysicalConstants: EPS_0, E_CHARGE

include("interface.jl")
include("solvers/fftigf/solver.jl")
include("solvers/fftigf/workspace.jl")
include("solvers/fftigf/deposition.jl")
include("solvers/fftigf/interpolation.jl")
include("solvers/fftigf/green_functions.jl")
include("solvers/fftigf/transforms.jl")
include("solvers/fftigf/solve.jl")

export FFTIGF, solve!, field!, deposit!

end # module SpaceCharge
