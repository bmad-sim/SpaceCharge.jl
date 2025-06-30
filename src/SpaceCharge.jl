module SpaceCharge

using FFTW
using StaticArrays
using Distributions
using Plots

include("utils.jl")
include("charge_deposition.jl")
include("field_solver.jl")
include("visualization.jl")

export deposit_charge_cic, calculate_field, green_function, generate_particles, plot_field

end # module