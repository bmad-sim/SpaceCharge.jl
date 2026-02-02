using SpaceCharge
using InteractiveUtils  # for @code_warntype

println("="^60)
println("Testing Type Stability of Green's Functions")
println("="^60)

# Test potential_green_function with Float64
println("\n1. potential_green_function with Float64:")
println("-"^60)
@code_warntype SpaceCharge.potential_green_function(1.0, 2.0, 3.0)

# Test potential_green_function with Float32
println("\n2. potential_green_function with Float32:")
println("-"^60)
@code_warntype SpaceCharge.potential_green_function(1.0f0, 2.0f0, 3.0f0)

# Test field_green_function with Float64
println("\n3. field_green_function with Float64:")
println("-"^60)
@code_warntype SpaceCharge.field_green_function(1.0, 2.0, 3.0)

# Test field_green_function with Float32
println("\n4. field_green_function with Float32:")
println("-"^60)
@code_warntype SpaceCharge.field_green_function(1.0f0, 2.0f0, 3.0f0)

println("\n"*"="^60)
println("Look for:")
println("  Blue text = good (type-stable)")
println("  Red text  = bad (type-unstable)")
println("="^60)
