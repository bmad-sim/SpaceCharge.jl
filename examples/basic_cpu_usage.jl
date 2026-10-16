using SpaceCharge

# Positions in metres; signed macroparticle charge in coulombs.
x = [-0.001, 0.001]; y = [0.0, 0.0]; z = [0.0, 0.0]
charge = [0.5e-9, 0.5e-9]
sc = FFTIGF((32,32,32))
bounds=((-0.01,-0.01,-0.01),(0.01,0.01,0.01))
Ex, Ey, Ez = similar(x), similar(y), similar(z)
solve!(sc, x, y, z, charge, 1; bounds) # room for the separate probe below
field!(Ex, Ey, Ez, sc, x, y, z)
println("Particle E (V/m): ", (Ex,Ey,Ez))

# Reuse the prepared field at separate evaluation points, without redeposition.
xt = [0.003]; yt = [0.0]; zt = [0.0]
probes = ntuple(_ -> similar(xt), 3)
field!(probes..., sc, xt, yt, zt)
println("Probe E (V/m): ", probes)
