using SpaceCharge

# Existing image-charge electric-field convention at the plane z=0.
x=[0.0]; y=[0.0]; z=[0.01]; charge=[1e-9]
sc=FFTIGF((32,32,32);bounds=((-0.02,-0.02,0.0),(0.02,0.02,0.04)),at_cathode=true)
outputs=ntuple(_->similar(x),3)
solve!(sc,x,y,z,charge)
field!(outputs...,sc,x,y,z)
println("E at the source (V/m): ",outputs)

# Expert workflow: reuse the deposited lab charges at another gamma.
solve!(sc;gamma=2)
field!(outputs...,sc,x,y,z)
println("E at gamma=2 (V/m): ",outputs)
