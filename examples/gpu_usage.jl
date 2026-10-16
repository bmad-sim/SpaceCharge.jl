using SpaceCharge, CUDA, KernelAbstractions
CUDA.functional(true)
CUDA.allowscalar(false)

x=CuArray(Float32[-0.001,0.001]);y=CUDA.zeros(Float32,2);z=copy(y)
charge=CuArray(Float32[0.5e-9,0.5e-9])
sc=FFTIGF((32,32,32);T=Float32,backend=get_backend(x))
outputs=ntuple(_->similar(x),3)
solve!(sc,x,y,z,charge;gamma=2,bounds=((-0.01,-0.01,-0.01),(0.01,0.01,0.01)))
field!(outputs...,sc,x,y,z)
CUDA.synchronize() # synchronize at host access/timing boundaries
println("GPU E (V/m): ",Array.(outputs))
