module SpaceChargeCUDAExt

using SpaceCharge, CUDA

# KernelAbstractions' CUDABackend does not encode the device ordinal. Include
# the calling device to keep both kernel launches and FFT plans on the owner.
SpaceCharge._same_device(a::CuArray, b::CuArray) =
    CUDA.device(a) == CUDA.device(b) == CUDA.device()

end
