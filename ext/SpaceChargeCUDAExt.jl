module SpaceChargeCUDAExt

using SpaceCharge
import CUDA
using CUDA: CuDeviceArray, AS

# CUDA 6 moved device intrinsics into CUDACore.
@static if isdefined(CUDA, :CUDACore)
    using CUDA.CUDACore: @asmcall
else
    using CUDA: @asmcall
end

@inline function SpaceCharge._deposit_add!(rho::CuDeviceArray{T,3,AS.Global},
                                           charge::T, i, j, k) where {T<:Union{Float32,Float64}}
    index = i + size(rho, 1) * (j - 1 + size(rho, 2) * (k - 1))
    ptr = pointer(rho, index)
    # CIC never uses the old value. Explicit reduction avoids a returned-value
    # atomic and its serialization cost under contention. Relaxed ordering is
    # sufficient: subsequent kernels consume rho in stream order.
    if T === Float32
        @asmcall("red.global.add.f32 [\$0], \$1;", "l,f,~{memory}", true, Cvoid,
                 Tuple{Core.LLVMPtr{Float32,AS.Global},Float32}, ptr, charge)
    else
        @asmcall("red.global.add.f64 [\$0], \$1;", "l,d,~{memory}", true, Cvoid,
                 Tuple{Core.LLVMPtr{Float64,AS.Global},Float64}, ptr, charge)
    end
    return nothing
end

end
