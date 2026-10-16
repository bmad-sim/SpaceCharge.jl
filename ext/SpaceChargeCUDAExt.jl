module SpaceChargeCUDAExt

import SpaceCharge, CUDA

function SpaceCharge._inverse_transform!(output::CUDA.CuArray{T,3}, plan,
                                        input::CUDA.CuArray{Complex{T},3}) where {T<:Union{Float32,Float64}}
    CUDA.CUFFT.assert_applicable(plan, input, output)
    # CUDA's mul! preserves C2R inputs; this solver owns and replaces its product
    # before every inverse. Keep the same cuFFT execution and stream handling.
    CUDA.CUFFT.unsafe_execute!(plan, input, output)
    return output
end

end
