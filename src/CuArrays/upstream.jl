using CuArrays: cudims, @cuda, cudaconvert, cufunction, mapreducedim_kernel_parallel
using CuArrays: CUDAnative, CUDAdrv, attribute, @cufunc

#
function Statistics.mean!(y::CuVector, x::CuArray{T,2}) where T
    ỹ = reshape(y, length(y), 1)
    mean!(ỹ, x)
    return y
end

function Statistics.mean!(y::CuVector, x::CuArray{T,3}) where T
    ỹ = reshape(y, length(y), 1, 1)
    mean!(ỹ, x)
    return y
end
