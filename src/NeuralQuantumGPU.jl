module NeuralQuantumGPU

using CuArrays: CuArrays, CuArray, @cufunc, CUBLAS
const use_cuda = Ref(false)

using NeuralQuantum: NeuralQuantum
using NNlib

gpu(x) = use_cuda[] ? fmap(CuArrays.cu, x) : x

# gpu stuff
include("CuArrays/cuda.jl")
include("CuArrays/upstream.jl")

function __init__()
  use_cuda[] = CuArrays.functional() # Can be overridden after load with `Flux.use_cuda[] = false`
  if CuArrays.functional()
    if !CuArrays.has_cudnn()
      @warn "CuArrays.jl found cuda, but did not find libcudnn. Some functionality will not be available."
    end
  end
end

end # module
