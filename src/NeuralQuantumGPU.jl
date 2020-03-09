module NeuralQuantumGPU

using CuArrays: CuArrays, CuArray, @cufunc, CUBLAS
const use_cuda = Ref(true)

using NeuralQuantum: NeuralQuantum
using NNlib

gpu(x) = use_cuda[] ? fmap(CuArrays.cu, x) : x

# gpu stuff
include("CuArrays/cuda.jl")
include("CuArrays/upstream.jl")

end # module
