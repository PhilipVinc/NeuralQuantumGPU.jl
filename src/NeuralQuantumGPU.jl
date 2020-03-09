module NeuralQuantumGPU

using CuArrays: CuArrays, CuArray, @cufunc, CUBLAS
const use_cuda = Ref(false)

using NeuralQuantum: NeuralQuantum
using NNlib

gpu(x) = use_cuda[] ? fmap(CuArrays.cu, x) : x

# gpu stuff

function __init__()

  # cuda stuff
  precompiling = ccall(:jl_generating_output, Cint, ()) != 0

  # we don't want to include the CUDA module when precompiling,
  # or we could end up replacing it at run time (triggering a warning)
  precompiling && return

  if !CuArrays.functional()
    # nothing to do here, and either CuArrays or one of its dependencies will have warned
  else
    use_cuda[] = true
    include("CuArrays/cuda.jl")
    include("CuArrays/upstream.jl")

    # FIXME: this functionality should be conditional at run time by checking `use_cuda`
    #        (or even better, get moved to CuArrays.jl as much as possible)
    if CuArrays.has_cudnn()
      #include(joinpath(@__DIR__, "cuda/cuda.jl"))
    else
      @warn "CuArrays.jl did not find libcudnn. Some functionality will not be available."
    end
  end
end


end # module
