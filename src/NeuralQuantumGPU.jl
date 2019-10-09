module NeuralQuantumGPU

using CuArrays: CuArrays, @cufunc
using CuArrays: CuArrays.GPUArrays.GPUArray
using NeuralQuantum: NeuralQuantum, State, _std_state_batch, store_state!
using UnsafeArrays
using NNlib

#@cufunc NNlib.softplus(x::Complex) = log1p(exp(x))#log(one(x) + exp(x))

@cufunc NeuralQuantum.ℒ(x) = one(x) + exp(x)
@cufunc NeuralQuantum.∂logℒ(x) = one(x)/(one(x)+exp(-x))

_gpu_logℒ(x) = log1p(exp(x))
@cufunc _gpu_logℒ(x::Real) = log1p(exp(x))
@cufunc _gpu_logℒ(x::Complex) = log(one(x) + exp(x))

@cufunc NeuralQuantum.logℒ(x) = _gpu_logℒ(x)


struct CPUCachedBatchState{G,V} <: NeuralQuantum.State
    gpu_state::G
    cpu_state::V
end
NeuralQuantum.config(v::CPUCachedBatchState) = _copy_to_gpu!(v.gpu_state, v.cpu_state)

_copy_to_gpu!(g::AbstractArray, v::AbstractArray) = copy!(g, v)
_copy_to_gpu!(g::Tuple, c::Tuple) = begin
    g1, g2 = g
    c1, c2 = c
    copy!(g1, c1)
    copy!(g2, c2)
    return g
end

NeuralQuantum.preallocate_state_batch(arrT::GPUArray,
                                      T::Type{<:Real},
                                      v::State,
                                      batch_sz) = begin
    v_gpu = _std_state_batch(arrT, T, v, batch_sz)
    v_cpu = _std_state_batch(collect(arrT), T, v, batch_sz)
    return CPUCachedBatchState(v_gpu, v_cpu)
end

NeuralQuantum.store_state!(cache::CPUCachedBatchState,
             v,
             i::Integer) = begin
    store_state!(cache.cpu_state, v, i)
    return cache
end

UnsafeArrays.uview(arr::GPUArray) = arr

end # module
