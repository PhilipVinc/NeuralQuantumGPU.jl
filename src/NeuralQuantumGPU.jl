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


using Base: ReshapedArray
@inline function NeuralQuantum._batched_outer_prod!(R::ReshapedArray,
    vb::CuArray, wb::CuArray)
    T=eltype(vb)
    dims_all = R.dims
    under    = R.parent
    under_indices = under.indices
    start = first(under_indices[1])
    n_batches = length(under_indices[2])
    batch_size = size(under.parent, 1)

    Ri  = CuArray{eltype(R),2}(under.parent.buf, dims_all[1:2], own=false)
    vbi = CuArray{eltype(R),1}(vb.buf, (size(vb, 1),), own=false)
    wbi = CuArray{eltype(R),1}(wb.buf, (size(wb, 1),), own=false)

    for i=1:size(R, 3)
        Ri.offset = (i-1)*Base.elsize(Ri)*batch_size
        vbi.offset = (i-1)*Base.elsize(vbi)*length(vbi)
        wbi.offset = (i-1)*Base.elsize(wbi)*length(wbi)
        fill!(Ri, 0)
        BLAS.ger!(one(T), vbi, wbi, Ri)
    end
    return R
end

@inline function NeuralQuantum._batched_outer_prod!(R::ReshapedArray, α,
    vb::GPUArray, wb::GPUArray)
    T=eltype(vb)
    dims_all = R.dims
    under    = R.parent
    under_indices = under.indices
    start = first(under_indices[1])
    n_batches = length(under_indices[2])
    batch_size = size(under.parent, 1)

    Ri  = CuArray{eltype(R),2}(under.parent.buf, dims_all[1:2], own=false)
    vbi = CuArray{eltype(R),1}(vb.buf, (size(vb, 1),), own=false)
    wbi = CuArray{eltype(R),1}(wb.buf, (size(wb, 1),), own=false)

    for i=1:size(R, 3)
        Ri.offset = (i-1)*Base.elsize(Ri)*batch_size
        vbi.offset = (i-1)*Base.elsize(vbi)*length(vbi)
        wbi.offset = (i-1)*Base.elsize(wbi)*length(wbi)
        fill!(Ri, 0)
        BLAS.ger!(T(α), vbi, wbi, Ri)
    end
    return R
end

@inline function NeuralQuantum._batched_outer_prod_∑!(R::ReshapedArray, α,
    vb::GPUArray, wb::GPUArray, vb2::GPUArray, wb2::GPUArray)
    T=eltype(vb)
    dims_all = R.dims
    under    = R.parent
    under_indices = under.indices
    start = first(under_indices[1])
    n_batches = length(under_indices[2])
    batch_size = size(under.parent, 1)

    Ri   = CuArray{eltype(R),2}(under.parent.buf, dims_all[1:2], own=false)
    vbi  = CuArray{eltype(R),1}(vb.buf, (size(vb, 1),), own=false)
    wbi  = CuArray{eltype(R),1}(wb.buf, (size(wb, 1),), own=false)

    vb2i = CuArray{eltype(R),1}(vb2.buf, (size(vb2, 1),), own=false)
    wb2i = CuArray{eltype(R),1}(wb2.buf, (size(wb2, 1),), own=false)

    for i=1:size(R, 3)
        Ri.offset = (i-1)*Base.elsize(Ri)*batch_size
        vbi.offset = (i-1)*Base.elsize(vbi)*length(vbi)
        wbi.offset = (i-1)*Base.elsize(wbi)*length(wbi)
        vb2i.offset = (i-1)*Base.elsize(vb2i)*length(vb2i)
        wb2i.offset = (i-1)*Base.elsize(wb2i)*length(wb2i)
        fill!(Ri, 0)
        BLAS.ger!(T(α), vbi, wbi, Ri)
        BLAS.ger!(T(α), vb2i, wb2i, Ri)
    end
    return R
end

@inline function NeuralQuantum._batched_outer_prod_Δ!(R::ReshapedArray, α,
    vb::GPUArray, wb::GPUArray, vb2::GPUArray, wb2::GPUArray)
    T=eltype(vb)
    dims_all = R.dims
    under    = R.parent
    under_indices = under.indices
    start = first(under_indices[1])
    n_batches = length(under_indices[2])
    batch_size = size(under.parent, 1)

    Ri   = CuArray{eltype(R),2}(under.parent.buf, dims_all[1:2], own=false)
    vbi  = CuArray{eltype(R),1}(vb.buf, (size(vb, 1),), own=false)
    wbi  = CuArray{eltype(R),1}(wb.buf, (size(wb, 1),), own=false)

    vb2i = CuArray{eltype(R),1}(vb2.buf, (size(vb2, 1),), own=false)
    wb2i = CuArray{eltype(R),1}(wb2.buf, (size(wb2, 1),), own=false)

    for i=1:size(R, 3)
        Ri.offset = (i-1)*Base.elsize(Ri)*batch_size
        vbi.offset = (i-1)*Base.elsize(vbi)*length(vbi)
        wbi.offset = (i-1)*Base.elsize(wbi)*length(wbi)
        vb2i.offset = (i-1)*Base.elsize(vb2i)*length(vb2i)
        wb2i.offset = (i-1)*Base.elsize(wb2i)*length(wb2i)
        fill!(Ri, 0)
        BLAS.ger!(T(α), vbi, wbi, Ri)
        BLAS.ger!(-T(α), vb2i, wb2i, Ri)
    end
    return R
end


end # module
