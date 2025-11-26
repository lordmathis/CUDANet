#pragma once

#include <cuda_runtime.h>

namespace CUDANet::Kernels {


template <typename T>
__global__ void sigmoid(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const unsigned int len
);

template <typename T>
__global__ void relu(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const unsigned int len
);

}  // namespace CUDANet::Kernels
