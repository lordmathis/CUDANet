#include "backend/cuda/kernels/activation_functions.cuh"

using namespace CUDANet;

template 
__global__ void Kernels::sigmoid<float>(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const unsigned int len
);

template <typename T>
__global__ void Kernels::sigmoid(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const unsigned int len
) {
    int stride = gridDim.x * blockDim.x;
    int tid    = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = tid; i < len; i += stride) {
        dst[i] = 1.0 / (1.0 + exp(-src[i]));
    }
}

template __global__ void Kernels::relu<float>(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const unsigned int len
);

template <typename T>
__global__ void Kernels::relu(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const unsigned int len
) {
    int stride = gridDim.x * blockDim.x;
    int tid    = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = tid; i < len; i += stride) {
        dst[i] = src[i] < 0.0 ? 0.0 : src[i];
    }
}
