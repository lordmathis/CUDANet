#pragma once

#include <cuda_runtime.h>

namespace CUDANet::Kernels {

template <typename T>
__global__ void mat_vec_mul(
    const T* __restrict__ d_matrix,
    const T* __restrict__ d_vector,
    T* __restrict__ d_output,
    const unsigned int w,
    const unsigned int h
);

template <typename T>
__global__ void vec_vec_add(
    const T* __restrict__ d_vector1,
    const T* __restrict__ d_vector2,
    T* __restrict__ d_output,
    const unsigned int w
);

template <typename T>
__global__ void vec_vec_sub(
    const T* __restrict__ d_vector1,
    const T* __restrict__ d_vector2,
    T* __restrict__ d_output,
    const unsigned int w
);

template <typename T>
__global__ void vec_vec_mul(
    const T* __restrict__ d_vector1,
    const T* __restrict__ d_vector2,
    T* __restrict__ d_output,
    const unsigned int w
);

template <typename T>
__global__ void vec_scalar_sub(
    const T* __restrict__ d_src,
    T* __restrict__ d_out,
    const T* __restrict__ d_scalar,
    const unsigned int len
);

template <typename T>
__global__ void vec_scalar_add(
    const T* __restrict__ d_src,
    T* __restrict__ d_out,
    const T* __restrict__ d_scalar,
    const unsigned int len
);

template <typename T>
__global__ void vec_scalar_div(
    const T* __restrict__ d_src,
    T* __restrict__ d_out,
    const T* __restrict__ d_scalar,
    const unsigned int len
);

template <typename T>
__global__ void vec_scalar_mul(
    const T* __restrict__ d_src,
    T* __restrict__ d_out,
    const T* __restrict__ d_scalar,
    const unsigned int len
);

template <typename T>
__global__ void vec_exp(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const unsigned int len
);

template <typename T>
__global__ void vec_sqrt(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const unsigned int len
);

template <typename T>
__global__ void vec_scale(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const T* __restrict__ scale,
    const T* epsilon,
    const unsigned int len
);

template <typename T>
__global__ void max_reduce(
    const T* __restrict__ d_vector,
    T* __restrict__ d_output,
    const unsigned int len
);

template <typename T>
__global__ void sum_reduce(
    const T* __restrict__ d_vector,
    T* __restrict__ d_output,
    const unsigned int len
);

}  // namespace CUDANet::Kernels
