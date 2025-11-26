#pragma once

#include <cuda_runtime.h>
#include "layer.hpp"

namespace CUDANet::Kernels {

template <typename T>
__global__ void max_pool(
    const T* __restrict__ d_input,
    T* __restrict__ d_output,
    const Shape input_shape,
    const Shape output_shape,
    const Shape pool_shape,
    const Shape stride_shape,
    const Shape padding_shape
);

template <typename T>
__global__ void avg_pool(
    const T* __restrict__ d_input,
    T* __restrict__ d_output,
    const Shape input_shape,
    const Shape output_shape,
    const Shape pool_shape,
    const Shape stride_shape,
    const Shape padding_shape
);

}  // namespace CUDANet::Kernels
