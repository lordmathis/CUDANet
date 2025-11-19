#pragma once

#include <cuda_runtime.h>
#include "layer.hpp"

namespace CUDANet::Kernels {

__global__ void max_pool(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    const Shape input_shape,
    const Shape output_shape,
    const Shape pool_shape,
    const Shape stride_shape,
    const Shape padding_shape
);

__global__ void avg_pool(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    const Shape input_shape,
    const Shape output_shape,
    const Shape pool_shape,
    const Shape stride_shape,
    const Shape padding_shape
);

}  // namespace CUDANet::Kernels
