#pragma once

#include <cuda_runtime.h>
#include "layer.hpp"

namespace CUDANet::Kernels {

__global__ void convolution(
    const float* __restrict__ d_input,
    const float* __restrict__ d_kernel,
    const float* __restrict__ d_bias,
    float* __restrict__ d_output,
    const Shape input_shape,
    const Shape padding_shape,
    const Shape kernel_shape,
    const Shape stride_shape,
    const Shape output_shape
);

}  // namespace CUDANet::Kernels
