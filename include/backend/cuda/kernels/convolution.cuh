#pragma once

#include <cuda_runtime.h>
#include "layer.hpp"

namespace CUDANet::Kernels {

template <typename T>
__global__ void convolution(
    const T* __restrict__ d_input,
    const T* __restrict__ d_kernel,
    const T* __restrict__ d_bias,
    T* __restrict__ d_output,
    const Shape input_shape,
    const Shape padding_shape,
    const Shape kernel_shape,
    const Shape stride_shape,
    const Shape output_shape
);

}  // namespace CUDANet::Kernels
