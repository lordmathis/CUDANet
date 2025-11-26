#include <iostream>

#include "backend/cuda/kernels/convolution.cuh"

using namespace CUDANet;

template __global__ void Kernels::convolution<float>(
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

template <typename T>
__global__ void Kernels::convolution(
    const T* __restrict__ d_input,
    const T* __restrict__ d_kernel,
    const T* __restrict__ d_bias,
    T* __restrict__ d_output,
    const Shape input_shape,
    const Shape padding_shape,
    const Shape kernel_shape,
    const Shape stride_shape,
    const Shape output_shape
) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int f = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= output_shape[0] || j >= output_shape[1] || f >= output_shape[2]) {
        return;
    }

    T sum = static_cast<t>(0);

    // Iterate over kernel and input matrix
    for (int c = 0; c < input_shape[2]; c++) {
        for (int k = 0; k < kernel_shape[0]; k++) {
            for (int l = 0; l < kernel_shape[1]; l++) {
                // if i, j is in the padding region
                if (i * stride_shape[0] + k < padding_shape[0] ||
                    i * stride_shape[0] + k >=
                        (input_shape[0] + padding_shape[0]) ||
                    j * stride_shape[1] + l < padding_shape[1] ||
                    j * stride_shape[1] + l >=
                        (input_shape[1] + padding_shape[1])) {
                    continue;
                }

                int kernel_idx =
                    f * kernel_shape[0] * kernel_shape[1] * input_shape[2] +
                    c * kernel_shape[0] * kernel_shape[1] +
                    k * kernel_shape[1] + l;
                int inputIndex = c * input_shape[0] * input_shape[1] +
                                 (i * stride_shape[0] + k - padding_shape[0]) *
                                     input_shape[1] +
                                 (j * stride_shape[1] + l - padding_shape[1]);

                sum += d_kernel[kernel_idx] * d_input[inputIndex];
            }
        }
    }

    d_output[f * output_shape[0] * output_shape[1] + i * output_shape[1] + j] =
        sum + d_bias[f];
}