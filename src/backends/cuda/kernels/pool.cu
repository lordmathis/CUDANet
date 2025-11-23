#include "layer.hpp"
#include "pool.cuh"

using namespace CUDANet;

__global__ void Kernels::max_pool(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    const Shape input_shape,
    const Shape output_shape,
    const Shape pool_shape,
    const Shape stride_shape,
    const Shape padding_shape
) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= output_shape[0] || j >= output_shape[1] || c >= output_shape[2]) {
        return;
    }

    float max = 0.0f;

    for (int k = 0; k < pool_shape[0]; k++) {
        for (int l = 0; l < pool_shape[1]; l++) {
            int inputRow = i * stride_shape[0] + k - padding_shape[0];
            int inputCol = j * stride_shape[1] + l - padding_shape[1];

            if (inputRow >= 0 && inputRow < input_shape[0] && inputCol >= 0 &&
                inputCol < input_shape[1]) {
                int inputIndex = c * input_shape[0] * input_shape[1] +
                                 inputRow * input_shape[1] + inputCol;
                if (d_input[inputIndex] > max) {
                    max = d_input[inputIndex];
                }
            }
        }
    }

    d_output
        [c * output_shape[0] * output_shape[1] + i * output_shape[1] + j] =
            max;
}

__global__ void Kernels::avg_pool(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    const Shape input_shape,
    const Shape output_shape,
    const Shape pool_shape,
    const Shape stride_shape,
    const Shape padding_shape
) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= output_shape[0] || j >= output_shape[1] || c >= output_shape[2]) {
        return;
    }

    float sum = 0.0f;

    for (int k = 0; k < pool_shape[0]; k++) {
        for (int l = 0; l < pool_shape[1]; l++) {
            int inputRow = i * stride_shape[0] + k - padding_shape[0];
            int inputCol = j * stride_shape[1] + l - padding_shape[1];

            if (inputRow >= 0 && inputRow < input_shape[0] && inputCol >= 0 &&
                inputCol < input_shape[1]) {
                int inputIndex = c * input_shape[0] * input_shape[1] +
                                 inputRow * input_shape[1] + inputCol;
                sum += d_input[inputIndex];
            }
        }
    }

    d_output
        [c * output_shape[0] * output_shape[1] + i * output_shape[1] + j] =
            sum / (pool_shape[0] * pool_shape[1]);
}