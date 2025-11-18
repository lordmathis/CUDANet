#include "backend/cuda.cuh"
#include "kernels/activation_functions.cuh"
#include "kernels/matmul.cuh"
#include "utils/cuda_helper.cuh"

using namespace CUDANet::Backend;

void CUDA::relu(Tensor& tensor) {
    int gridSize = (tensor.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    Kernels::relu<<<gridSize, BLOCK_SIZE>>>(
        tensor.data<float>(), tensor.data<float>(), tensor.numel()
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CUDA::sigmoid(Tensor& tensor) {
    int gridSize = (tensor.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    Kernels::sigmoid<<<gridSize, BLOCK_SIZE>>>(
        tensor.data<float>(), tensor.data<float>(), tensor.numel()
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CUDA::softmax(Tensor& tensor, Tensor& temp_max, Tensor& temp_sum) {
    int gridSize = (tensor.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Find max value
    max(tensor, temp_max);

    // Subtract max value to improve numerical stability
    Kernels::vec_scalar_sub<<<gridSize, BLOCK_SIZE>>>(
        tensor.data<float>(), tensor.data<float>(), temp_max.data<float>(),
        tensor.numel()
    );
    CUDA_CHECK(cudaGetLastError());

    // Compute exponentials
    Kernels::vec_exp<<<gridSize, BLOCK_SIZE>>>(
        tensor.data<float>(), tensor.data<float>(), tensor.numel()
    );
    CUDA_CHECK(cudaGetLastError());

    // Find sum
    sum(tensor, temp_sum);

    Kernels::vec_scalar_div<<<gridSize, BLOCK_SIZE>>>(
        tensor.data<float>(), tensor.data<float>(), temp_sum.data<float>(),
        tensor.numel()
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

CUDANet::Tensor& CUDA::dense(
    const CUDANet::Tensor& weights,
    const CUDANet::Tensor& biases,
    const CUDANet::Tensor& input,
    CUDANet::Tensor& output,
    const size_t           input_size,
    const size_t           output_size
) {
    auto forwardGridSize =
        (std::max(input_size, output_size) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    auto biasGridSize = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    Kernels::mat_vec_mul<<<forwardGridSize, BLOCK_SIZE>>>(
        weights.data<float>(), input.data<float>(), output.data<float>(),
        input_size, output_size
    );
    CUDA_CHECK(cudaGetLastError());

    Kernels::vec_vec_add<<<biasGridSize, BLOCK_SIZE>>>(
        biases.data<float>(), output.data<float>(), output.data<float>(),
        output_size
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}