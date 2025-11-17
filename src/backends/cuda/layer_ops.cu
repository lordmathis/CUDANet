#include "backend/cuda.cuh"
#include "utils/cuda_helper.cuh"
#include "kernels/activation_functions.cuh"
#include "kernels/matmul.cuh"

using namespace CUDANet::Backend;

void CUDABackend::relu(Tensor &tensor) {
    int gridSize = (tensor.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    Kernels::relu<<<gridSize, BLOCK_SIZE>>>(tensor.data<float>(), tensor.data<float>(), tensor.numel());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CUDABackend::sigmoid(Tensor &tensor) {
    int gridSize = (tensor.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    Kernels::sigmoid<<<gridSize, BLOCK_SIZE>>>(tensor.data<float>(), tensor.data<float>(), tensor.numel());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CUDABackend::softmax(Tensor &tensor, Tensor &temp_max, Tensor &temp_sum) {
    int gridSize = (tensor.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Find max value
    max(tensor, temp_max);

    // Subtract max value to improve numerical stability
    Kernels::vec_scalar_sub<<<gridSize, BLOCK_SIZE>>>(
        tensor.data<float>(), tensor.data<float>(), temp_max.data<float>(), tensor.numel()
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
        tensor.data<float>(), tensor.data<float>(), temp_sum.data<float>(), tensor.numel()
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}