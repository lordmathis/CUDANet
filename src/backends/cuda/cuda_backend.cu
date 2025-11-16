#include "backend/cuda_backend.cuh"
#include "utils/cuda_helper.cuh"
#include "kernels/activation_functions.cuh"
#include "kernels/matmul.cuh"
#include "utils/vector.cuh"

using namespace CUDANet::Backend;

void *CUDABackend::allocate(size_t bytes) {
    void* devicePtr = nullptr;
    CUDA_CHECK(cudaMalloc(&devicePtr, bytes));
    return devicePtr;
}

void CUDABackend::deallocate(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

// void CUDABackend::copyToDevice(void* devicePtr, const void* hostPtr, size_t bytes) {
//     CUDA_CHECK(cudaMemcpy(devicePtr, hostPtr, bytes, cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaDeviceSynchronize());
// }

// void CUDABackend::copyToHost(void* hostPtr, const void* devicePtr, size_t bytes) {
//     CUDA_CHECK(cudaMemcpy(hostPtr, devicePtr, bytes, cudaMemcpyDeviceToHost));
//     CUDA_CHECK(cudaDeviceSynchronize());
// }

void CUDABackend::relu(Tensor &tensor) {
    int gridSize = (tensor.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    Kernels::relu<<<gridSize, BLOCK_SIZE>>>((float*)tensor.data(), (float*)tensor.data(), tensor.numel());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CUDABackend::sigmoid(Tensor &tensor) {
    int gridSize = (tensor.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    Kernels::sigmoid<<<gridSize, BLOCK_SIZE>>>((float*)tensor.data(), (float*)tensor.data(), tensor.numel());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CUDABackend::softmax(Tensor &tensor, Tensor &temp_max, Tensor &temp_sum) {
    int gridSize = (tensor.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Find max value
    Utils::max(tensor, temp_max, tensor.numel());

    // Subtract max value to improve numerical stability
    Kernels::vec_scalar_sub<<<gridSize, BLOCK_SIZE>>>(
        (float*)tensor.data(), (float*)tensor.data(), (float*)temp_max.data(), tensor.numel()
    );
    CUDA_CHECK(cudaGetLastError());

    // Compute exponentials
    Kernels::vec_exp<<<gridSize, BLOCK_SIZE>>>(
        (float*)tensor.data(), (float*)tensor.data(), tensor.numel()
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Find sum
    Utils::sum(tensor, temp_sum, tensor.numel());

    Kernels::vec_scalar_div<<<gridSize, BLOCK_SIZE>>>(
        (float*)tensor.data(), (float*)tensor.data(), (float*)temp_sum.data(), tensor.numel()
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
