#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <format>

#include "backend/cuda/cuda.cuh"

using namespace CUDANet::Backends;


CUDA::CUDA(const BackendConfig& config) {
    device_id = config.device_id < 0 ? 0 : config.device_id;
    initialize();
}

bool CUDA::is_cuda_available() {
    int device_count;
    cudaError_t result = cudaGetDeviceCount(&device_count);
    
    // Return false instead of crashing
    if (result != cudaSuccess || device_count == 0) {
        return false;
    }
    return true;
}

void CUDA::initialize() {

    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_id >= device_count) {
        throw std::runtime_error(std::format("Invalid device id {}, only {} devices available", device_id, device_count));
    }

    CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device_id));

    std::printf("Using CUDA device %d: %s\n", device_id, deviceProp.name);
}

void* CUDA::allocate(size_t bytes) {
    void* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
    return d_ptr;
}

void CUDA::deallocate(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}
