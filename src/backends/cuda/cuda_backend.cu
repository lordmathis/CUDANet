#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cuda_helper.cuh>

#include "backend/cuda.cuh"

cudaDeviceProp initializeCUDA() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::fprintf(stderr, "No CUDA devices found. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device));

    std::printf("Using CUDA device %d: %s\n", device, deviceProp.name);

    return deviceProp;
}

using namespace CUDANet::Backend;

void* CUDABackend::allocate(size_t bytes) {
    void* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
    return d_ptr;
}

void CUDABackend::deallocate(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}
