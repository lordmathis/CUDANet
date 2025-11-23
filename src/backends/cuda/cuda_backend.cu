#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

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

using namespace CUDANet::Backends;

void* CUDA::allocate(size_t bytes) {
    void* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
    return d_ptr;
}

void CUDA::deallocate(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}
