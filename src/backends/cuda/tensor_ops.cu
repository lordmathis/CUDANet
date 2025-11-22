#include <iostream>

#include "backend.hpp"
#include "backend/cuda.cuh"
#include "kernels/matmul.cuh"

using namespace CUDANet::Backend;

void CUDA::print(const CUDANet::Tensor &input) {
    auto length = input.numel();
    std::vector<float> h_vec(input.numel());

    CUDA_CHECK(cudaMemcpy(
        h_vec.data(), input.data<float>(), sizeof(float) * length, cudaMemcpyDeviceToHost
    ));

    for (int i = 0; i < length; ++i) {
        std::cout << h_vec[i] << ", ";
    }

    std::cout << std::endl;
}

void CUDA::zero(CUDANet::Tensor &input) {
    fill(input, 0);
}

void CUDA::fill(CUDANet::Tensor &input, int value) {
    CUDA_CHECK(cudaMemset(input.data<float>(), value, sizeof(float) * input.numel()));

}

void CUDA::copy_to_device(CUDANet::Tensor &tensor, void *data, size_t size) {
    CUDA_CHECK(cudaMemcpy(tensor.data<float>(), data, size, cudaMemcpyHostToDevice));
}

void CUDA::sum(const CUDANet::Tensor &input, CUDANet::Tensor &sum) {
    auto length = input.numel();
    const int gridSize = ( + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDANet::Kernels::sum_reduce<<<gridSize, BLOCK_SIZE>>>(
        input.data<float>(), sum.data<float>(), length
    );
    CUDA_CHECK(cudaGetLastError());

    int remaining = gridSize;
    while (remaining > 1) {
        int blocks_needed = (remaining + BLOCK_SIZE - 1) / BLOCK_SIZE;
        CUDANet::Kernels::sum_reduce<<<blocks_needed, BLOCK_SIZE>>>(sum.data<float>(), sum.data<float>(), remaining);
        CUDA_CHECK(cudaGetLastError());

        remaining = blocks_needed;
    }
}

void CUDA::max(const CUDANet::Tensor &input, CUDANet::Tensor &max) {
    auto length = input.numel();
    const int grid_size = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;

    Kernels::max_reduce<<<grid_size, BLOCK_SIZE>>>(input.data<float>(), max.data<float>(), length);
    CUDA_CHECK(cudaGetLastError());

    int remaining = grid_size;

    while (remaining > 1) {
        int blocks_needed = (remaining + BLOCK_SIZE - 1) / BLOCK_SIZE;
        CUDANet::Kernels::max_reduce<<<blocks_needed, BLOCK_SIZE>>>(max.data<float>(), max.data<float>(), remaining);
        CUDA_CHECK(cudaGetLastError());

        remaining = blocks_needed;
    }
}
