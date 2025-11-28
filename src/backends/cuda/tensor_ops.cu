#include <iostream>

#include "backend.hpp"
#include "backend/cuda/cuda.cuh"
#include "backend/cuda/kernels/matmul.cuh"

using namespace CUDANet::Backends;

void CUDA::print(const CUDANet::Tensor &input) {
    switch (input.get_dtype()) {
    case DType::FLOAT32:
        print_impl<float>(input);
        break;

    default:
        throw std::runtime_error("Unsupported dtype");
        break;
    }
}

template void CUDA::print_impl<float> (const CUDANet::Tensor &input);

template <typename T>
void CUDA::print_impl(const CUDANet::Tensor &input) {
    auto length = input.numel();
    std::vector<T> h_vec(input.numel());

    CUDA_CHECK(cudaMemcpy(
        h_vec.data(), static_cast<const T*>(input.device_ptr()), sizeof(T) * length, cudaMemcpyDeviceToHost
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
    switch (input.get_dtype()) {
    case DType::FLOAT32:
        fill_impl<float>(input, value);
        break;

    default:
        throw std::runtime_error("Unsupported dtype");
        break;
    }
}

template void CUDA::fill_impl<float>(CUDANet::Tensor &input, int value);

template <typename T>
void CUDA::fill_impl(CUDANet::Tensor &input, int value) {
    CUDA_CHECK(cudaMemset(static_cast<T*>(input.device_ptr()), value, sizeof(T) * input.numel()));
}

void CUDA::copy_to_device(CUDANet::Tensor &tensor, void *data, size_t size) {
    switch (tensor.get_dtype()) {
    case DType::FLOAT32:
        copy_to_device_impl<float>(tensor, data, size);
        break;

    default:
        throw std::runtime_error("Unsupported dtype");
        break;
    }
}

template void CUDA::copy_to_device_impl<float>(CUDANet::Tensor &tensor, void *data, size_t size);

template <typename T>
void CUDA::copy_to_device_impl(CUDANet::Tensor &tensor, void *data, size_t size) {
    CUDA_CHECK(cudaMemcpy(static_cast<T*>(tensor.device_ptr()), data, size, cudaMemcpyHostToDevice));
}

void CUDA::sum(const CUDANet::Tensor &input, CUDANet::Tensor &sum) {
    switch (input.get_dtype()) {
    case DType::FLOAT32:
        sum_impl<float>(input, sum);
        break;

    default:
        throw std::runtime_error("Unsupported dtype");
        break;
    }
}

template void CUDA::sum_impl<float>(const CUDANet::Tensor &input, CUDANet::Tensor &sum);

template <typename T>
void CUDA::sum_impl(const CUDANet::Tensor &input, CUDANet::Tensor &sum) {
    auto length = input.numel();
    const int gridSize = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDANet::Kernels::sum_reduce<<<gridSize, BLOCK_SIZE>>>(
        static_cast<const T*>(input.device_ptr()), static_cast<T*>(sum.device_ptr()), length
    );
    CUDA_CHECK(cudaGetLastError());

    int remaining = gridSize;
    while (remaining > 1) {
        int blocks_needed = (remaining + BLOCK_SIZE - 1) / BLOCK_SIZE;
        CUDANet::Kernels::sum_reduce<<<blocks_needed, BLOCK_SIZE>>>(static_cast<T*>(sum.device_ptr()), static_cast<T*>(sum.device_ptr()), remaining);
        CUDA_CHECK(cudaGetLastError());

        remaining = blocks_needed;
    }
}

void CUDA::max(const CUDANet::Tensor &input, CUDANet::Tensor &max) {
    switch (input.get_dtype()) {
    case DType::FLOAT32:
        max_impl<float>(input, max);
        break;

    default:
        throw std::runtime_error("Unsupported dtype");
        break;
    }
}

template void CUDA::max_impl<float>(const CUDANet::Tensor &input, CUDANet::Tensor &max);

template <typename T>
void CUDA::max_impl(const CUDANet::Tensor &input, CUDANet::Tensor &max) {
    auto length = input.numel();
    const int grid_size = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;

    Kernels::max_reduce<<<grid_size, BLOCK_SIZE>>>(static_cast<const T*>(input.device_ptr()), static_cast<T*>(max.device_ptr()), length);
    CUDA_CHECK(cudaGetLastError());

    int remaining = grid_size;

    while (remaining > 1) {
        int blocks_needed = (remaining + BLOCK_SIZE - 1) / BLOCK_SIZE;
        CUDANet::Kernels::max_reduce<<<blocks_needed, BLOCK_SIZE>>>(static_cast<T*>(max.device_ptr()), static_cast<T*>(max.device_ptr()), remaining);
        CUDA_CHECK(cudaGetLastError());

        remaining = blocks_needed;
    }
}
