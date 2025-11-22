#pragma once

#include <cstdio>

#include "backend.hpp"
#include "tensor.hpp"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif // BLOCK_SIZE

/**
 * @brief CUDA error checking macro
 * 
 */
#define CUDA_CHECK(call) \
do { \
    cudaError_t result = call; \
    if (result != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", \
                __FILE__, __LINE__, static_cast<unsigned int>(result), \
                cudaGetErrorString(result), #call); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

namespace CUDANet::Backend {

class CUDA : public Backend {
  public:
    // Memory management
    void* allocate(size_t bytes) override;
    void  deallocate(void* ptr) override;

    // Tensor ops
    void print(const CUDANet::Tensor& input) override;
    void zero(CUDANet::Tensor& input) override;
    void fill(CUDANet::Tensor &input, int value) override;
    void
    copy_to_device(CUDANet::Tensor& tensor, void* data, size_t size) override;
    void sum(const CUDANet::Tensor& input, CUDANet::Tensor& sum) override;
    void max(const CUDANet::Tensor& input, CUDANet::Tensor& max) override;

    // Layer ops
    void relu(CUDANet::Tensor& tensor) override;
    void sigmoid(CUDANet::Tensor& tensor) override;
    void softmax(
        CUDANet::Tensor& tensor,
        CUDANet::Tensor& temp_max,
        CUDANet::Tensor& temp_sum
    ) override;

    CUDANet::Tensor& dense(
        const CUDANet::Tensor& weights,
        const CUDANet::Tensor& biases,
        const CUDANet::Tensor& input,
        CUDANet::Tensor& output,
        const size_t           input_size,
        const size_t           output_size
    ) override;

    CUDANet::Tensor& conv2d(
        const CUDANet::Tensor& weights,
        const CUDANet::Tensor& biases,
        const CUDANet::Tensor& input,
        CUDANet::Tensor& output,
        const CUDANet::Shape in_shape,
        const CUDANet::Shape padding_shape,
        const CUDANet::Shape kernel_shape,
        const CUDANet::Shape stride_shape,
        const CUDANet::Shape out_shape
    ) override;

    CUDANet::Tensor& max_pool2d(
        const CUDANet::Tensor& input,
        CUDANet::Tensor& output,
        CUDANet::Shape input_shape,
        CUDANet::Shape pool_shape,
        CUDANet::Shape stride_shape,
        CUDANet::Shape padding_shape,
        CUDANet::Shape output_shape
    ) override;

    CUDANet::Tensor& avg_pool2d(
        const CUDANet::Tensor& input,
        CUDANet::Tensor& output,
        CUDANet::Shape input_shape,
        CUDANet::Shape pool_shape,
        CUDANet::Shape stride_shape,
        CUDANet::Shape padding_shape,
        CUDANet::Shape output_shape
    ) override;

    CUDANet::Tensor& batch_norm(
        const CUDANet::Tensor& input,
        CUDANet::Tensor& output,
        CUDANet::Shape input_shape,
        CUDANet::Tensor& weights,
        CUDANet::Tensor& biases,
        CUDANet::Tensor& running_mean,
        CUDANet::Tensor& running_var,
        CUDANet::Tensor& epsilon
    ) override;

    CUDANet::Tensor& concat(
        CUDANet::Tensor& input_a,
        CUDANet::Tensor& input_b,
        CUDANet::Tensor& output
    ) override;

    CUDANet::Tensor& add(
        CUDANet::Tensor& input_a,
        CUDANet::Tensor& input_b,
        CUDANet::Tensor& output
    ) override;
};

}  // namespace CUDANet::Backend