#pragma once

#include <cstddef>

#include "tensor.hpp"

namespace CUDANet {

class Backend {
  public:
    // Memory management
    virtual void* allocate(size_t bytes) = 0;
    virtual void  deallocate(void* ptr)  = 0;

    // Tensor ops
    virtual void print(const CUDANet::Tensor& input) = 0;
    virtual void zero(CUDANet::Tensor& input)        = 0;

    virtual void
    copy_to_device(CUDANet::Tensor& tensor, void* data, size_t size) = 0;

    virtual void sum(const CUDANet::Tensor& input, CUDANet::Tensor& sum) = 0;
    virtual void max(const CUDANet::Tensor& input, CUDANet::Tensor& max) = 0;

    // Layer ops
    virtual void relu(CUDANet::Tensor& tensor)    = 0;
    virtual void sigmoid(CUDANet::Tensor& tensor) = 0;
    virtual void softmax(
        CUDANet::Tensor& tensor,
        CUDANet::Tensor& temp_max,
        CUDANet::Tensor& temp_sum
    ) = 0;

    virtual CUDANet::Tensor& dense(
        const CUDANet::Tensor& weights,
        const CUDANet::Tensor& biases,
        const CUDANet::Tensor& input,
        CUDANet::Tensor& output,
        const size_t           input_size,
        const size_t           output_size
    ) = 0;
};

}  // namespace CUDANet