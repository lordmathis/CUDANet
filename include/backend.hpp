#pragma once

#include <cstddef>

#include "tensor.hpp"

namespace CUDANet
{   

class Backend
{
public:

    // Memory management
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;

    // Tensor ops
    virtual void print(const CUDANet::Tensor &input) = 0;
    virtual void zero(CUDANet::Tensor &input) = 0;
    virtual void sum(const CUDANet::Tensor &input, CUDANet::Tensor &sum) = 0;
    virtual void max(const CUDANet::Tensor &input, CUDANet::Tensor &max) = 0;

    // Layer ops
    virtual void relu(CUDANet::Tensor &tensor) = 0;
    virtual void sigmoid(CUDANet::Tensor &tensor) = 0;
    virtual void softmax(CUDANet::Tensor &tensor, CUDANet::Tensor &temp_max, CUDANet::Tensor &temp_sum) = 0;
};

} // namespace CUDANet::Backend