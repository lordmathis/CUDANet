#pragma once

#include <cstddef>
#include "backend/tensor.hpp"

namespace CUDANet::Backend
{   

class IBackend
{
public:

    // Memory management
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;

    // Tensor ops
    virtual void print(const CUDANet::Backend::Tensor &input) = 0;
    virtual void clear(CUDANet::Backend::Tensor &input) = 0;
    virtual void sum(const CUDANet::Backend::Tensor &input, CUDANet::Backend::Tensor &sum) = 0;
    virtual void max(const CUDANet::Backend::Tensor &input, CUDANet::Backend::Tensor &max) = 0;

    // Layer ops
    virtual void relu(CUDANet::Backend::Tensor &tensor) = 0;
    virtual void sigmoid(CUDANet::Backend::Tensor &tensor) = 0;
    virtual void softmax(CUDANet::Backend::Tensor &tensor, CUDANet::Backend::Tensor &temp_max, CUDANet::Backend::Tensor &temp_sum) = 0;
};

} // namespace CUDANet::Backend