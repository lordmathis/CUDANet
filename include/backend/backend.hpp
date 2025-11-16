#pragma once

#include <cstddef>

namespace CUDANet::Backend
{   

class IBackend
{
public:

    // Memory management
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;

    // Layer operations
    virtual void relu(CUDANet::Backend::Tensor &tensor) = 0;
    virtual void sigmoid(CUDANet::Backend::Tensor &tensor) = 0;
    virtual void softmax(CUDANet::Backend::Tensor &tensor, CUDANet::Backend::Tensor &temp_max, CUDANet::Backend::Tensor &temp_sum) = 0;
};

} // namespace CUDANet::Backend