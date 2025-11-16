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

    virtual void copyToDevice(void* devicePtr, const void* hostPtr, size_t bytes) = 0;
    virtual void copyToHost(void* hostPtr, const void* devicePtr, size_t bytes) = 0;

};

} // namespace CUDANet::Backend