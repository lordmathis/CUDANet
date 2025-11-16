#pragma once
#include <cstddef>
#include "backend/backend.hpp"
#include <vector>

namespace CUDANet::Backend
{

enum class DType
{
    FLOAT32,
    // FLOAT16,  // Not implemented yet
    // INT32,  // Not implemented yet
};

typedef std::vector<size_t> Shape;

class Tensor
{
public:

    Tensor() = default;
    Tensor(Shape shape, DType dtype, IBackend* backend);
    ~Tensor();

    void* allocate();
    void deallocate();

    void toDevice(const void* hostPtr);
    void toHost(void* hostPtr);

    size_t size() const;
    size_t numel() const;
    void* data() const;

private:
    Shape       shape;
    DType       dtype;
    IBackend*   backend;
    void*       devicePtr;
    void*       hostPtr;
};

} // namespace CUDANet::Backend