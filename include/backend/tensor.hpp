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

    size_t size() const;
    size_t numel() const;

    template <typename T>
    const T* data() const;

    template <typename T>
    T* data();

private:
    Shape       shape;
    DType       dtype;

    size_t total_elms;
    size_t total_size;

    IBackend*   backend;
    void*       d_ptr;
};

} // namespace CUDANet::Backend