#pragma once

#include <cstddef>
#include <vector>

#include "backend.hpp"
#include "shape.hpp"

namespace CUDANet
{

enum class DType
{
    FLOAT32,
    // FLOAT16,  // Not implemented yet
    // INT32,  // Not implemented yet
};

class Tensor
{
public:

    Tensor() = default;
    Tensor(Shape shape, DType dtype, CUDANet::Backend* backend);
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

    CUDANet::Backend*   backend;
    void*       d_ptr;
};

} // namespace CUDANet