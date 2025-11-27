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

size_t dtype_size(DType dtype) {
    switch (dtype)
    {
    case DType::FLOAT32:
        return 4;
        break;
    
    default:
        throw std::runtime_error("Unknown DType");
        break;
    }
}

class Backend;

class Tensor
{
public:

    Tensor() = default;
    Tensor(Shape shape, CUDANet::Backend* backend);
    Tensor(Shape shape, DType dtype, CUDANet::Backend* backend);

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    ~Tensor();

    DType get_dtype() const;

    size_t size() const;
    size_t numel() const;

    void* device_ptr() const;
    void* device_ptr();

    void zero();

    void fill(int value);

    void set_data(void *data);

private:
    Shape       shape;
    DType       dtype;

    size_t total_elms;
    size_t total_size;

    CUDANet::Backend*   backend;
    void*       d_ptr;
};

} // namespace CUDANet