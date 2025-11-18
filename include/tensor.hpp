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

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    ~Tensor();

    size_t size() const;
    size_t numel() const;

    template <typename T>
    const T* data() const {
        return static_cast<T*>(d_ptr);
    }

    template <typename T>
    T* data() {
        return static_cast<T*>(d_ptr);
    }

    void zero();

    template <typename T>
    void set_data(T *data) {
        backend->copy_to_device(*this, data, total_size);
    }

private:
    Shape       shape;
    DType       dtype;

    size_t total_elms;
    size_t total_size;

    CUDANet::Backend*   backend;
    void*       d_ptr;
};

} // namespace CUDANet