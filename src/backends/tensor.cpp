#include <stdexcept>

#include "tensor.hpp"

using namespace CUDANet;

Tensor::Tensor(Shape shape, DType dtype, Backend* backend)
    : shape(shape), dtype(dtype), backend(backend), d_ptr(nullptr) {
    // Count total elements
    size_t count = 1;
    for (const auto& dim : shape) {
        count *= dim;
    }
    total_elms = count;

    // Compute total size (bytes)
    size_t type_size = 0;
    switch (dtype) {
        case DType::FLOAT32:
            type_size = 4;
            break;
        default:
            throw std::runtime_error("Unsupported data type");
    }
    total_size = total_elms * type_size;

    // Allocate memory on backend
    d_ptr = backend->allocate(total_size);
}

Tensor::~Tensor() {
    backend->deallocate(d_ptr);
    d_ptr = nullptr;
}

size_t Tensor::numel() const {
    return total_elms;
}

size_t Tensor::size() const {
    return total_size;
}

template <typename T>
const T* Tensor::data() const {
    return static_cast<T*>(d_ptr);
}

template <typename T>
T* Tensor::data() {
    return static_cast<T*>(d_ptr);
}

void Tensor::zero() {
    backend->zero(*this);
}
