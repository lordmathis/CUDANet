#include <stdexcept>

#include "backend/tensor.hpp"

using namespace CUDANet::Backend;

Tensor::Tensor(Shape shape, DType dtype, IBackend* backend)
    : shape(shape), dtype(dtype), backend(backend), d_ptr(nullptr) {}

Tensor::~Tensor() {
    deallocate();
}

size_t Tensor::numel() const {
    size_t totalElements = 1;
    for (const auto& dim : shape) {
        totalElements *= dim;
    }
    return totalElements;
}

size_t Tensor::size() const {
    size_t totalSize = numel();

    size_t typeSize = 0;
    switch (dtype) {
        case DType::FLOAT32:
            typeSize = 4;
            break;
        default:
            throw std::runtime_error("Unsupported data type");
    }

    return totalSize * typeSize;
}

template <typename T>
const T* Tensor::data() const {
    return static_cast<T*>(d_ptr);
}

template <typename T>
T* Tensor::data() {
    return static_cast<T*>(d_ptr);
}