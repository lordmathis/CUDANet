#include <stdexcept>

#include "backend/tensor.hpp"

using namespace CUDANet::Backend;

Tensor::Tensor(Shape shape, DType dtype, IBackend* backend)
    : shape(shape), dtype(dtype), backend(backend), devicePtr(nullptr), hostPtr(nullptr) {}

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

void* Tensor::data() const {
    return devicePtr;
}