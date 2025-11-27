#include <stdexcept>

#include "tensor.hpp"

using namespace CUDANet;

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

Tensor::Tensor(Shape shape, CUDANet::Backend* backend)
    : Tensor(shape, backend->get_default_dtype(), backend) {}

Tensor::Tensor(Shape shape, DType dtype, Backend* backend)
    : shape(shape), dtype(dtype), backend(backend), d_ptr(nullptr) {
    if (shape.empty()) {
        throw std::runtime_error("Tensor shape cannot be empty");
    }

    // Check if backend supports DType
    if (!backend->supports_dtype(dtype)) {
        throw std::runtime_error("Unsupported DType");
    }

    // Count total elements
    size_t count = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        count *= shape[i];
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

Tensor::Tensor(Tensor&& other) noexcept
    : shape(std::move(other.shape)),
      dtype(other.dtype),
      total_elms(other.total_elms),
      total_size(other.total_size),
      backend(other.backend),
      d_ptr(other.d_ptr) {
    other.d_ptr   = nullptr;
    other.backend = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        // Clean up our current resources
        if (d_ptr != nullptr && backend != nullptr) {
            backend->deallocate(d_ptr);
        }

        // Steal other's resources
        shape      = std::move(other.shape);
        dtype      = other.dtype;
        total_elms = other.total_elms;
        total_size = other.total_size;
        backend    = other.backend;
        d_ptr      = other.d_ptr;

        // Leave other in valid but empty state
        other.d_ptr   = nullptr;
        other.backend = nullptr;
    }
    return *this;
}

Tensor::~Tensor() {
    if (backend && d_ptr) {
        backend->deallocate(d_ptr);
        d_ptr = nullptr;
    }
}

DType Tensor::get_dtype() const {
    return dtype;
}

size_t Tensor::numel() const {
    return total_elms;
}

size_t Tensor::size() const {
    return total_size;
}

void* Tensor::device_ptr() const {
    return d_ptr;
}

void* Tensor::device_ptr() {
    return d_ptr;
}

void Tensor::zero() {
    backend->zero(*this);
}

void Tensor::fill(int value) {
    backend->fill(*this, value);
}

void Tensor::set_data(void *data) {
    backend->copy_to_device(*this, data, total_size);
}
