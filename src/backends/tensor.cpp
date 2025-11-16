#include "backend/tensor.hpp"

using namespace CUDANet::Backend;

Tensor::Tensor(Shape shape, DType dtype, IBackend* backend)
    : shape(shape), dtype(dtype), backend(backend), devicePtr(nullptr), hostPtr(nullptr) {}

Tensor::~Tensor() {
    deallocate();
}

