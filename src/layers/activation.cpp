#include <stdexcept>
#include <vector>

#include "activation.hpp"
#include "backend/tensor.hpp"

using namespace CUDANet::Layers;

Activation::Activation(CUDANet::Backend::IBackend* backend, ActivationType activation, const int length)
    : backend(backend), activationType(activation), length(length) {


    if (activationType == SOFTMAX) {
      softmax_sum = CUDANet::Backend::Tensor({static_cast<size_t>(length)}, CUDANet::Backend::DType::FLOAT32, backend);
      tensor_max = CUDANet::Backend::Tensor({static_cast<size_t>(length)}, CUDANet::Backend::DType::FLOAT32, backend);
    }
}

void Activation::activate(CUDANet::Backend::Tensor input) {
    switch (activationType)
    {
    case ActivationType::SIGMOID:
        backend->sigmoid(input);
        break;
    case ActivationType::RELU:
        backend->relu(input);
        break;
    case ActivationType::SOFTMAX:
        backend->softmax(input, tensor_max, softmax_sum);
        break;
    default:
        break;
    }
}