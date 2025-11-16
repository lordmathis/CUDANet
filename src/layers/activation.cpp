#include <stdexcept>
#include <vector>

#include "activation.hpp"
#include "backend/tensor.hpp"

using namespace CUDANet::Layers;

Activation::Activation(ActivationType activation, const int length)
    : activationType(activation), length(length) {


    if (activationType == SOFTMAX) {
      softmax_sum = CUDANet::Backend::Tensor({static_cast<size_t>(length)}, CUDANet::Backend::DType::FLOAT32, nullptr);
      tensor_max = CUDANet::Backend::Tensor({static_cast<size_t>(length)}, CUDANet::Backend::DType::FLOAT32, nullptr);
    }
}

void Activation::activate(CUDANet::Backend::Tensor input) {
    switch (activationType)
    {
    case ActivationType::SIGMOID:
        backend->sigmoid(input);
        break;
    case ActivationType::RELU:
        /* code */
        break;
    case ActivationType::SOFTMAX:
        /* code */
        break;
    default:
        break;
    }
}