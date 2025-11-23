#include <format>
#include <stdexcept>
#include <vector>

#include "activation.hpp"
#include "tensor.hpp"

using namespace CUDANet::Layers;

Activation::Activation(ActivationType activation, const CUDANet::Shape &shape, CUDANet::Backend* backend)
    : backend(backend), activationType(activation), shape(shape) {

    if (shape.size() != 1) {
        throw InvalidShapeException("input", 1, shape.size());
    }

    auto length = shape[0];

    if (activationType == SOFTMAX) {
      softmax_sum = CUDANet::Tensor({static_cast<size_t>(length)}, CUDANet::DType::FLOAT32, backend);
      tensor_max = CUDANet::Tensor({static_cast<size_t>(length)}, CUDANet::DType::FLOAT32, backend);
    }
}

CUDANet::Tensor& Activation::forward(CUDANet::Tensor &input) {
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

    return input;
}

CUDANet::Shape Activation::input_shape() {
    return shape;
}

CUDANet::Shape Activation::output_shape() {
    return shape;
}

size_t Activation::input_size() {
    return shape[0];
}

size_t Activation::output_size() {
    return shape[0];
}

void Activation::set_weights(void *input) {}

CUDANet::Tensor& Activation::get_weights() {}

void Activation::set_biases(void *input) {}

CUDANet::Tensor& Activation::get_biases() {}