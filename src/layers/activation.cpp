#include "activation.hpp"

#include <format>
#include <stdexcept>
#include <vector>

#include "tensor.hpp"

using namespace CUDANet::Layers;

Activation::Activation(
    ActivationType        activation,
    const CUDANet::Shape& shape,
    CUDANet::Backend*     backend
)
    : Activation(activation, shape, backend->get_default_dtype(), backend) {}

Activation::Activation(
    ActivationType        activation,
    const CUDANet::Shape& shape,
    CUDANet::DType        dtype,
    CUDANet::Backend*     backend
)
    : activation_type(activation),
      shape(shape),
      backend(backend) {
    this->dtype = dtype;
    
    if (shape.size() != 1) {
        throw InvalidShapeException("input", 1, shape.size());
    }

    auto length = shape[0];

    if (activation_type == SOFTMAX) {
        softmax_sum =
            CUDANet::Tensor({static_cast<size_t>(length)}, dtype, backend);
        tensor_max =
            CUDANet::Tensor({static_cast<size_t>(length)}, dtype, backend);
    }
}

CUDANet::Tensor& Activation::forward(CUDANet::Tensor& input) {
    switch (activation_type) {
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

void Activation::set_weights(void* input) {}

size_t Activation::get_weights_size() {
    return 0;
}

void Activation::set_biases(void* input) {}

size_t Activation::get_biases_size() {
    return 0;
}