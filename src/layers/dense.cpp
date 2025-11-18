#include <format>
#include <stdexcept>

#include "dense.hpp"

using namespace CUDANet::Layers;

Dense::Dense(CUDANet::Backend *backend, CUDANet::Shape input_shape, CUDANet::Shape output_shape)
    : backend(backend), in_shape(input_shape), out_shape(output_shape) {
    // Allocate memory for weights and biases

    if (input_shape.size() != 1) {
        throw std::runtime_error(std::format("Invalid shape. Expected [1], got {}", input_shape));
    }
    
    if (output_shape.size() != 1) {
        throw std::runtime_error(std::format("Invalid shape. Expected [1], got {}", output_shape));
    }

    auto input_len = input_shape[0];
    auto output_len = output_shape[0];

    auto weights = CUDANet::Tensor{Shape(input_len * output_len), CUDANet::DType::FLOAT32, backend};
    auto biases = CUDANet::Tensor(Shape(output_len), CUDANet::DType::FLOAT32, backend);
    auto output = CUDANet::Tensor(Shape(output_len), CUDANet::DType::FLOAT32, backend);

    weights.zero();
    biases.zero();
}

CUDANet::Tensor& Dense::forward(CUDANet::Tensor &input) {
    backend->dense(weights, biases, input, output, in_shape[0], out_shape[0]);
    return output;
}

CUDANet::Shape Dense::input_shape() {
    return in_shape;
}

CUDANet::Shape Dense::output_shape() {
    return out_shape;
}

size_t Dense::input_size() {
    return in_shape[0];
};

size_t Dense::output_size() {
    return out_shape[0];
};

void Dense::set_weights(void *input) {
    weights.set_data<float>(static_cast<float*>(input));
}

CUDANet::Tensor& Dense::get_weights() {
    return weights;
}

void Dense::set_biases(void *input) {
    biases.set_data<float>(static_cast<float*>(input));
}

CUDANet::Tensor& Dense::get_biases() {
    return biases;
}