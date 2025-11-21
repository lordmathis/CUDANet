#include "dense.hpp"

#include <format>
#include <stdexcept>

using namespace CUDANet::Layers;

Dense::Dense(CUDANet::Shape in_shape, CUDANet::Shape out_shape, CUDANet::Backend* backend)
    : backend(backend),
      in_shape(in_shape),
      out_shape(out_shape) {

    if (in_shape.size() != 1) {
        throw InvalidShapeException("input", 1, in_shape.size());
    }

    if (out_shape.size() != 1) {
        throw InvalidShapeException("output", 1, out_shape.size());
    }

    weights = CUDANet::Tensor(Shape{out_shape[0], in_shape[0]}, CUDANet::DType::FLOAT32, backend);
    biases = CUDANet::Tensor(Shape{out_shape[0]}, CUDANet::DType::FLOAT32, backend);
    output = CUDANet::Tensor(Shape{out_shape[0]}, CUDANet::DType::FLOAT32, backend);

    weights.zero();
    biases.zero();
    output.zero();
}

Dense::~Dense() {}

CUDANet::Tensor& Dense::forward(CUDANet::Tensor& input) {
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

void Dense::set_weights(void* input) {
    weights.set_data<float>(static_cast<float*>(input));
}

CUDANet::Tensor& Dense::get_weights() {
    return weights;
}

void Dense::set_biases(void* input) {
    biases.set_data<float>(static_cast<float*>(input));
}

CUDANet::Tensor& Dense::get_biases() {
    return biases;
}