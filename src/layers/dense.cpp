#include <format>
#include <stdexcept>

#include "layers/dense.hpp"

using namespace CUDANet::Layers;

Dense::Dense(CUDANet::Shape in_shape, CUDANet::Shape out_shape, CUDANet::Backend* backend)
    : Dense(in_shape, out_shape, backend->get_default_dtype(), backend) {}

Dense::Dense(CUDANet::Shape in_shape, CUDANet::Shape out_shape, CUDANet::DType dtype, CUDANet::Backend* backend)
    : backend(backend),
      in_shape(in_shape),
      out_shape(out_shape) {

    if (in_shape.size() != 1) {
        throw InvalidShapeException("input", 1, in_shape.size());
    }

    if (out_shape.size() != 1) {
        throw InvalidShapeException("output", 1, out_shape.size());
    }

    this->dtype = dtype;

    weights = CUDANet::Tensor(Shape{out_shape[0], in_shape[0]}, dtype, backend);
    biases = CUDANet::Tensor(Shape{out_shape[0]}, dtype, backend);
    output = CUDANet::Tensor(Shape{out_shape[0]}, dtype, backend);

    weights.zero();
    biases.zero();
    output.zero();
}

Dense::~Dense() {}

CUDANet::Tensor& Dense::forward(CUDANet::Tensor& input) {
    output.zero();
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

// TODO: Use dtype
void Dense::set_weights(void* input) {
    weights.set_data(input);
}

size_t Dense::get_weights_size() {
    return weights.size();
}

void Dense::set_biases(void* input) {
    biases.set_data(input);
}

size_t Dense::get_biases_size() {
    return biases.size();
}