#include "batch_norm.hpp"

#include <stdexcept>
#include <vector>

#include "activation.hpp"
#include "layer.hpp"

using namespace CUDANet::Layers;

BatchNorm2d::BatchNorm2d(
    CUDANet::Shape input_shape,
    float          eps,
    CUDANet::Backend *backend
)
    : BatchNorm2d(input_shape, eps, backend->get_default_dtype(), backend) {}

BatchNorm2d::BatchNorm2d(
    CUDANet::Shape input_shape,
    float          eps,
    CUDANet::DType dtype,
    CUDANet::Backend *backend
)
    : in_shape(input_shape), backend(backend)  {

    if (in_shape.size() != 3) {
        throw InvalidShapeException("input", 3, in_shape.size());
    }

    this->dtype = dtype;

    epsilon = CUDANet::Tensor({1}, dtype, backend);
    epsilon.set_data<float>(&eps);

    running_mean = CUDANet::Tensor({in_shape[2]}, dtype, backend);
    running_mean.zero();

    running_var = CUDANet::Tensor({in_shape[2]}, dtype, backend);
    running_var.fill(1);

    weights = CUDANet::Tensor({in_shape[2]}, dtype, backend);
    weights.fill(1);

    biases = CUDANet::Tensor({in_shape[2]}, dtype, backend);
    biases.zero();

    output = CUDANet::Tensor(in_shape, dtype, backend);
}

BatchNorm2d::~BatchNorm2d() {}

CUDANet::Tensor& BatchNorm2d::forward(CUDANet::Tensor& input) {
    output.zero();
    backend->batch_norm(
        input,
        output,
        in_shape,
        weights,
        biases,
        running_mean,
        running_var,
        epsilon
    );
    return output;
}

CUDANet::Shape BatchNorm2d::input_shape() {
    return in_shape;
}

CUDANet::Shape BatchNorm2d::output_shape() {
    return in_shape;
}

size_t BatchNorm2d::input_size() {
    return sizeof(float) * in_shape[0] * in_shape[1] * in_shape[2];
}

size_t BatchNorm2d::output_size() {
    return sizeof(float) * in_shape[0] * in_shape[1] * in_shape[2];
}

void BatchNorm2d::set_weights(void* input) {
    weights.set_data<float>(static_cast<float*>(input));
}

size_t BatchNorm2d::get_weights_size() {
    return weights.size();
}

void BatchNorm2d::set_biases(void* input) {
    biases.set_data<float>(static_cast<float*>(input));
}

size_t BatchNorm2d::get_biases_size() {
    return biases.size();
}

void BatchNorm2d::set_running_mean(void* input) {
    running_mean.set_data<float>(static_cast<float*>(input));
}

size_t BatchNorm2d::get_running_mean_size() {
    return running_mean.size();
}

void BatchNorm2d::set_running_var(void* input) {
    running_var.set_data<float>(static_cast<float*>(input));
}

size_t BatchNorm2d::get_running_var_size() {
    return running_var.size();
}