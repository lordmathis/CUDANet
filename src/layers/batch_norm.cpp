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
    : in_shape(input_shape), backend(backend)  {

    if (in_shape.size() != 3) {
        throw InvalidShapeException("input", 3, in_shape.size());
    }

    epsilon = CUDANet::Tensor({1}, CUDANet::DType::FLOAT32, backend);
    epsilon.set_data<float>(&eps);

    running_mean = CUDANet::Tensor({in_shape[2]}, CUDANet::DType::FLOAT32, backend);
    running_mean.zero();

    running_var = CUDANet::Tensor({in_shape[2]}, CUDANet::DType::FLOAT32, backend);
    running_var.fill(1);

    weights = CUDANet::Tensor({in_shape[2]}, CUDANet::DType::FLOAT32, backend);
    weights.fill(1);

    biases = CUDANet::Tensor({in_shape[2]}, CUDANet::DType::FLOAT32, backend);
    biases.zero();

    output = CUDANet::Tensor(in_shape, CUDANet::DType::FLOAT32, backend);
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

CUDANet::Tensor& BatchNorm2d::get_weights() {
    return weights;
}

void BatchNorm2d::set_biases(void* input) {
    biases.set_data<float>(static_cast<float*>(input));
}

CUDANet::Tensor& BatchNorm2d::get_biases() {
    return biases;
}

void BatchNorm2d::set_running_mean(void* input) {
    running_mean.set_data<float>(static_cast<float*>(input));
}

CUDANet::Tensor& BatchNorm2d::get_running_mean() {
    return running_mean;
}

void BatchNorm2d::set_running_var(void* input) {
    running_var.set_data<float>(static_cast<float*>(input));
}

CUDANet::Tensor& BatchNorm2d::get_running_var() {
    return running_var;
}