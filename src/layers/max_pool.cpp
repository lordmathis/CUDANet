#include "max_pool.hpp"

#include <stdexcept>

using namespace CUDANet::Layers;

MaxPool2d::MaxPool2d(
    CUDANet::Shape    input_shape,
    CUDANet::Shape    pooling_shape,
    CUDANet::Shape    stride_shape,
    CUDANet::Shape    padding_shape,
    CUDANet::Backend* backend
)
    : in_shape(input_shape),
      pooling_shape(pooling_shape),
      stride_shape(stride_shape),
      padding_shape(padding_shape),
      backend(backend) {
    size_t out_h = (in_shape[0] + 2 * padding_shape[0] - pooling_shape[0]) /
                       stride_shape[0] +
                   1;
    size_t out_w = (in_shape[1] + 2 * padding_shape[1] - pooling_shape[1]) /
                       stride_shape[1] +
                   1;

    out_shape.resize(3);
    out_shape[0] = out_h;
    out_shape[1] = out_w;
    out_shape[2] = in_shape[2];

    output = CUDANet::Tensor(
        Shape{out_shape[0] * out_shape[1] * out_shape[3]},
        CUDANet::DType::FLOAT32, backend
    );
}

MaxPool2d::~MaxPool2d() {}

CUDANet::Tensor& MaxPool2d::forward(CUDANet::Tensor& input) {
    output.zero();
    backend->maxPool2d(
        input, output, in_shape, pooling_shape, stride_shape, padding_shape,
        out_shape
    );
    return output;
}

CUDANet::Shape MaxPool2d::input_shape() {
    return in_shape;
}

CUDANet::Shape MaxPool2d::output_shape() {
    return out_shape;
}

size_t MaxPool2d::input_size() {
    return sizeof(float) * in_shape[0] * in_shape[1] * in_shape[2];
}

size_t MaxPool2d::output_size() {
    return sizeof(float) * out_shape[0] * out_shape[1] * out_shape[2];
}

void MaxPool2d::set_weights(void* input) {}

CUDANet::Tensor& MaxPool2d::get_weights() {}

void MaxPool2d::set_biases(void* input) {}

CUDANet::Tensor& MaxPool2d::get_biases() {}