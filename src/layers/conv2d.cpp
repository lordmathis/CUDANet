#include "conv2d.hpp"

#include <format>
#include <stdexcept>

#include "layer.hpp"
#include "tensor.hpp"

using namespace CUDANet::Layers;

Conv2d::Conv2d(
    CUDANet::Shape    input_shape,
    CUDANet::Shape    kernel_shape,
    CUDANet::Shape    stride_shape,
    CUDANet::Shape    padding_shape,
    CUDANet::Backend* backend
)
    : Conv2d(input_shape, kernel_shape, stride_shape, padding_shape, backend->get_default_dtype(), backend) {}

Conv2d::Conv2d(
    CUDANet::Shape    input_shape,
    CUDANet::Shape    kernel_shape,
    CUDANet::Shape    stride_shape,
    CUDANet::Shape    padding_shape,
    CUDANet::DType    dtype,
    CUDANet::Backend* backend
)
    : in_shape(input_shape),
      kernel_shape(kernel_shape),
      stride_shape(stride_shape),
      padding_shape(padding_shape),
      backend(backend) {
    if (in_shape.size() != 3) {
        throw InvalidShapeException("input", 3, in_shape.size());
    }

    if (kernel_shape.size() != 3) {
        throw InvalidShapeException("kernel", 3, kernel_shape.size());
    }

    if (stride_shape.size() != 2) {
        throw InvalidShapeException("stride", 3, stride_shape.size());
    }

    if (padding_shape.size() != 2) {
        throw InvalidShapeException("padding", 3, padding_shape.size());
    }

    this->dtype = dtype;

    out_shape = {
        (in_shape[0] - kernel_shape[0] + 2 * padding_shape[0]) /
                stride_shape[0] +
            1,
        (in_shape[1] - kernel_shape[1] + 2 * padding_shape[1]) /
                stride_shape[1] +
            1,
        kernel_shape[2]
    };

    output = CUDANet::Tensor(
        Shape{out_shape[0], out_shape[1], out_shape[2]},
        dtype, backend
    );

    weights = CUDANet::Tensor(
        Shape{
            kernel_shape[0], kernel_shape[1], kernel_shape[2], in_shape[2]
        },
        dtype, backend
    );
    biases = CUDANet::Tensor(
        Shape{kernel_shape[2]}, dtype, backend
    );

    weights.zero();
    biases.zero();
}

Conv2d::~Conv2d() {}

CUDANet::Tensor& Conv2d::forward(CUDANet::Tensor& input) {
    output.zero();
    backend->conv2d(
        weights, biases, input, output, in_shape, padding_shape, kernel_shape,
        stride_shape, out_shape
    );
    return output;
}

CUDANet::Shape Conv2d::input_shape() {
    return in_shape;
}

CUDANet::Shape Conv2d::output_shape() {
    return out_shape;
}

size_t Conv2d::input_size() {
    return sizeof(float) * in_shape[0] * in_shape[1] * in_shape[2];
}

size_t Conv2d::output_size() {
    return sizeof(float) * out_shape[0] * out_shape[1] * out_shape[2];
}

void Conv2d::set_weights(void* input) {
    weights.set_data<float>(static_cast<float*>(input));
}

size_t Conv2d::get_weights_size() {
    return weights.size();
}

void Conv2d::set_biases(void* input) {
    biases.set_data<float>(static_cast<float*>(input));
}

size_t Conv2d::get_biases_size() {
    return biases.size();
}

CUDANet::Shape Conv2d::get_padding_shape() {
    return padding_shape;
}