#include <stdexcept>

#include "avg_pool.hpp"
#include <format>

using namespace CUDANet::Layers;

AvgPool2d::AvgPool2d(
    CUDANet::Shape    input_shape,
    CUDANet::Shape    pool_shape,
    CUDANet::Shape    stride_shape,
    CUDANet::Shape    padding_shape,
    CUDANet::Backend* backend
)
    : AvgPool2d(input_shape, pool_shape, stride_shape, padding_shape, backend->get_default_dtype(), backend) {}

AvgPool2d::AvgPool2d(
    CUDANet::Shape    input_shape,
    CUDANet::Shape    pool_shape,
    CUDANet::Shape    stride_shape,
    CUDANet::Shape    padding_shape,
    CUDANet::DType    dtype,
    CUDANet::Backend* backend
)
    : in_shape(input_shape),
      pool_shape(pool_shape),
      stride_shape(stride_shape),
      padding_shape(padding_shape),
      backend(backend) {
    if (in_shape.size() != 3) {
        throw InvalidShapeException("input", 3, in_shape.size());
    }

    if (pool_shape.size() != 2) {
        throw InvalidShapeException("pool", 2, pool_shape.size());
    }

    if (stride_shape.size() != 2) {
        throw InvalidShapeException("stride", 2, stride_shape.size());
    }

    if (padding_shape.size() != 2) {
        throw InvalidShapeException("padding", 2, padding_shape.size());
    }

    this->dtype = dtype;

    out_shape = {
        (in_shape[0] + 2 * padding_shape[0] - pool_shape[0]) / stride_shape[0] +
            1,
        (in_shape[1] + 2 * padding_shape[1] - pool_shape[1]) / stride_shape[1] +
            1,
        in_shape[2]
    };

    output = CUDANet::Tensor(
        Shape{out_shape[0] * out_shape[1] * out_shape[2]},
        dtype, backend
    );
}

AvgPool2d::~AvgPool2d() {}

CUDANet::Tensor& AvgPool2d::forward(CUDANet::Tensor& input) {
    output.zero();
    backend->avg_pool2d(
        input,
        output,
        in_shape,
        pool_shape,
        stride_shape,
        padding_shape,
        out_shape
    );
    return output;
}

CUDANet::Shape AvgPool2d::input_shape() {
    return in_shape;
}

CUDANet::Shape AvgPool2d::output_shape() {
    return out_shape;
}

size_t AvgPool2d::input_size() {
    return sizeof(float) * in_shape[0] * in_shape[1] * in_shape[2];
}

size_t AvgPool2d::output_size() {
    return sizeof(float) * out_shape[0] * out_shape[1] * out_shape[2];
}

void AvgPool2d::set_weights(void* input) {}

size_t AvgPool2d::get_weights_size() {
    return 0;
}

void AvgPool2d::set_biases(void* input) {}

size_t AvgPool2d::get_biases_size() {
    return 0;
}


AdaptiveAvgPool2d::AdaptiveAvgPool2d(
    CUDANet::Shape        input_shape,
    CUDANet::Shape        output_shape,
    CUDANet::Backend *backend
)
    : AdaptiveAvgPool2d(input_shape, output_shape, backend->get_default_dtype(), backend) {}

AdaptiveAvgPool2d::AdaptiveAvgPool2d(
    CUDANet::Shape        input_shape,
    CUDANet::Shape        output_shape,
    CUDANet::DType        dtype,
    CUDANet::Backend *backend
)
    : AvgPool2d(
        input_shape,
        // pool_shape
        {
            input_shape[0] - (output_shape[0] - 1) * (input_shape[0] / output_shape[0]),
            input_shape[1] - (output_shape[1] - 1) * (input_shape[1] / output_shape[1])
        },
        // stride_shape
        {
            input_shape[0] / output_shape[0],
            input_shape[1] / output_shape[1]
        },
        // padding_shape
        {
            (input_shape[0] - (output_shape[0] - 1) * (input_shape[0] / output_shape[0]) - 1) / 2,
            (input_shape[1] - (output_shape[1] - 1) * (input_shape[1] / output_shape[1]) - 1) / 2
        },
        dtype,
        backend
    ) {
    out_shape = output_shape;

    output = CUDANet::Tensor(
        Shape{out_shape[0] * out_shape[1] * out_shape[2]},
        dtype, backend
    );
}
