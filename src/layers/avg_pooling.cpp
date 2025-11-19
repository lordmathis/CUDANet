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
    : in_shape(input_shape),
      pool_shape(pool_shape),
      stride_shape(stride_shape),
      padding_shape(padding_shape),
      backend(backend) {
    if (in_shape.size() != 3) {
        throw std::runtime_error(
            std::format(
                "Invalid input shape. Expected 3 dims, got {}", input_shape.size()
            )
        );
    }

    if (pool_shape.size() != 2) {
        throw std::runtime_error(
            std::format(
                "Invalid pool shape. Expected 2 dims, got {}", pool_shape.size()
            )
        );
    }

    if (stride_shape.size() != 2) {
        throw std::runtime_error(
            std::format(
                "Invalid stride shape. Expected 2 dims, got {}", stride_shape.size()
            )
        );
    }

    if (padding_shape.size() != 2) {
        throw std::runtime_error(
            std::format(
                "Invalid padding shape. Expected 2 dims, got {}", padding_shape.size()
            )
        );
    }

    out_shape = {
        (in_shape[0] + 2 * padding_shape[0] - pool_shape[0]) / stride_shape[0] +
            1,
        (in_shape[1] + 2 * padding_shape[1] - pool_shape[1]) / stride_shape[1] +
            1,
        in_shape[2]
    };

    output = CUDANet::Tensor(
        Shape{out_shape[0] * out_shape[1] * out_shape[2]},
        CUDANet::DType::FLOAT32, backend
    );
}

AvgPool2d::~AvgPool2d() {}

CUDANet::Tensor& AvgPool2d::forward(CUDANet::Tensor& input);

CUDANet::Shape AvgPool2d::input_shape();

CUDANet::Shape AvgPool2d::output_shape();

size_t AvgPool2d::input_size();

size_t AvgPool2d::output_size();

void AvgPool2d::set_weights(void* input);

CUDANet::Tensor& AvgPool2d::get_weights();

void AvgPool2d::set_biases(void* input);

CUDANet::Tensor& AvgPool2d::get_biases();


AdaptiveAvgPool2d::AdaptiveAvgPool2d(
    CUDANet::Shape        input_shape,
    CUDANet::Shape        output_shape,
    CUDANet::Backend *backend
)
    : AvgPool2d(input_shape, {1, 1}, {1, 1}, {0, 0}, backend) {
    stride_shape = {
        input_shape[0] / output_shape[0],
        input_shape[1] / output_shape[1]
    };
    pool_shape = {
        input_shape[0] - (output_shape[0] - 1) * stride_shape[0],
        input_shape[1] - (output_shape[1] - 1) * stride_shape[1]
    };
    padding_shape    = {(pool_shape[0] - 1) / 2, (pool_shape[1] - 1) / 2};
    out_shape = output_shape;
}