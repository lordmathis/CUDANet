#pragma once

#include <vector>

#include "activation.hpp"
#include "layer.hpp"

namespace CUDANet::Layers {

/**
 * @brief 2D convolutional layer
 *
 */
class Conv2d : public Layer {
  public:
    Conv2d(
        CUDANet::Shape    input_shape,
        CUDANet::Shape    kernel_shape,
        CUDANet::Shape    stride_shape,
        CUDANet::Shape    padding_shape,
        CUDANet::Backend* backend
    );

    ~Conv2d() {};

    CUDANet::Tensor& forward(const CUDANet::Tensor& input) override;

    CUDANet::Shape input_shape() override;

    CUDANet::Shape output_shape() override;

    size_t input_size() override;

    size_t output_size();

    void set_weights(void* input) override;

    CUDANet::Tensor& get_weights() override;

    void set_biases(void* input) override;

    CUDANet::Tensor& get_biases() override;

    CUDANet::Shape get_padding_shape();

  private:
    CUDANet::Backend* backend;

    CUDANet::Shape in_shape;
    CUDANet::Shape out_shape;

    CUDANet::Shape kernel_shape;
    CUDANet::Shape stride_shape;
    CUDANet::Shape padding_shape;

    CUDANet::Tensor weights;
    CUDANet::Tensor biases;

    CUDANet::Tensor output;
};

}  // namespace CUDANet::Layers
