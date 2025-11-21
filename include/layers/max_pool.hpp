#pragma once

#include "layer.hpp"

namespace CUDANet::Layers {

class MaxPool2d : public Layer {
  public:
    MaxPool2d(
        CUDANet::Shape        input_shape,
        CUDANet::Shape        pool_shape,
        CUDANet::Shape        stride_shape,
        CUDANet::Shape        padding_shape,
        CUDANet::Backend* backend
    );
    ~MaxPool2d();

    CUDANet::Tensor& forward(CUDANet::Tensor &input) override;
    
    CUDANet::Shape input_shape() override;

    CUDANet::Shape output_shape() override;

    size_t input_size() override;

    size_t output_size() override;

    void set_weights(void *input) override;

    CUDANet::Tensor& get_weights() override;

    void set_biases(void *input) override;

    CUDANet::Tensor& get_biases() override;



  private:
    CUDANet::Shape in_shape;

    CUDANet::Shape pool_shape;
    CUDANet::Shape stride_shape;
    CUDANet::Shape padding_shape;

    CUDANet::Shape out_shape;
    CUDANet::Tensor output;

    CUDANet::Backend *backend;
};

}  // namespace CUDANet::Layers
