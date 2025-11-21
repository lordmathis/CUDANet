#pragma once

#include "layer.hpp"

namespace CUDANet::Layers {

class BatchNorm2d : public Layer {
  public:
    BatchNorm2d(CUDANet::Shape input_shape, float epsilon, CUDANet::Backend *backend);

    ~BatchNorm2d();

    CUDANet::Tensor& forward(CUDANet::Tensor& input) override;

    CUDANet::Shape input_shape() override;

    CUDANet::Shape output_shape() override;

    size_t input_size() override;

    size_t output_size() override;

    void set_weights(void* input) override;

    CUDANet::Tensor& get_weights() override;

    void set_biases(void* input) override;

    CUDANet::Tensor& get_biases() override;

    void set_running_mean(void* input);

    CUDANet::Tensor& get_running_mean();

    void set_running_var(void* input);

    CUDANet::Tensor& get_running_var();

  private:
    CUDANet::Shape  in_shape;
    CUDANet::Tensor epsilon;

    CUDANet::Tensor running_mean;
    CUDANet::Tensor running_var;

    CUDANet::Tensor weights;
    CUDANet::Tensor biases;

    CUDANet::Tensor output;

    CUDANet::Backend *backend;
};

}  // namespace CUDANet::Layers
