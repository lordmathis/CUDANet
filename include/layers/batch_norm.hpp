#pragma once

#include "layer.hpp"

namespace CUDANet::Layers {

class BatchNorm2d : public CUDANet::Layer {
  public:
    BatchNorm2d(CUDANet::Shape input_shape, float epsilon, CUDANet::Backend *backend);
    BatchNorm2d(CUDANet::Shape input_shape, float epsilon, CUDANet::DType dtype, CUDANet::Backend *backend);

    ~BatchNorm2d();

    CUDANet::Tensor& forward(CUDANet::Tensor& input) override;

    CUDANet::Shape input_shape() override;

    CUDANet::Shape output_shape() override;

    size_t input_size() override;

    size_t output_size() override;

    void set_weights(void* input) override;

    size_t get_weights_size() override;

    void set_biases(void* input) override;

    size_t get_biases_size() override;

    void set_running_mean(void* input);

    size_t get_running_mean_size();

    void set_running_var(void* input);

    size_t get_running_var_size();

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
