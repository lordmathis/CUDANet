#pragma once

#include "backend.hpp"
#include "layer.hpp"

namespace CUDANet::Layers {

/**
 * @brief Dense (fully connected) layer
 *
 */
class Dense : public Layer {
  public:

    Dense(CUDANet::Shape input_shape, CUDANet::Shape output_shape, CUDANet::Backend *backend);

    ~Dense();

    CUDANet::Tensor& forward(CUDANet::Tensor &input) override;

    CUDANet::Shape input_shape() override;

    CUDANet::Shape output_shape() override;

    size_t input_size() override;

    size_t output_size() override;

    void set_weights(void *input) override;

    size_t get_weights_size() override;

    void set_biases(void *input) override;

    size_t get_biases_size() override;

  private:
    CUDANet::Backend *backend;

    CUDANet::Shape in_shape;
    CUDANet::Shape out_shape;

    CUDANet::Tensor weights;
    CUDANet::Tensor biases;

    CUDANet::Tensor output;
};

}  // namespace CUDANet::Layers

