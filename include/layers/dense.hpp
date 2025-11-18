#pragma once

#include <vector>

#include "backend.hpp"
#include "layer.hpp"

namespace CUDANet::Layers {

/**
 * @brief Dense (fully connected) layer
 *
 */
class Dense : public Layer {
  public:

    Dense(CUDANet::Backend *backend, CUDANet::Shape input_shape, CUDANet::Shape output_shape);

    ~Dense();

    CUDANet::Tensor& forward(const CUDANet::Tensor &input) override;

    CUDANet::Shape input_shape() override;

    CUDANet::Shape output_shape() override;

    size_t input_size() override;

    size_t output_size() override;

    void set_weights(void *input) override;

    CUDANet::Tensor& get_weights() override;

    void set_biases(void *input) override;

    CUDANet::Tensor& get_biases() override;

  private:
    CUDANet::Backend *backend;

    CUDANet::Shape in_shape;
    CUDANet::Shape out_shape;

    CUDANet::Tensor weights;
    CUDANet::Tensor biases;

    CUDANet::Tensor output;
};

}  // namespace CUDANet::Layers

