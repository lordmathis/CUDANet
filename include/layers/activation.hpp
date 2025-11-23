#pragma once

#include "tensor.hpp"
#include "backend.hpp"
#include "layer.hpp"

namespace CUDANet::Layers {

/**
 * @brief Activation functions
 * 
 * SIGMOID: Sigmoid
 * RELU: Rectified Linear Unit
 * SOFTMAX: Softmax
 *
 */
enum ActivationType { SIGMOID, RELU, SOFTMAX, NONE };

/**
 * @brief Utility class that performs activation
 * 
 */
class Activation : public Layer {
  public:

    Activation() = default;

    Activation(ActivationType activation, const CUDANet::Shape &shape, CUDANet::Backend* backend);

    ~Activation() = default;

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
    CUDANet::Backend* backend;
    ActivationType activationType;
    CUDANet::Shape shape;

    CUDANet::Tensor softmax_sum;
    CUDANet::Tensor tensor_max;
};

}  // namespace CUDANet::Layers
