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

    Activation(CUDANet::Backend* backend, ActivationType activation, const CUDANet::Shape &shape);

    ~Activation() = default;

    CUDANet::Tensor& forward(CUDANet::Tensor &input);
    
    CUDANet::Shape input_shape();

    CUDANet::Shape output_shape();

    size_t input_size();

    size_t output_size();

    void set_weights(CUDANet::Tensor &input);

    CUDANet::Tensor& get_weights();

    void set_biases(CUDANet::Tensor &input);

    CUDANet::Tensor& get_biases();


  private:
    CUDANet::Backend* backend;
    ActivationType activationType;
    CUDANet::Shape shape;

    CUDANet::Tensor softmax_sum;
    CUDANet::Tensor tensor_max;
};

}  // namespace CUDANet::Layers
