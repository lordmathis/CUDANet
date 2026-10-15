#pragma once

#include <array>
#include <stdexcept>
#include <string>

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
 * @brief Array of all activation functions
 * 
 */
constexpr std::array ALL_ACTIVATIONS = {
    SIGMOID,
    RELU,
    SOFTMAX,
    NONE,
};

inline ActivationType str2at(const std::string& activation_type) {
    if (activation_type == "sigmoid") {
        return SIGMOID;
    }
    if (activation_type == "relu") {
        return RELU;
    }
    if (activation_type == "softmax") {
        return SOFTMAX;
    }
    if (activation_type == "none") {
        return NONE;
    }
    throw std::runtime_error("Unknown activation type: " + activation_type);
}

/**
 * @brief Utility class that performs activation
 * 
 */
class Activation : public CUDANet::Layer {
  public:

    Activation() = default;

    Activation(ActivationType activation, const CUDANet::Shape &shape, CUDANet::Backend* backend);
    Activation(ActivationType activation, const CUDANet::Shape &shape, CUDANet::DType dtype, CUDANet::Backend* backend);

    ~Activation() = default;

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
    CUDANet::Backend* backend;
    ActivationType activation_type;
    CUDANet::Shape shape;

    CUDANet::Tensor softmax_sum;
    CUDANet::Tensor tensor_max;
};

}  // namespace CUDANet::Layers
