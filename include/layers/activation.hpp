#pragma once

#include "backend/tensor.hpp"
#include "backend/backend.hpp"

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
class Activation {
  public:

    Activation() = default;

    /**
     * @brief Construct a new Activation object
     * 
     * @param activation Type of activation
     * @param length     Length of the input
     */
    Activation(ActivationType activation, const int length);

    /**
     * @brief Destroy the Activation object
     * 
     */
    ~Activation();

    /**
     * @brief Run the activation function on the input
     * 
     * @param d_input Pointer to the input vector on the device
     */
    void activate(CUDANet::Backend::Tensor input);


  private:
    CUDANet::Backend::IBackend* backend;
    ActivationType activationType;
    int length;

    CUDANet::Backend::Tensor softmax_sum;
    CUDANet::Backend::Tensor tensor_max;
};

}  // namespace CUDANet::Layers
