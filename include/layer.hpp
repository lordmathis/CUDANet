#pragma once

#include <vector>

#include "shape.hpp"
#include "tensor.hpp"

#define CUDANET_SAME_PADDING(inputSize, kernelSize, stride) \
    ((stride - 1) * inputSize - stride + kernelSize) / 2;


namespace CUDANet {

/**
 * @brief Basic Sequential Layer
 *
 */
class Layer {
  public:

    virtual ~Layer(){};

    virtual CUDANet::Tensor& forward(CUDANet::Tensor &input) = 0;
    
    virtual CUDANet::Shape input_shape() = 0;

    virtual CUDANet::Shape output_shape() = 0;

    virtual size_t input_size() = 0;

    virtual size_t output_size() = 0;

    virtual void set_weights(void *input) = 0;

    virtual CUDANet::Tensor& get_weights() = 0;

    virtual void set_biases(void *input) = 0;

    virtual CUDANet::Tensor& get_biases() = 0;
};

}  // namespace CUDANet::Layers
