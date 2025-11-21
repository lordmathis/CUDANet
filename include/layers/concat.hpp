#pragma once

#include "layer.hpp"

namespace CUDANet::Layers {

/**
 * @brief Concatenate layers
 *
 */
class Concat {
  public:

    Concat(const CUDANet::Shape a_shape, const CUDANet::Shape b_shape, CUDANet::Backend *backend);

    ~Concat();

    CUDANet::Tensor& forward(CUDANet::Tensor& input_a, CUDANet::Tensor& input_b);

    CUDANet::Shape output_shape();

  private:
    CUDANet::Shape a_shape;
    CUDANet::Shape b_shape;

    CUDANet::Shape out_shape;
    CUDANet::Tensor output;

    CUDANet::Backend *backend;
};

}  // namespace CUDANet::Layers

