#pragma once

#include "shape.hpp"
#include "tensor.hpp"

namespace CUDANet::Layers {

class Add {
  public:
    Add(CUDANet::Shape a_shape, CUDANet::Shape b_shape, CUDANet::Backend* backend);
    Add(CUDANet::Shape a_shape, CUDANet::Shape b_shape, CUDANet::DType dtype, CUDANet::Backend* backend);

    ~Add();

    CUDANet::Tensor&
    forward(CUDANet::Tensor& input_a, CUDANet::Tensor& input_b);

  private:
    CUDANet::Shape out_shape;
    CUDANet::Tensor output;

    CUDANet::Backend *backend;

    CUDANet::DType dtype;
};

}  // namespace CUDANet::Layers
