#pragma once

#include <cstddef>

namespace CUDANet {

// Forward declaration
class Tensor;

class Backend {
  public:
    // Memory management
    virtual void* allocate(size_t bytes) = 0;
    virtual void  deallocate(void* ptr)  = 0;

    // Tensor ops
    virtual void print(const CUDANet::Tensor& input) = 0;
    virtual void zero(CUDANet::Tensor& input)        = 0;

    virtual void
    copy_to_device(CUDANet::Tensor& tensor, void* data, size_t size) = 0;

    virtual void sum(const CUDANet::Tensor& input, CUDANet::Tensor& sum) = 0;
    virtual void max(const CUDANet::Tensor& input, CUDANet::Tensor& max) = 0;

    // Layer ops
    virtual void relu(CUDANet::Tensor& tensor)    = 0;
    virtual void sigmoid(CUDANet::Tensor& tensor) = 0;
    virtual void softmax(
        CUDANet::Tensor& tensor,
        CUDANet::Tensor& temp_max,
        CUDANet::Tensor& temp_sum
    ) = 0;

    virtual CUDANet::Tensor& dense(
        const CUDANet::Tensor& weights,
        const CUDANet::Tensor& biases,
        const CUDANet::Tensor& input,
        CUDANet::Tensor& output,
        const size_t           input_size,
        const size_t           output_size
    ) = 0;

    virtual CUDANet::Tensor& conv2d(
        const CUDANet::Tensor& weights,
        const CUDANet::Tensor& biases,
        const CUDANet::Tensor& input,
        CUDANet::Tensor& output,
        const CUDANet::Shape in_shape,
        const CUDANet::Shape padding_shape,
        const CUDANet::Shape kernel_shape,
        const CUDANet::Shape stride_shape,
        const CUDANet::Shape out_shape
    ) = 0;

    virtual CUDANet::Tensor& maxPool2d(
        const CUDANet::Tensor& input,
        CUDANet::Tensor& output,
        CUDANet::Shape input_shape,
        CUDANet::Shape pool_shape,
        CUDANet::Shape stride_shape,
        CUDANet::Shape padding_shape,
        CUDANet::Shape output_shape
    ) = 0;

    virtual CUDANet::Tensor& avgPool2d(
        const CUDANet::Tensor& input,
        CUDANet::Tensor& output,
        CUDANet::Shape input_shape,
        CUDANet::Shape pool_shape,
        CUDANet::Shape stride_shape,
        CUDANet::Shape padding_shape,
        CUDANet::Shape output_shape
    ) = 0;
};

}  // namespace CUDANet