#pragma once

#include <memory>
#include <optional>

#include "shape.hpp"
#include "tensor.hpp"

namespace CUDANet {

// Forward declarations
class Backend;
class Tensor;
enum class DType;

enum BackendType { CUDA_BACKEND, CPU_BACKEND };

struct BackendConfig {
    int device_id = 0;
};

class BackendFactory {
  public:
    static std::unique_ptr<Backend> create(BackendType backend_type, const BackendConfig& config);
};

class Backend {
  protected:
    std::optional<DType> default_dtype;
  public:

    // Dtypes
    virtual bool supports_dtype(DType dtype) const = 0;
    virtual void set_default_dtype(DType dtype) = 0;
    virtual DType get_default_dtype() const = 0;

    // Memory management
    virtual void* allocate(size_t bytes) = 0;
    virtual void  deallocate(void* ptr)  = 0;

    // Tensor ops
    virtual void print(const CUDANet::Tensor& input) = 0;
    virtual void zero(CUDANet::Tensor& input)        = 0;
    virtual void fill(CUDANet::Tensor& input, int data) = 0;

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

    virtual CUDANet::Tensor& max_pool2d(
        const CUDANet::Tensor& input,
        CUDANet::Tensor& output,
        CUDANet::Shape input_shape,
        CUDANet::Shape pool_shape,
        CUDANet::Shape stride_shape,
        CUDANet::Shape padding_shape,
        CUDANet::Shape output_shape
    ) = 0;

    virtual CUDANet::Tensor& avg_pool2d(
        const CUDANet::Tensor& input,
        CUDANet::Tensor& output,
        CUDANet::Shape input_shape,
        CUDANet::Shape pool_shape,
        CUDANet::Shape stride_shape,
        CUDANet::Shape padding_shape,
        CUDANet::Shape output_shape
    ) = 0;

    virtual CUDANet::Tensor& batch_norm(
        const CUDANet::Tensor& input,
        CUDANet::Tensor& output,
        CUDANet::Shape input_shape,
        CUDANet::Tensor& weights,
        CUDANet::Tensor& biases,
        CUDANet::Tensor& running_mean,
        CUDANet::Tensor& running_var,
        CUDANet::Tensor& epsilon
    ) = 0;

    virtual CUDANet::Tensor& concat(
        CUDANet::Tensor& input_a,
        CUDANet::Tensor& input_b,
        CUDANet::Tensor& output
    ) = 0;

    virtual CUDANet::Tensor& add(
        CUDANet::Tensor& input_a,
        CUDANet::Tensor& input_b,
        CUDANet::Tensor& output
    ) = 0;
};

}  // namespace CUDANet