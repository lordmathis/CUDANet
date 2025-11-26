#pragma once

#include <cstdio>
#include <set>

#include "backend.hpp"
#include "tensor.hpp"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif  // BLOCK_SIZE

/**
 * @brief CUDA error checking macro
 *
 */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t result = call;                                             \
        if (result != cudaSuccess) {                                           \
            fprintf(                                                           \
                stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", __FILE__, \
                __LINE__, static_cast<unsigned int>(result),                   \
                cudaGetErrorString(result), #call                              \
            );                                                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

namespace CUDANet::Backends {

template <DType dtype>
struct cuda_dtype_map;

template <>
struct cuda_dtype_map<DType::FLOAT32> {
    using type = float;
};

class CUDA : public Backend {
  public:
    CUDA(const BackendConfig& config);

    bool  supports_dtype(DType dtype) const override;
    void  set_default_dtype(DType dtype) override;
    DType get_default_dtype() const override;

    static bool is_cuda_available();
    void        initialize();

    // Memory management
    void* allocate(size_t bytes) override;
    void  deallocate(void* ptr) override;

    // Tensor ops dispatchers
    void print(const CUDANet::Tensor& input) override;
    void zero(CUDANet::Tensor& input) override;
    void fill(CUDANet::Tensor& input, int value) override;
    void
    copy_to_device(CUDANet::Tensor& tensor, void* data, size_t size) override;
    void sum(const CUDANet::Tensor& input, CUDANet::Tensor& sum) override;
    void max(const CUDANet::Tensor& input, CUDANet::Tensor& max) override;

    // Layer ops dispatchers
    void relu(CUDANet::Tensor& tensor) override;
    void sigmoid(CUDANet::Tensor& tensor) override;
    void softmax(
        CUDANet::Tensor& tensor,
        CUDANet::Tensor& temp_max,
        CUDANet::Tensor& temp_sum
    ) override;

    CUDANet::Tensor& dense(
        const CUDANet::Tensor& weights,
        const CUDANet::Tensor& biases,
        const CUDANet::Tensor& input,
        CUDANet::Tensor&       output,
        const size_t           input_size,
        const size_t           output_size
    ) override;

    CUDANet::Tensor& conv2d(
        const CUDANet::Tensor& weights,
        const CUDANet::Tensor& biases,
        const CUDANet::Tensor& input,
        CUDANet::Tensor&       output,
        const CUDANet::Shape   in_shape,
        const CUDANet::Shape   padding_shape,
        const CUDANet::Shape   kernel_shape,
        const CUDANet::Shape   stride_shape,
        const CUDANet::Shape   out_shape
    ) override;

    CUDANet::Tensor& max_pool2d(
        const CUDANet::Tensor& input,
        CUDANet::Tensor&       output,
        CUDANet::Shape         input_shape,
        CUDANet::Shape         pool_shape,
        CUDANet::Shape         stride_shape,
        CUDANet::Shape         padding_shape,
        CUDANet::Shape         output_shape
    ) override;

    CUDANet::Tensor& avg_pool2d(
        const CUDANet::Tensor& input,
        CUDANet::Tensor&       output,
        CUDANet::Shape         input_shape,
        CUDANet::Shape         pool_shape,
        CUDANet::Shape         stride_shape,
        CUDANet::Shape         padding_shape,
        CUDANet::Shape         output_shape
    ) override;

    CUDANet::Tensor& batch_norm(
        const CUDANet::Tensor& input,
        CUDANet::Tensor&       output,
        CUDANet::Shape         input_shape,
        CUDANet::Tensor&       weights,
        CUDANet::Tensor&       biases,
        CUDANet::Tensor&       running_mean,
        CUDANet::Tensor&       running_var,
        CUDANet::Tensor&       epsilon
    ) override;

    CUDANet::Tensor& concat(
        CUDANet::Tensor& input_a,
        CUDANet::Tensor& input_b,
        CUDANet::Tensor& output
    ) override;

    CUDANet::Tensor& add(
        CUDANet::Tensor& input_a,
        CUDANet::Tensor& input_b,
        CUDANet::Tensor& output
    ) override;

  private:
    int             device_id;
    std::set<DType> supported_dtypes;

    // Tensor ops template impls
    template <typename T>
    void print_impl(const CUDANet::Tensor& input);

    template <typename T>
    void fill_impl(CUDANet::Tensor& input, int value);

    template <typename T>
    void copy_to_device_impl(CUDANet::Tensor& tensor, void* data, size_t size);

    template <typename T>
    void sum_impl(const CUDANet::Tensor& input, CUDANet::Tensor& sum);

    template <typename T>
    void max_impl(const CUDANet::Tensor& input, CUDANet::Tensor& max);

    // Layer ops template impls
    template <typename T>
    void relu_impl(CUDANet::Tensor& tensor);

    template <typename T>
    void sigmoid_impl(CUDANet::Tensor& tensor);

    template <typename T>
    void softmax_impl(
        CUDANet::Tensor& tensor,
        CUDANet::Tensor& temp_max,
        CUDANet::Tensor& temp_sum
    );

    template <typename T>
    CUDANet::Tensor& dense_impl(
        const CUDANet::Tensor& weights,
        const CUDANet::Tensor& biases,
        const CUDANet::Tensor& input,
        CUDANet::Tensor&       output,
        const size_t           input_size,
        const size_t           output_size
    );

    template <typename T>
    CUDANet::Tensor& conv2d_impl(
        const CUDANet::Tensor& weights,
        const CUDANet::Tensor& biases,
        const CUDANet::Tensor& input,
        CUDANet::Tensor&       output,
        const CUDANet::Shape   in_shape,
        const CUDANet::Shape   padding_shape,
        const CUDANet::Shape   kernel_shape,
        const CUDANet::Shape   stride_shape,
        const CUDANet::Shape   out_shape
    );

    template <typename T>
    CUDANet::Tensor& max_pool2d_impl(
        const CUDANet::Tensor& input,
        CUDANet::Tensor&       output,
        CUDANet::Shape         input_shape,
        CUDANet::Shape         pool_shape,
        CUDANet::Shape         stride_shape,
        CUDANet::Shape         padding_shape,
        CUDANet::Shape         output_shape
    );

    template <typename T>
    CUDANet::Tensor& avg_pool2d_impl(
        const CUDANet::Tensor& input,
        CUDANet::Tensor&       output,
        CUDANet::Shape         input_shape,
        CUDANet::Shape         pool_shape,
        CUDANet::Shape         stride_shape,
        CUDANet::Shape         padding_shape,
        CUDANet::Shape         output_shape
    );

    template <typename T>
    CUDANet::Tensor& batch_norm_impl(
        const CUDANet::Tensor& input,
        CUDANet::Tensor&       output,
        CUDANet::Shape         input_shape,
        CUDANet::Tensor&       weights,
        CUDANet::Tensor&       biases,
        CUDANet::Tensor&       running_mean,
        CUDANet::Tensor&       running_var,
        CUDANet::Tensor&       epsilon
    );

    template <typename T>
    CUDANet::Tensor& concat_impl(
        CUDANet::Tensor& input_a,
        CUDANet::Tensor& input_b,
        CUDANet::Tensor& output
    );

    template <typename T>
    CUDANet::Tensor& add_impl(
        CUDANet::Tensor& input_a,
        CUDANet::Tensor& input_b,
        CUDANet::Tensor& output
    );
};

}  // namespace CUDANet::Backends