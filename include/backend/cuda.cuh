#pragma once

#include "backend.hpp"
#include "tensor.hpp"

namespace CUDANet::Backend {

class CUDA : public Backend {
  public:
    // Memory management
    void* allocate(size_t bytes) override;
    void  deallocate(void* ptr) override;

    // Tensor ops
    void print(const CUDANet::Tensor& input) override;
    void zero(CUDANet::Tensor& input) override;
    void
    copy_to_device(CUDANet::Tensor& tensor, void* data, size_t size) override;
    void sum(const CUDANet::Tensor& input, CUDANet::Tensor& sum) override;
    void max(const CUDANet::Tensor& input, CUDANet::Tensor& max) override;

    // Layer ops
    void relu(CUDANet::Tensor& tensor) override;
    void sigmoid(CUDANet::Tensor& tensor) override;
    void softmax(
        CUDANet::Tensor& tensor,
        CUDANet::Tensor& temp_max,
        CUDANet::Tensor& temp_sum
    ) override;

    CUDANet::Tensor& dense(
        CUDANet::Tensor& weights,
        CUDANet::Tensor& biases,
        CUDANet::Tensor& input,
        CUDANet::Tensor& output,
        size_t           input_size,
        size_t           output_size
    ) override;
};

}  // namespace CUDANet::Backend