#pragma once

#include "backend/backend.hpp"
#include "backend/tensor.hpp"

namespace CUDANet::Backend {

class CUDABackend : public IBackend {
public:
    // Memory management
    void* allocate(size_t bytes) override;
    void deallocate(void* ptr) override;

    // Layer operations
    void relu(CUDANet::Backend::Tensor &tensor) override;
    void sigmoid(CUDANet::Backend::Tensor &tensor) override;
    void softmax(CUDANet::Backend::Tensor &tensor, CUDANet::Backend::Tensor &temp_max, CUDANet::Backend::Tensor &temp_sum) override;

private:
    static constexpr int BLOCK_SIZE = 256;
};

}  // namespace CUDANet::Backend