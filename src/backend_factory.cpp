#include <stdexcept>
#include <memory>

#ifdef USE_CUDA
#include "backend/cuda/cuda.cuh"
#endif

#include "backend.hpp"

namespace CUDANet {    
    
std::unique_ptr<Backend> BackendFactory::create(BackendType backend_type, const BackendConfig& config) {
    switch (backend_type)
    {
    case BackendType::CUDA_BACKEND:
        {
        #ifdef USE_CUDA

        if (!CUDANet::Backends::CUDA::is_cuda_available()) {
            throw std::runtime_error("No CUDA devices found");
        }

        auto cuda = std::make_unique<CUDANet::Backends::CUDA>(config);
        return cuda;

        #else
        throw std::runtime_error("Library was compiled without CUDA support.");
        #endif
        }
        break;
    
    default:
        throw std::runtime_error("Invalid backend");
        break;
    }

    return nullptr;
}

} // namespace CUDANet