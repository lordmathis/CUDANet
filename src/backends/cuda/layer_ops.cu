#include "backend/cuda/cuda.cuh"
#include "backend/cuda/kernels/activation_functions.cuh"
#include "backend/cuda/kernels/convolution.cuh"
#include "backend/cuda/kernels/matmul.cuh"
#include "backend/cuda/kernels/pool.cuh"

using namespace CUDANet::Backends;

void CUDA::relu(Tensor& tensor) {
    switch (tensor.get_dtype()) {
        case DType::FLOAT32:
            relu_impl<float>(tensor);
            break;

        default:
            throw std::runtime_error("Unsupported dtype");
            break;
    }
}

template void CUDA::relu_impl<float>(Tensor& tensor);

template <typename T>
void CUDA::relu_impl(Tensor& tensor) {
    int gridSize = (tensor.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    Kernels::relu<<<gridSize, BLOCK_SIZE>>>(
        static_cast<T*>(tensor.device_ptr()), static_cast<T*>(tensor.device_ptr()), tensor.numel()
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CUDA::sigmoid(CUDANet::Tensor& tensor) {
    switch (tensor.get_dtype()) {
        case DType::FLOAT32:
            sigmoid_impl<float>(tensor);
            break;

        default:
            throw std::runtime_error("Unsupported dtype");
            break;
    }
}

template void CUDA::sigmoid_impl<float>(Tensor& tensor);

template <typename T>
void CUDA::sigmoid_impl(CUDANet::Tensor& tensor) {
    int gridSize = (tensor.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    Kernels::sigmoid<<<gridSize, BLOCK_SIZE>>>(
        static_cast<T*>(tensor.device_ptr()), static_cast<T*>(tensor.device_ptr()), tensor.numel()
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CUDA::softmax(Tensor& tensor, Tensor& temp_max, Tensor& temp_sum) {
    switch (tensor.get_dtype()) {
        case DType::FLOAT32:
            softmax_impl<float>(tensor, temp_max, temp_sum);
            break;

        default:
            throw std::runtime_error("Unsupported dtype");
            break;
    }
}

template void
CUDA::softmax_impl<float>(Tensor& tensor, Tensor& temp_max, Tensor& temp_sum);

template <typename T>
void CUDA::softmax_impl(Tensor& tensor, Tensor& temp_max, Tensor& temp_sum) {
    int gridSize = (tensor.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Find max value
    max(tensor, temp_max);

    // Subtract max value to improve numerical stability
    Kernels::vec_scalar_sub<<<gridSize, BLOCK_SIZE>>>(
        static_cast<T*>(tensor.device_ptr()), static_cast<T*>(tensor.device_ptr()), static_cast<T*>(temp_max.device_ptr()), tensor.numel()
    );
    CUDA_CHECK(cudaGetLastError());

    // Compute exponentials
    Kernels::vec_exp<<<gridSize, BLOCK_SIZE>>>(
        static_cast<T*>(tensor.device_ptr()), static_cast<T*>(tensor.device_ptr()), tensor.numel()
    );
    CUDA_CHECK(cudaGetLastError());

    // Find sum
    sum(tensor, temp_sum);

    Kernels::vec_scalar_div<<<gridSize, BLOCK_SIZE>>>(
        static_cast<T*>(tensor.device_ptr()), static_cast<T*>(tensor.device_ptr()), static_cast<T*>(temp_sum.device_ptr()), tensor.numel()
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

CUDANet::Tensor& CUDA::dense(
    const CUDANet::Tensor& weights,
    const CUDANet::Tensor& biases,
    const CUDANet::Tensor& input,
    CUDANet::Tensor&       output,
    const size_t           input_size,
    const size_t           output_size
) {
    switch (input.get_dtype()) {
        case DType::FLOAT32:
            return dense_impl<float>(
                weights, biases, input, output, input_size, output_size
            );
            break;

        default:
            throw std::runtime_error("Unsupported dtype");
            break;
    }
}

template CUDANet::Tensor& CUDA::dense_impl<float>(
    const CUDANet::Tensor& weights,
    const CUDANet::Tensor& biases,
    const CUDANet::Tensor& input,
    CUDANet::Tensor&       output,
    const size_t           input_size,
    const size_t           output_size
);

template <typename T>
CUDANet::Tensor& CUDA::dense_impl(
    const CUDANet::Tensor& weights,
    const CUDANet::Tensor& biases,
    const CUDANet::Tensor& input,
    CUDANet::Tensor&       output,
    const size_t           input_size,
    const size_t           output_size
) {
    auto forwardGridSize =
        (std::max(input_size, output_size) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    auto biasGridSize = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    Kernels::mat_vec_mul<<<forwardGridSize, BLOCK_SIZE>>>(
        static_cast<const T*>(weights.device_ptr()), static_cast<const T*>(input.device_ptr()), static_cast<T*>(output.device_ptr()), input_size,
        output_size
    );
    CUDA_CHECK(cudaGetLastError());

    Kernels::vec_vec_add<<<biasGridSize, BLOCK_SIZE>>>(
        static_cast<const T*>(biases.device_ptr()), static_cast<T*>(output.device_ptr()), static_cast<T*>(output.device_ptr()), output_size
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

CUDANet::Tensor& CUDA::conv2d(
    const CUDANet::Tensor& weights,
    const CUDANet::Tensor& biases,
    const CUDANet::Tensor& input,
    CUDANet::Tensor&       output,
    const CUDANet::Shape   in_shape,
    const CUDANet::Shape   padding_shape,
    const CUDANet::Shape   kernel_shape,
    const CUDANet::Shape   stride_shape,
    const CUDANet::Shape   out_shape
) {
    switch (input.get_dtype()) {
        case DType::FLOAT32:
            return conv2d_impl<float>(
                weights, biases, input, output, in_shape, padding_shape,
                kernel_shape, stride_shape, out_shape
            );
            break;

        default:
            throw std::runtime_error("Unsupported dtype");
            break;
    }
}

template CUDANet::Tensor& CUDA::conv2d_impl<float>(
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
CUDANet::Tensor& CUDA::conv2d_impl(
    const CUDANet::Tensor& weights,
    const CUDANet::Tensor& biases,
    const CUDANet::Tensor& input,
    CUDANet::Tensor&       output,
    const CUDANet::Shape   in_shape,
    const CUDANet::Shape   padding_shape,
    const CUDANet::Shape   kernel_shape,
    const CUDANet::Shape   stride_shape,
    const CUDANet::Shape   out_shape
) {
    dim3 block(8, 8, 8);
    dim3 grid(
        (out_shape[0] + block.x - 1) / block.x,
        (out_shape[1] + block.y - 1) / block.y,
        (out_shape[2] + block.z - 1) / block.z
    );

    Kernels::convolution<<<grid, block>>>(
        static_cast<const T*>(input.device_ptr()), static_cast<const T*>(weights.device_ptr()), static_cast<const T*>(biases.device_ptr()), static_cast<T*>(output.device_ptr()),
        in_shape, padding_shape, kernel_shape, stride_shape, out_shape
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

CUDANet::Tensor& CUDA::max_pool2d(
    const CUDANet::Tensor& input,
    CUDANet::Tensor&       output,
    CUDANet::Shape         input_shape,
    CUDANet::Shape         pool_shape,
    CUDANet::Shape         stride_shape,
    CUDANet::Shape         padding_shape,
    CUDANet::Shape         output_shape
) {
    switch (input.get_dtype()) {
        case DType::FLOAT32:
            return max_pool2d_impl<float>(
                input, output, input_shape, pool_shape, stride_shape,
                padding_shape, output_shape
            );
            break;

        default:
            throw std::runtime_error("Unsupported dtype");
            break;
    }
}

template CUDANet::Tensor& CUDA::max_pool2d_impl<float>(
    const CUDANet::Tensor& input,
    CUDANet::Tensor&       output,
    CUDANet::Shape         input_shape,
    CUDANet::Shape         pool_shape,
    CUDANet::Shape         stride_shape,
    CUDANet::Shape         padding_shape,
    CUDANet::Shape         output_shape
);

template <typename T>
CUDANet::Tensor& CUDA::max_pool2d_impl(
    const CUDANet::Tensor& input,
    CUDANet::Tensor&       output,
    CUDANet::Shape         input_shape,
    CUDANet::Shape         pool_shape,
    CUDANet::Shape         stride_shape,
    CUDANet::Shape         padding_shape,
    CUDANet::Shape         output_shape
) {
    dim3 block(8, 8, 8);
    dim3 grid(
        (output_shape[0] + block.x - 1) / block.x,
        (output_shape[1] + block.y - 1) / block.y,
        (output_shape[2] + block.z - 1) / block.z
    );

    Kernels::max_pool<<<grid, block>>>(
        static_cast<const T*>(input.device_ptr()), static_cast<T*>(output.device_ptr()), input_shape, output_shape,
        pool_shape, stride_shape, padding_shape
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

CUDANet::Tensor& CUDA::avg_pool2d(
    const CUDANet::Tensor& input,
    CUDANet::Tensor&       output,
    CUDANet::Shape         input_shape,
    CUDANet::Shape         pool_shape,
    CUDANet::Shape         stride_shape,
    CUDANet::Shape         padding_shape,
    CUDANet::Shape         output_shape
) {
    switch (input.get_dtype()) {
        case DType::FLOAT32:
            return avg_pool2d_impl<float>(
                input, output, input_shape, pool_shape, stride_shape,
                padding_shape, output_shape
            );
            break;

        default:
            throw std::runtime_error("Unsupported dtype");
            break;
    }
}

template CUDANet::Tensor& CUDA::avg_pool2d_impl<float>(
    const CUDANet::Tensor& input,
    CUDANet::Tensor&       output,
    CUDANet::Shape         input_shape,
    CUDANet::Shape         pool_shape,
    CUDANet::Shape         stride_shape,
    CUDANet::Shape         padding_shape,
    CUDANet::Shape         output_shape
);

template <typename T>
CUDANet::Tensor& CUDA::avg_pool2d_impl(
    const CUDANet::Tensor& input,
    CUDANet::Tensor&       output,
    CUDANet::Shape         input_shape,
    CUDANet::Shape         pool_shape,
    CUDANet::Shape         stride_shape,
    CUDANet::Shape         padding_shape,
    CUDANet::Shape         output_shape
) {
    dim3 block(8, 8, 8);
    dim3 grid(
        (output_shape[0] + block.x - 1) / block.x,
        (output_shape[1] + block.y - 1) / block.y,
        (output_shape[2] + block.z - 1) / block.z
    );

    Kernels::avg_pool<<<grid, block>>>(
        static_cast<const T*>(input.device_ptr()), static_cast<T*>(output.device_ptr()), input_shape, output_shape,
        pool_shape, stride_shape, padding_shape
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

CUDANet::Tensor& CUDA::batch_norm(
    const CUDANet::Tensor& input,
    CUDANet::Tensor&       output,
    CUDANet::Shape         input_shape,
    CUDANet::Tensor&       weights,
    CUDANet::Tensor&       biases,
    CUDANet::Tensor&       running_mean,
    CUDANet::Tensor&       running_var,
    CUDANet::Tensor&       epsilon
) {
    switch (input.get_dtype()) {
        case DType::FLOAT32:
            return batch_norm_impl<float>(
                input, output, input_shape, weights, biases, running_mean,
                running_var, epsilon
            );
            break;

        default:
            throw std::runtime_error("Unsupported dtype");
            break;
    }
}

template CUDANet::Tensor& CUDA::batch_norm_impl<float>(
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
CUDANet::Tensor& CUDA::batch_norm_impl(
    const CUDANet::Tensor& input,
    CUDANet::Tensor&       output,
    CUDANet::Shape         input_shape,
    CUDANet::Tensor&       weights,
    CUDANet::Tensor&       biases,
    CUDANet::Tensor&       running_mean,
    CUDANet::Tensor&       running_var,
    CUDANet::Tensor&       epsilon
) {
    auto gridSize =
        (input_shape[0] * input_shape[1] + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int i = 0; i < input_shape[2]; i++) {
        // Subtract mean from input
        Kernels::vec_scalar_sub<<<gridSize, BLOCK_SIZE>>>(
            static_cast<const T*>(input.device_ptr()) + i * input_shape[0] * input_shape[1],
            static_cast<T*>(output.device_ptr()) + i * input_shape[0] * input_shape[1],
            &static_cast<T*>(running_mean.device_ptr())[i], input_shape[0] * input_shape[1]
        );
        CUDA_CHECK(cudaGetLastError());

        // Divide by sqrt(running_var + epsilon)
        Kernels::vec_scale<<<gridSize, BLOCK_SIZE>>>(
            static_cast<T*>(output.device_ptr()) + i * input_shape[0] * input_shape[1],
            static_cast<T*>(output.device_ptr()) + i * input_shape[0] * input_shape[1],
            &static_cast<T*>(running_var.device_ptr())[i], static_cast<T*>(epsilon.device_ptr()),
            input_shape[0] * input_shape[1]
        );
        CUDA_CHECK(cudaGetLastError());

        // Multiply by weights
        Kernels::vec_scalar_mul<<<gridSize, BLOCK_SIZE>>>(
            static_cast<T*>(output.device_ptr()) + i * input_shape[0] * input_shape[1],
            static_cast<T*>(output.device_ptr()) + i * input_shape[0] * input_shape[1],
            &static_cast<T*>(weights.device_ptr())[i], input_shape[0] * input_shape[1]
        );
        CUDA_CHECK(cudaGetLastError());

        // Add biases
        Kernels::vec_scalar_add<<<gridSize, BLOCK_SIZE>>>(
            static_cast<T*>(output.device_ptr()) + i * input_shape[0] * input_shape[1],
            static_cast<T*>(output.device_ptr()) + i * input_shape[0] * input_shape[1],
            &static_cast<T*>(biases.device_ptr())[i], input_shape[0] * input_shape[1]
        );
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return output;
}

CUDANet::Tensor& CUDA::concat(
    CUDANet::Tensor& input_a,
    CUDANet::Tensor& input_b,
    CUDANet::Tensor& output
) {
    switch (input_a.get_dtype()) {
        case DType::FLOAT32:
            return concat_impl<float>(
                input_a, input_b, output
            );
            break;

        default:
            throw std::runtime_error("Unsupported dtype");
            break;
    }
}

template CUDANet::Tensor& CUDA::concat_impl<float>(
    CUDANet::Tensor& input_a,
    CUDANet::Tensor& input_b,
    CUDANet::Tensor& output
);

template <typename T>
CUDANet::Tensor& CUDA::concat_impl(
    CUDANet::Tensor& input_a,
    CUDANet::Tensor& input_b,
    CUDANet::Tensor& output
) {
    CUDA_CHECK(cudaMemcpy(
        static_cast<T*>(output.device_ptr()), static_cast<const T*>(input_a.device_ptr()), input_a.size(),
        cudaMemcpyDeviceToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        static_cast<T*>(output.device_ptr()) + input_a.numel(), static_cast<const T*>(input_b.device_ptr()), input_b.size(),
        cudaMemcpyDeviceToDevice
    ));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

CUDANet::Tensor& CUDA::add(
    CUDANet::Tensor& input_a,
    CUDANet::Tensor& input_b,
    CUDANet::Tensor& output
) {
    switch (input_a.get_dtype()) {
    case DType::FLOAT32:
        return add_impl<float>(
            input_a, input_b, output
        );
        break;

    default:
        throw std::runtime_error("Unsupported dtype");
        break;
    }
}

template CUDANet::Tensor& CUDA::add_impl<float>(
    CUDANet::Tensor& input_a,
    CUDANet::Tensor& input_b,
    CUDANet::Tensor& output
);

template <typename T>
CUDANet::Tensor& CUDA::add_impl(
    CUDANet::Tensor& input_a,
    CUDANet::Tensor& input_b,
    CUDANet::Tensor& output
) {
    auto gridSize = (input_a.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;

    Kernels::vec_vec_add<<<gridSize, BLOCK_SIZE>>>(
        static_cast<const T*>(input_a.device_ptr()), static_cast<const T*>(input_b.device_ptr()), static_cast<T*>(output.device_ptr()), input_a.numel()
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}