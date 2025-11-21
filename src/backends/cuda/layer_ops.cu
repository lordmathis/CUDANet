#include "backend/cuda.cuh"
#include "kernels/activation_functions.cuh"
#include "kernels/convolution.cuh"
#include "kernels/matmul.cuh"
#include "kernels/pool.cuh"
#include "utils/cuda_helper.cuh"

using namespace CUDANet::Backend;

void CUDA::relu(Tensor& tensor) {
    int gridSize = (tensor.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    Kernels::relu<<<gridSize, BLOCK_SIZE>>>(
        tensor.data<float>(), tensor.data<float>(), tensor.numel()
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CUDA::sigmoid(Tensor& tensor) {
    int gridSize = (tensor.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    Kernels::sigmoid<<<gridSize, BLOCK_SIZE>>>(
        tensor.data<float>(), tensor.data<float>(), tensor.numel()
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CUDA::softmax(Tensor& tensor, Tensor& temp_max, Tensor& temp_sum) {
    int gridSize = (tensor.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Find max value
    max(tensor, temp_max);

    // Subtract max value to improve numerical stability
    Kernels::vec_scalar_sub<<<gridSize, BLOCK_SIZE>>>(
        tensor.data<float>(), tensor.data<float>(), temp_max.data<float>(),
        tensor.numel()
    );
    CUDA_CHECK(cudaGetLastError());

    // Compute exponentials
    Kernels::vec_exp<<<gridSize, BLOCK_SIZE>>>(
        tensor.data<float>(), tensor.data<float>(), tensor.numel()
    );
    CUDA_CHECK(cudaGetLastError());

    // Find sum
    sum(tensor, temp_sum);

    Kernels::vec_scalar_div<<<gridSize, BLOCK_SIZE>>>(
        tensor.data<float>(), tensor.data<float>(), temp_sum.data<float>(),
        tensor.numel()
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
    auto forwardGridSize =
        (std::max(input_size, output_size) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    auto biasGridSize = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    Kernels::mat_vec_mul<<<forwardGridSize, BLOCK_SIZE>>>(
        weights.data<float>(), input.data<float>(), output.data<float>(),
        input_size, output_size
    );
    CUDA_CHECK(cudaGetLastError());

    Kernels::vec_vec_add<<<biasGridSize, BLOCK_SIZE>>>(
        biases.data<float>(), output.data<float>(), output.data<float>(),
        output_size
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
    dim3 block(8, 8, 8);
    dim3 grid(
        (out_shape[0] + block.x - 1) / block.x,
        (out_shape[1] + block.y - 1) / block.y,
        (out_shape[3] + block.z - 1) / block.z
    );

    Kernels::convolution<<<grid, block>>>(
        input.data<float>(), weights.data<float>(), biases.data<float>(),
        output.data<float>(), in_shape, padding_shape, kernel_shape,
        stride_shape, out_shape
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

CUDANet::Tensor& CUDA::max_pool2d(
    const CUDANet::Tensor& input,
    CUDANet::Tensor& output,
    CUDANet::Shape input_shape,
    CUDANet::Shape pool_shape,
    CUDANet::Shape stride_shape,
    CUDANet::Shape padding_shape,
    CUDANet::Shape output_shape
) {
    dim3 block(8, 8, 8);
    dim3 grid(
        (output_shape[0] + block.x - 1) / block.x,
        (output_shape[1] + block.y - 1) / block.y,
        (output_shape[2] + block.z - 1) / block.z
    );

    Kernels::max_pool<<<grid, block>>>(
        input.data<float>(), output.data<float>(), input_shape, output_shape, pool_shape,
        stride_shape, padding_shape
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

CUDANet::Tensor& CUDA::avg_pool2d(
    const CUDANet::Tensor& input,
    CUDANet::Tensor& output,
    CUDANet::Shape input_shape,
    CUDANet::Shape pool_shape,
    CUDANet::Shape stride_shape,
    CUDANet::Shape padding_shape,
    CUDANet::Shape output_shape
) {
    dim3 block(8, 8, 8);
    dim3 grid(
        (output_shape[0] + block.x - 1) / block.x,
        (output_shape[1] + block.y - 1) / block.y,
        (output_shape[2] + block.z - 1) / block.z
    );

    Kernels::avg_pool<<<grid, block>>>(
        input.data<float>(), output.data<float>(), input_shape, output_shape, pool_shape,
        stride_shape, padding_shape
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

CUDANet::Tensor& CUDA::batch_norm(
    const CUDANet::Tensor& input,
    CUDANet::Tensor& output,
    CUDANet::Shape input_shape,
    CUDANet::Tensor& weights,
    CUDANet::Tensor& biases,
    CUDANet::Tensor& running_mean,
    CUDANet::Tensor& running_var,
    CUDANet::Tensor& epsilon
) {
    auto gridSize =
        (input_shape[0] * input_shape[1] + BLOCK_SIZE - 1) / BLOCK_SIZE;


    for (int i = 0; i < input_shape[2]; i++) {
        // Subtract mean from input
        Kernels::vec_scalar_sub<<<gridSize, BLOCK_SIZE>>>(
            input.data<float>() + i * input_shape[0] * input_shape[1],
            output.data<float>() + i * input_shape[0] * input_shape[1],
            &running_mean.data<float>()[i], input_shape[0] * input_shape[1]
        );
        CUDA_CHECK(cudaGetLastError());

        // Divide by sqrt(running_var + epsilon)
        Kernels::vec_scale<<<gridSize, BLOCK_SIZE>>>(
            output.data<float>() + i * input_shape[0] * input_shape[1],
            output.data<float>() + i * input_shape[0] * input_shape[1],
            &running_var.data<float>()[i], epsilon.data<float>(), input_shape[0] * input_shape[1]
        );
        CUDA_CHECK(cudaGetLastError());

        // Multiply by weights
        Kernels::vec_scalar_mul<<<gridSize, BLOCK_SIZE>>>(
            output.data<float>() + i * input_shape[0] * input_shape[1],
            output.data<float>() + i * input_shape[0] * input_shape[1], &weights.data<float>()[i],
            input_shape[0] * input_shape[1]
        );
        CUDA_CHECK(cudaGetLastError());

        // Add biases
        Kernels::vec_scalar_add<<<gridSize, BLOCK_SIZE>>>(
            output.data<float>() + i * input_shape[0] * input_shape[1],
            output.data<float>() + i * input_shape[0] * input_shape[1], &biases.data<float>()[i],
            input_shape[0] * input_shape[1]
        );
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

CUDANet::Tensor& CUDA::concat(
    CUDANet::Tensor& input_a,
    CUDANet::Tensor& input_b,
    CUDANet::Tensor& output
) {
    CUDA_CHECK(cudaMemcpy(
        output.data<float>(), input_a.data<float>(), input_a.size(),
        cudaMemcpyDeviceToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        output.data<float>() + input_a.numel(), input_b.data<float>(), input_b.size(),
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
    auto gridSize = (input_a.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;

    Kernels::vec_vec_add<<<gridSize, BLOCK_SIZE>>>(
        input_a.data<float>(), input_b.data<float>(), output.data<float>(), input_a.numel()
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}