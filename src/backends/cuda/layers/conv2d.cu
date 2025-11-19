#include <vector>

#include "activation.hpp"
#include "conv2d.hpp"
#include "convolution.cuh"
#include "cuda_helper.cuh"
#include "layer.hpp"
#include "matmul.cuh"
#include "vector.cuh"

using namespace CUDANet::Layers;

void Conv2d::initCUDA() {
    d_output = nullptr;
    CUDA_CHECK(cudaMalloc(
        (void**)&d_output,
        sizeof(float) * outputSize.first * outputSize.second * numFilters
    ));

    d_weights = nullptr;
    CUDA_CHECK(cudaMalloc(
        (void**)&d_weights, sizeof(float) * kernelSize.first *
                                kernelSize.second * inputChannels * numFilters
    ));

    d_biases = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_biases, sizeof(float) * numFilters));
}

void Conv2d::delCUDA() {
    cudaFree(d_output);
    cudaFree(d_weights);
    cudaFree(d_biases);
}

void Conv2d::toCuda() {
    CUDA_CHECK(cudaMemcpy(
        d_weights, weights.data(),
        sizeof(float) * kernelSize.first * kernelSize.second * inputChannels *
            numFilters,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_biases, biases.data(), sizeof(float) * numFilters,
        cudaMemcpyHostToDevice
    ));
}

float* Conv2d::forwardCUDA(const float* d_input) {
    // Convolve

}
