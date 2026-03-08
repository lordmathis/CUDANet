#include "cudanet.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"
#include "cuda_test_utils.hpp"
#include "backend/cuda/kernels/activation_functions.cuh"

/*

Sigmoid

*/

class SigmoidTest : public ::testing::TestWithParam<UnaryOpParams> {};

template <typename T>
void run_sigmoid_test(const UnaryOpParams params) {
    auto vector_data   = load_binary<T>(params.vector_path);
    auto expected_data = load_binary<T>(params.expected_path);

    auto backend = create_backend();
    auto shape   = CUDANet::Shape{params.size};

    auto vector   = create_tensor<T>(shape, params.dtype, backend.get(), vector_data);
    auto expected = create_tensor<T>(shape, params.dtype, backend.get(), expected_data);
    auto output   = create_output_tensor<T>(shape, params.dtype, backend.get());

    auto grid_size = calc_grid_size(params.size);
    CUDANet::Kernels::sigmoid<<<grid_size, BLOCK_SIZE>>>(
        static_cast<const T*>(vector.device_ptr()),
        static_cast<T*>(output.device_ptr()), params.size
    );

    verify_output<T>(output, expected);
}

template void run_sigmoid_test<float>(const UnaryOpParams params);

TEST_P(SigmoidTest, SigmoidActivation) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_sigmoid_test<float>(param);
    }
}

INSTANTIATE_TEST_SUITE_P(
    SigmoidTestCases,
    SigmoidTest,
    testing::ValuesIn(initialize_unary_params<UnaryOpParams>("/activation/sigmoid/metadata.csv"))
);

/*

ReLU

*/

class ReLUTest : public ::testing::TestWithParam<UnaryOpParams> {};

template <typename T>
void run_relu_test(const UnaryOpParams params) {
    auto vector_data   = load_binary<T>(params.vector_path);
    auto expected_data = load_binary<T>(params.expected_path);

    auto backend = create_backend();
    auto shape   = CUDANet::Shape{params.size};

    auto vector   = create_tensor<T>(shape, params.dtype, backend.get(), vector_data);
    auto expected = create_tensor<T>(shape, params.dtype, backend.get(), expected_data);
    auto output   = create_output_tensor<T>(shape, params.dtype, backend.get());

    auto grid_size = calc_grid_size(params.size);
    CUDANet::Kernels::relu<<<grid_size, BLOCK_SIZE>>>(
        static_cast<const T*>(vector.device_ptr()),
        static_cast<T*>(output.device_ptr()), params.size
    );

    verify_output<T>(output, expected);
}

template void run_relu_test<float>(const UnaryOpParams params);

TEST_P(ReLUTest, ReLUActivation) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_relu_test<float>(param);
    }
}

INSTANTIATE_TEST_SUITE_P(
    ReLUTestCases,
    ReLUTest,
    testing::ValuesIn(initialize_unary_params<UnaryOpParams>("/activation/relu/metadata.csv"))
);
