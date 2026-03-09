#include "cudanet.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"
#include "cuda_test_utils.hpp"
#include "backend/cuda/kernels/pool.cuh"

class MaxPoolTest : public ::testing::TestWithParam<PoolParams> {};

template <typename T>
void run_max_pool_test(const PoolParams params) {
    auto input_data    = load_binary<T>(params.input_path);
    auto expected_data = load_binary<T>(params.expected_path);

    auto backend = create_backend();
    auto in_shape   = CUDANet::Shape{params.inH, params.inW, params.inC};
    auto out_shape  = CUDANet::Shape{params.outH, params.outW, params.inC};
    auto pool_shape = CUDANet::Shape{params.kH, params.kW};
    auto stride_shape = CUDANet::Shape{params.sH, params.sW};
    auto padding_shape = CUDANet::Shape{params.pH, params.pW};

    auto input    = create_tensor<T>(in_shape, params.dtype, backend.get(), input_data);
    auto expected = create_tensor<T>(out_shape, params.dtype, backend.get(), expected_data);
    auto output   = create_output_tensor<T>(out_shape, params.dtype, backend.get());

    dim3 block(8, 8, 8);
    dim3 grid(
        (params.outH + block.x - 1) / block.x,
        (params.outW + block.y - 1) / block.y,
        (params.inC + block.z - 1) / block.z
    );

    CUDANet::Kernels::max_pool<<<grid, block>>>(
        static_cast<const T*>(input.device_ptr()),
        static_cast<T*>(output.device_ptr()),
        in_shape, out_shape, pool_shape, stride_shape, padding_shape
    );

    verify_output<T>(output, expected);
}

template void run_max_pool_test<float>(const PoolParams params);

TEST_P(MaxPoolTest, MaxPooling) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_max_pool_test<float>(param);
    }
}

INSTANTIATE_TEST_SUITE_P(
    MaxPoolTestCases,
    MaxPoolTest,
    testing::ValuesIn(initialize_pool_params("/pool/max_pool/metadata.csv"))
);

class AvgPoolTest : public ::testing::TestWithParam<PoolParams> {};

template <typename T>
void run_avg_pool_test(const PoolParams params) {
    auto input_data    = load_binary<T>(params.input_path);
    auto expected_data = load_binary<T>(params.expected_path);

    auto backend = create_backend();
    auto in_shape   = CUDANet::Shape{params.inH, params.inW, params.inC};
    auto out_shape  = CUDANet::Shape{params.outH, params.outW, params.inC};
    auto pool_shape = CUDANet::Shape{params.kH, params.kW};
    auto stride_shape = CUDANet::Shape{params.sH, params.sW};
    auto padding_shape = CUDANet::Shape{params.pH, params.pW};

    auto input    = create_tensor<T>(in_shape, params.dtype, backend.get(), input_data);
    auto expected = create_tensor<T>(out_shape, params.dtype, backend.get(), expected_data);
    auto output   = create_output_tensor<T>(out_shape, params.dtype, backend.get());

    dim3 block(8, 8, 8);
    dim3 grid(
        (params.outH + block.x - 1) / block.x,
        (params.outW + block.y - 1) / block.y,
        (params.inC + block.z - 1) / block.z
    );

    CUDANet::Kernels::avg_pool<<<grid, block>>>(
        static_cast<const T*>(input.device_ptr()),
        static_cast<T*>(output.device_ptr()),
        in_shape, out_shape, pool_shape, stride_shape, padding_shape
    );

    verify_output<T>(output, expected);
}

template void run_avg_pool_test<float>(const PoolParams params);

TEST_P(AvgPoolTest, AvgPooling) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_avg_pool_test<float>(param);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AvgPoolTestCases,
    AvgPoolTest,
    testing::ValuesIn(initialize_pool_params("/pool/avg_pool/metadata.csv"))
);
