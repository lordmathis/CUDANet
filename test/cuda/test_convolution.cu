#include "cudanet.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"
#include "cuda_test_utils.hpp"

class ConvolutionTest : public ::testing::TestWithParam<ConvolutionParams> {};

template <typename T>
void run_convolution_test(const ConvolutionParams params) {
    auto input_data    = load_binary<T>(params.input_path);
    auto kernel_data   = load_binary<T>(params.kernel_path);
    auto bias_data     = load_binary<T>(params.bias_path);
    auto expected_data = load_binary<T>(params.expected_path);

    auto backend = create_backend();

    auto input_shape = CUDANet::Shape{params.inH, params.inW, params.inC};
    auto kernel_shape = CUDANet::Shape{params.kH, params.kW};
    auto padding_shape = CUDANet::Shape{params.pH, params.pW};
    auto stride_shape = CUDANet::Shape{params.sH, params.sW};
    auto output_shape = CUDANet::Shape{params.outH, params.outW, params.outC};

    auto input = create_tensor<T>(
        input_shape, params.dtype, backend.get(), input_data
    );
    auto kernel = create_tensor<T>(
        CUDANet::Shape{params.outC, params.inC, params.kH, params.kW},
        params.dtype, backend.get(), kernel_data
    );
    auto bias = create_tensor<T>(
        CUDANet::Shape{params.outC}, params.dtype, backend.get(), bias_data
    );
    auto expected = create_tensor<T>(
        output_shape, params.dtype, backend.get(), expected_data
    );
    auto output =
        create_output_tensor<T>(output_shape, params.dtype, backend.get());

    dim3 block_size(16, 16, 1);
    dim3 grid_size(
        (params.outW + block_size.x - 1) / block_size.x,
        (params.outH + block_size.y - 1) / block_size.y,
        (params.outC + block_size.z - 1) / block_size.z
    );

    CUDANet::Kernels::convolution<<<grid_size, block_size>>>(
        static_cast<const T*>(input.device_ptr()),
        static_cast<const T*>(kernel.device_ptr()),
        static_cast<const T*>(bias.device_ptr()),
        static_cast<T*>(output.device_ptr()),
        input_shape,
        padding_shape,
        kernel_shape,
        stride_shape,
        output_shape
    );

    verify_output<T>(output, expected);
}

template void run_convolution_test<float>(const ConvolutionParams params);

TEST_P(ConvolutionTest, ConvolutionForward) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_convolution_test<float>(param);
    }
}

INSTANTIATE_TEST_SUITE_P(
    ConvolutionTestCases,
    ConvolutionTest,
    testing::ValuesIn(initialize_convolution_params("/convolution/metadata.csv"))
);
