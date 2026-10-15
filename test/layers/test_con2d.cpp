#include "cudanet.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"

TEST(TestConv2dLayer, HandlesCorrectShape) {
    auto backends = get_backends();

    for (auto& backend : backends) {
        auto input_shape   = CUDANet::Shape{128, 128, 3};
        auto kernel_shape  = CUDANet::Shape{5, 5, 64};
        auto stride_shape  = CUDANet::Shape{2, 2};
        auto padding_shape = CUDANet::Shape{2, 2};

        auto conv2d = CUDANet::Layers::Conv2d{
            input_shape, kernel_shape, stride_shape, padding_shape,
            backend.get()
        };

        auto dtype      = backend.get()->get_default_dtype();
        auto dtype_size = CUDANet::dtype_size(dtype);

        EXPECT_EQ(
            conv2d.get_weights_size(), dtype_size * kernel_shape[0] *
                                           kernel_shape[1] * kernel_shape[2] *
                                           input_shape[2]
        );
        EXPECT_EQ(conv2d.get_biases_size(), dtype_size * kernel_shape[2]);

        auto output_shape = conv2d.output_shape();
        EXPECT_EQ(output_shape.ndim, 3);
        EXPECT_EQ(
            output_shape[0],
            (input_shape[0] - kernel_shape[0] + 2 * padding_shape[0]) /
                    stride_shape[0] +
                1
        );
        EXPECT_EQ(
            output_shape[1],
            (input_shape[1] - kernel_shape[1] + 2 * padding_shape[1]) /
                    stride_shape[1] +
                1
        );
        EXPECT_EQ(output_shape[2], kernel_shape[2]);
    }
}

TEST(TestConv2dLayer, ThrowsExceptionsForIncorrectShapes) {
    auto backends = get_backends();
    for (auto& backend : backends) {
        auto kernel_shape  = CUDANet::Shape{3, 3, 16};
        auto stride_shape  = CUDANet::Shape{1, 1};
        auto padding_shape = CUDANet::Shape{0, 0};

        EXPECT_THROW(
            (CUDANet::Layers::Conv2d{
                CUDANet::Shape{8, 8},
                kernel_shape,
                stride_shape,
                padding_shape,
                backend.get()
            }),
            CUDANet::InvalidShapeException
        );

        auto input_shape = CUDANet::Shape{8, 8, 3};

        EXPECT_THROW(
            (CUDANet::Layers::Conv2d{
                input_shape,
                CUDANet::Shape{3, 3},
                stride_shape,
                padding_shape,
                backend.get()
            }),
            CUDANet::InvalidShapeException
        );

        EXPECT_THROW(
            (CUDANet::Layers::Conv2d{
                input_shape,
                kernel_shape,
                CUDANet::Shape{1, 1, 1},
                padding_shape,
                backend.get()
            }),
            CUDANet::InvalidShapeException
        );

        EXPECT_THROW(
            (CUDANet::Layers::Conv2d{
                input_shape,
                kernel_shape,
                stride_shape,
                CUDANet::Shape{0},
                backend.get()
            }),
            CUDANet::InvalidShapeException
        );
    }
}

struct Conv2dParams {
    CUDANet::DType dtype;
    size_t         input_0;
    size_t         input_1;
    size_t         input_2;
    size_t         kernel_0;
    size_t         kernel_1;
    size_t         kernel_2;
    size_t         stride_0;
    size_t         stride_1;
    size_t         padding_0;
    size_t         padding_1;
    std::string    weights_path;
    std::string    biases_path;
    std::string    input_path;
    std::string    expected_path;
};

std::vector<Conv2dParams> initialize_conv2d_params(
    const std::string& csv_relative_path
) {
    std::vector<std::vector<std::string>> rows =
    load_csv(FIXTURE_PATH + csv_relative_path);

    std::vector<Conv2dParams> params;
    for (const auto& row : rows) {
        params.push_back({
            parse_dtype(row),
            std::stoul(row[1]),
            std::stoul(row[2]),
            std::stoul(row[3]),
            std::stoul(row[4]),
            std::stoul(row[5]),
            std::stoul(row[6]),
            std::stoul(row[7]),
            std::stoul(row[8]),
            std::stoul(row[9]),
            std::stoul(row[10]),
            row[11],
            row[12],
            row[13],
            row[14]
        });
    }
    if (params.empty()) {
        throw std::runtime_error(
            "No test cases found in: " + FIXTURE_PATH + csv_relative_path
        );
    }
    return params;
}

class Conv2dForwardTest : public ::testing::TestWithParam<Conv2dParams> {};

template <typename T>
void run_conv2d_forward_test(const Conv2dParams params) {
    auto backends = get_backends();

    auto weights_data  = load_binary<T>(params.weights_path);
    auto bias_data     = load_binary<T>(params.biases_path);
    auto input_data    = load_binary<T>(params.input_path);
    auto expected_data = load_binary<T>(params.expected_path);

    auto input_shape = CUDANet::Shape{params.input_0, params.input_1, params.input_2};
    auto kernel_shape = CUDANet::Shape{params.kernel_0, params.kernel_1, params.kernel_2};
    auto stride_shape = CUDANet::Shape{params.stride_0, params.stride_1};
    auto padding_shape = CUDANet::Shape{params.padding_0, params.padding_1};

    for (auto& backend : backends) {

        auto conv2d = CUDANet::Layers::Conv2d{
            input_shape, kernel_shape, stride_shape, padding_shape, backend.get()
        };
        conv2d.set_weights(weights_data.data());
        conv2d.set_biases(bias_data.data());

        auto input = create_tensor<T>(
            input_shape, params.dtype, backend.get(),
            input_data
        );

        auto output_shape = conv2d.output_shape();
        auto expected = create_tensor<T>(
            output_shape, params.dtype, backend.get(),
            expected_data
        );

        auto& output = conv2d.forward(input);

        verify_output<T>(output, expected);
    }
}

template void run_conv2d_forward_test<float>(const Conv2dParams params);

TEST_P(Conv2dForwardTest, LayerForward) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_conv2d_forward_test<float>(param);
    } else {
        GTEST_SKIP() << "Unsupported dtype";
    }
}

INSTANTIATE_TEST_SUITE_P(
    Conv2dForwardTestCases,
    Conv2dForwardTest,
    testing::ValuesIn(initialize_conv2d_params("/layers/conv2d/metadata.csv"))
);