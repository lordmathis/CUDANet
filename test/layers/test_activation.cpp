#include "cudanet.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"

TEST(TestActivationLayer, HandlesCorrectShape) {
    auto backends = get_backends();

    for (auto activation_type : CUDANet::Layers::ALL_ACTIVATIONS) {
        for (auto& backend : backends) {
            auto in_shape  = CUDANet::Shape{128};

            auto activation = CUDANet::Layers::Activation{
                activation_type,
                in_shape,
                backend.get()
            };

            EXPECT_EQ(activation.input_shape(), in_shape);
            EXPECT_EQ(activation.output_shape(), in_shape);

            auto dtype      = backend.get()->get_default_dtype();
            auto dtype_size = CUDANet::dtype_size(dtype);

            EXPECT_EQ(activation.input_size(), in_shape[0] * dtype_size);
            EXPECT_EQ(activation.output_size(), in_shape[0] * dtype_size);
        }
    }
}

TEST(TestActivationLayer, ThrowsExceptionsForIncorrectShapes) {
    auto backends = get_backends();

    for (auto activation_type : CUDANet::Layers::ALL_ACTIVATIONS) {
        for (auto& backend : backends) {
            EXPECT_THROW(
                (CUDANet::Layers::Activation{
                    activation_type,
                    CUDANet::Shape{1, 1},
                    backend.get()
                }),
                CUDANet::InvalidShapeException
            );
        }
    }
}

struct ActivationParams {
    CUDANet::DType                  dtype;
    CUDANet::Layers::ActivationType activation_type;
    size_t                          size;
    std::string                     input_path;
    std::string                     expected_path;
};

std::vector<ActivationParams> initialize_activation_params(
    const std::string& csv_relative_path
) {
    std::vector<std::vector<std::string>> rows =
        load_csv(FIXTURE_PATH + csv_relative_path);
    std::vector<ActivationParams> params;
    for (const auto& row : rows) {
        params.push_back(
            {parse_dtype(row),
             CUDANet::Layers::str2at(row[1]), std::stoul(row[2]), row[3], row[4]}
        );
    }
    if (params.empty()) {
        throw std::runtime_error(
            "No test cases found in: " + FIXTURE_PATH + csv_relative_path
        );
    }
    return params;
}

class ActivationForwardTest
    : public ::testing::TestWithParam<ActivationParams> {};

template <typename T>
void run_activation_forward_test(const ActivationParams params) {
    // TODO: Implement forward test
}

template void run_activation_forward_test<float>(const ActivationParams params);

TEST_P(ActivationForwardTest, LayerForward) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_activation_forward_test<float>(param);
    } else {
        GTEST_SKIP() << "Unsupported dtype";
    }
}

INSTANTIATE_TEST_SUITE_P(
    ActivationForwardTestCases,
    ActivationForwardTest,
    testing::ValuesIn(
        initialize_activation_params("/layers/activation/metadata.csv")
    )
);

