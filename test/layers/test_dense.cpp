#include "cudanet.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"

TEST(TestDenseLayer, HandlesCorrectShape) {
    auto backends = get_backends();
    for (auto& backend : backends) {
        size_t in_size = 256;
        size_t out_size = 128;

        auto in_shape = CUDANet::Shape{in_size};
        auto out_shape = CUDANet::Shape{out_size};

        auto dense = CUDANet::Layers::Dense{
            in_shape,
            out_shape,
            backend.get()
        };

        EXPECT_EQ(dense.input_shape(), in_shape);
        EXPECT_EQ(dense.output_shape(), out_shape);
        
        EXPECT_EQ(dense.input_size(), in_size);
        EXPECT_EQ(dense.output_size(), out_size);

        auto dtype = backend.get()->get_default_dtype();
        auto dtype_size = CUDANet::dtype_size(dtype);

        EXPECT_EQ(dense.get_weights_size(), dtype_size * in_size * out_size);
        EXPECT_EQ(dense.get_biases_size(), dtype_size * out_size);
    }
}

TEST(TestDenseLayer, ThrowsExceptionsForIncorrectShapes) {
    auto backends = get_backends();
    for (auto& backend : backends) {
        EXPECT_THROW(
            (CUDANet::Layers::Dense{
                CUDANet::Shape{1,1},
                CUDANet::Shape{1},
                backend.get()
            }),
            CUDANet::InvalidShapeException
        );
        EXPECT_THROW(
            (CUDANet::Layers::Dense{
                CUDANet::Shape{1},
                CUDANet::Shape{1, 1},
                backend.get()
            }),
            CUDANet::InvalidShapeException
        );
    }
}

struct DenseParams {
    CUDANet::DType dtype;
    size_t         input_size;
    size_t         output_size;
    std::string    weights_path;
    std::string    biases_path;
    std::string    input_path;
    std::string    expected_path;
};

std::vector<DenseParams> initialize_dense_params(
    const std::string& csv_relative_path
) {
    try {
        std::vector<std::vector<std::string>> rows =
            load_csv(FIXTURE_PATH + csv_relative_path);
        std::vector<DenseParams> params;
        for (const auto& row : rows) {
            params.push_back({
                parse_dtype(row),
                std::stoul(row[1]),
                std::stoul(row[2]),
                row[3],
                row[4],
                row[5],
                row[6]
            });
        }
        return params;
    } catch (const std::exception& e) {
        return {};
    }
}

class DenseForwardTest : public ::testing::TestWithParam<DenseParams> {};

template <typename T>
void run_dense_forward_test(const DenseParams params) {
    auto backends = get_backends();

    auto weights_data   = load_binary<T>(params.weights_path);
    auto bias_data = load_binary<T>(params.biases_path);
    auto input_data = load_binary<T>(params.input_path);
    auto expected_data = load_binary<T>(params.expected_path);

    for (auto& backend : backends) {
        auto input = create_tensor<T>(
            CUDANet::Shape{params.input_size}, params.dtype, backend.get(),
            input_data
        );
        auto expected = create_tensor<T>(
            CUDANet::Shape{params.output_size}, params.dtype, backend.get(),
            expected_data
        );

        auto in_shape = CUDANet::Shape{params.input_size};
        auto out_shape = CUDANet::Shape{params.output_size};

        auto dense = CUDANet::Layers::Dense{
            in_shape,
            out_shape,
            backend.get()
        };
        dense.set_weights(static_cast<void*>(weights_data));
        dense.set_biases(static_cast<void*>(bias_data));

        auto output = dense.forward(input);

        verify_output<T>(output, expected);
    }
}

template void run_dense_forward_test<float>(const DenseParams params);

TEST_P(DenseForwardTest, LayerForward) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_dense_forward_test<float>(param);
    }
}

INSTANTIATE_TEST_SUITE_P(
    DenseForwardTestCases,
    DenseForwardTest,
    testing::ValuesIn(initialize_dense_params("/layers/dense/metadata.csv"))
);