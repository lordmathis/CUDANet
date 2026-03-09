#include <memory>
#include <vector>

#include "cudanet.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"

struct TensorParams {
    CUDANet::DType dtype;
    size_t         size;
    std::string    vector_path;
    std::string    expected_path;
};

std::vector<TensorParams> initialize_tensor_params(
    const std::string& csv_relative_path
) {
    try {
        std::vector<std::vector<std::string>> rows =
            load_csv(FIXTURE_PATH + csv_relative_path);
        std::vector<TensorParams> params;
        for (const auto& row : rows) {
            params.push_back(
                {parse_dtype(row), std::stoul(row[1]), row[2], row[3]}
            );
        }
        return params;
    } catch (const std::exception& e) {
        return {};
    }
}

class TensorMaxTest : public ::testing::TestWithParam<TensorParams> {};

template <typename T>
void run_tensor_max_test(const TensorParams params) {
    std::vector<std::unique_ptr<CUDANet::Backend>> backends;

#ifdef USE_CUDA
    backends.push_back(
        CUDANet::BackendFactory::create(
            CUDANet::BackendType::CUDA_BACKEND, CUDANet::BackendConfig()
        )
    );
#endif

    auto vector_data   = load_binary<T>(params.vector_path);
    auto expected_data = load_binary<T>(params.expected_path);

    for (auto& backend : backends) {
        auto vector = create_tensor<T>(
            CUDANet::Shape{params.size}, params.dtype, backend.get(),
            vector_data
        );
        auto expected = create_tensor<T>(
            CUDANet::Shape{1}, params.dtype, backend.get(), expected_data
        );

        auto output = create_output_tensor<T>(
            CUDANet::Shape{1}, params.dtype, backend.get()
        );

        backend->max(vector, output);

        verify_output<T>(output, expected);
    }
}

template void run_tensor_max_test<float>(const TensorParams params);

TEST_P(TensorMaxTest, TensorMax) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_tensor_max_test<float>(param);
    }
}

INSTANTIATE_TEST_SUITE_P(
    TensorMaxTestCases,
    TensorMaxTest,
    testing::ValuesIn(initialize_tensor_params("/tensor/max/metadata.csv"))
);
