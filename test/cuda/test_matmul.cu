#include "cudanet.hpp"
#include "gtest/gtest.h"

#include "test_utils.hpp"

const std::string FIXTURES_PATH="matmul";

struct MatMulParams
{
    CUDANet::DType dtype;
    const int rows;
    const int cols;
    std::string matrix_path;
    std::string vector_path;
    std::string expected_path;
};

class MatVecMulTest : public ::testing::TestWithParam<MatMulParams> {};

TEST_P(MatVecMulTest, MatrixVectorMultiplication) {  
    auto param = GetParam(); // Get the current SumTestParams  

    // DType dispatch
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_matmul_test<float>();
    }
}

template<typename T>
void run_matmul_test(const MatMulParams params) {
    // Load binary data
    auto matrix_data = load_binary<T>(FIXTURES_PATH + "/" + params.matrix_path);
    auto vector_data = load_binary<T>(FIXTURES_PATH + "/" + params.vector_path);
    auto expected_data = load_binary<T>(FIXTURES_PATH + "/" + params.expected_path);

    auto backend = CUDANet::BackendFactory::create(CUDANet::BackendType::CUDA_BACKEND, CUDANet::BackendConfig());

    auto matrix_shape = CUDANet::Shape{params.rows, params.cols};
    auto matrix = CUDANet::Tensor(matrix_shape, params.dtype, backend);
    matrix.set_data(matrix_data.data());

    auto vector_shape = CUDANet::Shape{params.cols};
    auto vector = CUDANet::Tensor(vector_shape, params.dtype, backend);
    vector.set_data(vector_data.data());

    auto expected_shape = CUDANet::Shape{params.rows};
    auto expected = CUDANet::Tensor(expected_shape, params.dtype, backend);
    expected.set_data(expected_data.data());

    auto output = CUDANet::Tensor(expected_shape, params.dtype, backend);
    output::zero();

    auto BLOCK_SIZE = 256;
    auto grid_size =
        (std::max(params.rows, params.cols) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    Kernels::mat_vec_mul<<<grid_size, BLOCK_SIZE>>>(
        matrix.device_ptr(),
        vector.device_ptr(),
        output.device_ptr(),
        params.rows,
        params.cols
    );

    std::vector<T> h_output = output.to_host();
    std::vector<T> h_expected = expected.to_host();

    ASSERT_EQ(h_output.size(), h_expected.size());

    assert_elements_near(h_output, h_expected);
}

std::vector<MatMulParams> initialize_params() {
    std::vector<std::vector<std::string>> rows = load_csv("matmul/metadata.csv");

    std::vector<MatMulParams> params;

    for (const auto& row : rows) {
        CUDANet::DType dtype = (row[0] == "float32")
            ? CUDANet::DType::FLOAT32
            : throw std::runtime_error("Unknown dtype: " + row[0]);

        int rows = std::stoi(row[1]);
        int cols = std::stoi(row[2]);

        params.push_back(MatMulParams{
            dtype,
            rows,
            cols,
            row[3],
            row[4],
            row[5]
        });
    }

    return params;
}

// Instantiate with test cases  
INSTANTIATE_TEST_SUITE_P(  
    MatVecMulTestCases,  
    MatVecMulTest,  
    testing::ValuesIn(initialize_params())
); 
