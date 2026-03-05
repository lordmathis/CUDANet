#include "cudanet.hpp"
#include "gtest/gtest.h"

#include "test_utils.hpp"

/*

Mat Vec Mul

*/


struct MatVecMulParams
{
    CUDANet::DType dtype;
    const size_t rows;
    const size_t cols;
    std::string matrix_path;
    std::string vector_path;
    std::string expected_path;
};

class MatVecMulTest : public ::testing::TestWithParam<MatVecMulParams> {};

template<typename T>
void run_mat_vec_mul_test(const MatVecMulParams params) {
    // Load binary data
    auto matrix_data = load_binary<T>(params.matrix_path);
    auto vector_data = load_binary<T>(params.vector_path);
    auto expected_data = load_binary<T>(params.expected_path);

    auto backend = CUDANet::BackendFactory::create(CUDANet::BackendType::CUDA_BACKEND, CUDANet::BackendConfig());

    auto matrix_shape = CUDANet::Shape{params.rows, params.cols};
    auto matrix = CUDANet::Tensor(matrix_shape, params.dtype, backend.get());
    matrix.set_data(matrix_data.data());

    auto vector_shape = CUDANet::Shape{params.cols};
    auto vector = CUDANet::Tensor(vector_shape, params.dtype, backend.get());
    vector.set_data(vector_data.data());

    auto expected_shape = CUDANet::Shape{params.rows};
    auto expected = CUDANet::Tensor(expected_shape, params.dtype, backend.get());
    expected.set_data(expected_data.data());

    auto output = CUDANet::Tensor(expected_shape, params.dtype, backend.get());
    output.zero();

    auto grid_size =
        (std::max(params.rows, params.cols) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDANet::Kernels::mat_vec_mul<<<grid_size, BLOCK_SIZE>>>(
        static_cast<const T*>(matrix.device_ptr()),
        static_cast<const T*>(vector.device_ptr()),
        static_cast<T*>(output.device_ptr()),
        params.cols,
        params.rows
    );
    cudaDeviceSynchronize();

    std::vector<T> h_output = output.to_host<T>();
    std::vector<T> h_expected = expected.to_host<T>();

    ASSERT_EQ(h_output.size(), h_expected.size());

    assert_elements_near(h_output, h_expected);
}

template void run_mat_vec_mul_test<float>(const MatVecMulParams params);

TEST_P(MatVecMulTest, MatrixVectorMultiplication) {  
    auto param = GetParam(); 

    // DType dispatch
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_mat_vec_mul_test<float>(param);
    }
}

std::vector<MatVecMulParams> initialize_mat_vec_mul_params() {
    std::vector<std::vector<std::string>> rows = load_csv(FIXTURE_PATH + "/matmul/mat_vec_mul/metadata.csv");

    std::vector<MatVecMulParams> params;

    for (const auto& row : rows) {
        CUDANet::DType dtype = (row[0] == "float32")
            ? CUDANet::DType::FLOAT32
            : throw std::runtime_error("Unknown dtype: " + row[0]);

        size_t rows = std::stoul(row[1]);
        size_t cols = std::stoul(row[2]);

        params.push_back(MatVecMulParams{
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
    testing::ValuesIn(initialize_mat_vec_mul_params())
); 

/*

Vec Vec Add

*/

struct VecVecAddParams
{
    CUDANet::DType dtype;
    const size_t size;
    std::string vec_a_path;
    std::string vec_b_path;
    std::string expected_path;
};

class VecVecAddTest : public ::testing::TestWithParam<VecVecAddParams> {};

template<typename T>
void run_vec_vec_add_test(const VecVecAddParams params) {
    auto vector_a_data = load_binary<T>(params.vec_a_path);
    auto vector_b_data = load_binary<T>(params.vec_b_path);
    auto expected_data = load_binary<T>(params.expected_path);

    auto backend = CUDANet::BackendFactory::create(CUDANet::BackendType::CUDA_BACKEND, CUDANet::BackendConfig());

    auto vector_shape = CUDANet::Shape{params.size};

    auto vector_a = CUDANet::Tensor(vector_shape, params.dtype, backend.get());
    vector_a.set_data(vector_a_data.data());

    auto vector_b = CUDANet::Tensor(vector_shape, params.dtype, backend.get());
    vector_b.set_data(vector_b_data.data());

    auto expected = CUDANet::Tensor(vector_shape, params.dtype, backend.get());
    expected.set_data(expected_data.data());

    auto output = CUDANet::Tensor(vector_shape, params.dtype, backend.get());
    output.zero();

    auto grid_size =
        (params.size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDANet::Kernels::vec_vec_add<<<grid_size, BLOCK_SIZE>>>(
        static_cast<const T*>(vector_a.device_ptr()),
        static_cast<const T*>(vector_b.device_ptr()),
        static_cast<T*>(output.device_ptr()),
        params.size
    );
    cudaDeviceSynchronize();

    std::vector<T> h_output = output.to_host<T>();
    std::vector<T> h_expected = expected.to_host<T>();

    ASSERT_EQ(h_output.size(), h_expected.size());

    assert_elements_near(h_output, h_expected);
}

template void run_vec_vec_add_test<float>(const VecVecAddParams params);

TEST_P(VecVecAddTest, VectorVectorAddition) {  
    auto param = GetParam(); 

    // DType dispatch
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_vec_vec_add_test<float>(param);
    }
}

std::vector<VecVecAddParams> initialize_vec_vec_add_params() {
    std::vector<std::vector<std::string>> rows = load_csv(FIXTURE_PATH + "/matmul/vec_vec_add/metadata.csv");

    std::vector<VecVecAddParams> params;

    for (const auto& row : rows) {
        CUDANet::DType dtype = (row[0] == "float32")
            ? CUDANet::DType::FLOAT32
            : throw std::runtime_error("Unknown dtype: " + row[0]);

        size_t size = std::stoul(row[1]);

        params.push_back(VecVecAddParams{
            dtype,
            size,
            row[2],
            row[3],
            row[4]
        });
    }

    return params;
}

// Instantiate with test cases  
INSTANTIATE_TEST_SUITE_P(  
    VecVecAddTestCases,  
    VecVecAddTest,  
    testing::ValuesIn(initialize_vec_vec_add_params())
); 