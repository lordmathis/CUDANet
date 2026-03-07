#include "cudanet.hpp"
#include "gtest/gtest.h"

#include "test_utils.hpp"


#define CREATE_BACKEND() \
    CUDANet::BackendFactory::create(CUDANet::BackendType::CUDA_BACKEND, CUDANet::BackendConfig())

#define CREATE_TENSOR(name, shape, dtype, backend, data) \
    auto name = CUDANet::Tensor(shape, dtype, backend.get()); \
    name.set_data(data.data())

#define CREATE_OUTPUT_TENSOR(name, shape, dtype, backend) \
    auto name = CUDANet::Tensor(shape, dtype, backend.get()); \
    name.zero()

#define GRID_SIZE(size) ((size + BLOCK_SIZE - 1) / BLOCK_SIZE)

#define PARSE_DTYPE(row) \
    ((row[0] == "float32") ? CUDANet::DType::FLOAT32 : throw std::runtime_error("Unknown dtype: " + row[0]))

#define VERIFY_OUTPUT(output, expected) \
    cudaDeviceSynchronize(); \
    auto h_output = output.to_host<T>(); \
    auto h_expected = expected.to_host<T>(); \
    ASSERT_EQ(h_output.size(), h_expected.size()); \
    assert_elements_near(h_output, h_expected)

#define DEFINE_TEST(ParamsType, TestName, RunFn) \
    TEST_P(ParamsType, TestName) { \
        auto param = GetParam(); \
        if (param.dtype == CUDANet::DType::FLOAT32) { \
            RunFn<float>(param); \
        } \
    }


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
    auto matrix_data = load_binary<T>(params.matrix_path);
    auto vector_data = load_binary<T>(params.vector_path);
    auto expected_data = load_binary<T>(params.expected_path);

    auto backend = CREATE_BACKEND();

    auto matrix_shape = CUDANet::Shape{params.rows, params.cols};
    CREATE_TENSOR(matrix, matrix_shape, params.dtype, backend, matrix_data);

    auto vector_shape = CUDANet::Shape{params.cols};
    CREATE_TENSOR(vector, vector_shape, params.dtype, backend, vector_data);

    auto expected_shape = CUDANet::Shape{params.rows};
    CREATE_TENSOR(expected, expected_shape, params.dtype, backend, expected_data);

    CREATE_OUTPUT_TENSOR(output, expected_shape, params.dtype, backend);

    auto grid_size = GRID_SIZE(std::max(params.rows, params.cols));

    CUDANet::Kernels::mat_vec_mul<<<grid_size, BLOCK_SIZE>>>(
        static_cast<const T*>(matrix.device_ptr()),
        static_cast<const T*>(vector.device_ptr()),
        static_cast<T*>(output.device_ptr()),
        params.cols,
        params.rows
    );

    VERIFY_OUTPUT(output, expected);
}

template void run_mat_vec_mul_test<float>(const MatVecMulParams params);

DEFINE_TEST(MatVecMulTest, MatrixVectorMultiplication, run_mat_vec_mul_test);

std::vector<MatVecMulParams> initialize_mat_vec_mul_params() {
    std::vector<std::vector<std::string>> rows = load_csv(FIXTURE_PATH + "/matmul/mat_vec_mul/metadata.csv");

    std::vector<MatVecMulParams> params;

    for (const auto& row : rows) {
        CUDANet::DType dtype = PARSE_DTYPE(row);

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

    auto backend = CREATE_BACKEND();

    auto vector_shape = CUDANet::Shape{params.size};

    CREATE_TENSOR(vector_a, vector_shape, params.dtype, backend, vector_a_data);
    CREATE_TENSOR(vector_b, vector_shape, params.dtype, backend, vector_b_data);
    CREATE_TENSOR(expected, vector_shape, params.dtype, backend, expected_data);
    CREATE_OUTPUT_TENSOR(output, vector_shape, params.dtype, backend);

    auto grid_size = GRID_SIZE(params.size);

    CUDANet::Kernels::vec_vec_add<<<grid_size, BLOCK_SIZE>>>(
        static_cast<const T*>(vector_a.device_ptr()),
        static_cast<const T*>(vector_b.device_ptr()),
        static_cast<T*>(output.device_ptr()),
        params.size
    );

    VERIFY_OUTPUT(output, expected);
}

template void run_vec_vec_add_test<float>(const VecVecAddParams params);

DEFINE_TEST(VecVecAddTest, VectorVectorAddition, run_vec_vec_add_test);

std::vector<VecVecAddParams> initialize_vec_vec_add_params() {
    std::vector<std::vector<std::string>> rows = load_csv(FIXTURE_PATH + "/matmul/vec_vec_add/metadata.csv");

    std::vector<VecVecAddParams> params;

    for (const auto& row : rows) {
        CUDANet::DType dtype = PARSE_DTYPE(row);

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

INSTANTIATE_TEST_SUITE_P(  
    VecVecAddTestCases,  
    VecVecAddTest,  
    testing::ValuesIn(initialize_vec_vec_add_params())
);

/*

Vec Vec Sub

*/

struct VecVecSubParams
{
    CUDANet::DType dtype;
    const size_t size;
    std::string vec_a_path;
    std::string vec_b_path;
    std::string expected_path;
};

class VecVecSubTest : public ::testing::TestWithParam<VecVecSubParams> {};

template<typename T>
void run_vec_vec_sub_test(const VecVecSubParams params) {
    auto vector_a_data = load_binary<T>(params.vec_a_path);
    auto vector_b_data = load_binary<T>(params.vec_b_path);
    auto expected_data = load_binary<T>(params.expected_path);

    auto backend = CREATE_BACKEND();

    auto vector_shape = CUDANet::Shape{params.size};

    CREATE_TENSOR(vector_a, vector_shape, params.dtype, backend, vector_a_data);
    CREATE_TENSOR(vector_b, vector_shape, params.dtype, backend, vector_b_data);
    CREATE_TENSOR(expected, vector_shape, params.dtype, backend, expected_data);
    CREATE_OUTPUT_TENSOR(output, vector_shape, params.dtype, backend);

    auto grid_size = GRID_SIZE(params.size);

    CUDANet::Kernels::vec_vec_sub<<<grid_size, BLOCK_SIZE>>>(
        static_cast<const T*>(vector_a.device_ptr()),
        static_cast<const T*>(vector_b.device_ptr()),
        static_cast<T*>(output.device_ptr()),
        params.size
    );

    VERIFY_OUTPUT(output, expected);
}

template void run_vec_vec_sub_test<float>(const VecVecSubParams params);

DEFINE_TEST(VecVecSubTest, VectorVectorSubtraction, run_vec_vec_sub_test);

std::vector<VecVecSubParams> initialize_vec_vec_sub_params() {
    std::vector<std::vector<std::string>> rows = load_csv(FIXTURE_PATH + "/matmul/vec_vec_sub/metadata.csv");

    std::vector<VecVecSubParams> params;

    for (const auto& row : rows) {
        CUDANet::DType dtype = PARSE_DTYPE(row);

        size_t size = std::stoul(row[1]);

        params.push_back(VecVecSubParams{
            dtype,
            size,
            row[2],
            row[3],
            row[4]
        });
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(  
    VecVecSubTestCases,  
    VecVecSubTest,  
    testing::ValuesIn(initialize_vec_vec_sub_params())
);

/*

Vec Vec Mul

*/

struct VecVecMulParams
{
    CUDANet::DType dtype;
    const size_t size;
    std::string vec_a_path;
    std::string vec_b_path;
    std::string expected_path;
};

class VecVecMulTest : public ::testing::TestWithParam<VecVecMulParams> {};

template<typename T>
void run_vec_vec_mul_test(const VecVecMulParams params) {
    auto vector_a_data = load_binary<T>(params.vec_a_path);
    auto vector_b_data = load_binary<T>(params.vec_b_path);
    auto expected_data = load_binary<T>(params.expected_path);

    auto backend = CREATE_BACKEND();

    auto vector_shape = CUDANet::Shape{params.size};

    CREATE_TENSOR(vector_a, vector_shape, params.dtype, backend, vector_a_data);
    CREATE_TENSOR(vector_b, vector_shape, params.dtype, backend, vector_b_data);
    CREATE_TENSOR(expected, vector_shape, params.dtype, backend, expected_data);
    CREATE_OUTPUT_TENSOR(output, vector_shape, params.dtype, backend);

    auto grid_size = GRID_SIZE(params.size);

    CUDANet::Kernels::vec_vec_mul<<<grid_size, BLOCK_SIZE>>>(
        static_cast<const T*>(vector_a.device_ptr()),
        static_cast<const T*>(vector_b.device_ptr()),
        static_cast<T*>(output.device_ptr()),
        params.size
    );

    VERIFY_OUTPUT(output, expected);
}

template void run_vec_vec_mul_test<float>(const VecVecMulParams params);

DEFINE_TEST(VecVecMulTest, VectorVectorMultiplication, run_vec_vec_mul_test);

std::vector<VecVecMulParams> initialize_vec_vec_mul_params() {
    std::vector<std::vector<std::string>> rows = load_csv(FIXTURE_PATH + "/matmul/vec_vec_mul/metadata.csv");

    std::vector<VecVecMulParams> params;

    for (const auto& row : rows) {
        CUDANet::DType dtype = PARSE_DTYPE(row);

        size_t size = std::stoul(row[1]);

        params.push_back(VecVecMulParams{
            dtype,
            size,
            row[2],
            row[3],
            row[4]
        });
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(  
    VecVecMulTestCases,  
    VecVecMulTest,  
    testing::ValuesIn(initialize_vec_vec_mul_params())
);
