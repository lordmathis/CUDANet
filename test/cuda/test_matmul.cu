#include "cudanet.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"

inline auto create_backend() {
    return CUDANet::BackendFactory::create(
        CUDANet::BackendType::CUDA_BACKEND, CUDANet::BackendConfig()
    );
}

template <typename T>
inline CUDANet::Tensor create_tensor(
    const CUDANet::Shape& shape,
    CUDANet::DType        dtype,
    CUDANet::Backend*     backend,
    const std::vector<T>& data
) {
    auto tensor = CUDANet::Tensor(shape, dtype, backend);
    tensor.set_data(static_cast<void*>(const_cast<T*>(data.data())));
    return tensor;
}

template <typename T>
inline CUDANet::Tensor create_output_tensor(
    const CUDANet::Shape& shape,
    CUDANet::DType        dtype,
    CUDANet::Backend*     backend
) {
    auto tensor = CUDANet::Tensor(shape, dtype, backend);
    tensor.zero();
    return tensor;
}

inline size_t calc_grid_size(size_t size) {
    return (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

inline CUDANet::DType parse_dtype(const std::vector<std::string>& row) {
    if (row[0] == "float32") return CUDANet::DType::FLOAT32;
    throw std::runtime_error("Unknown dtype: " + row[0]);
}

template <typename T>
inline void verify_output(CUDANet::Tensor& output, CUDANet::Tensor& expected) {
    cudaDeviceSynchronize();
    auto h_output   = output.to_host<T>();
    auto h_expected = expected.to_host<T>();
    ASSERT_EQ(h_output.size(), h_expected.size());
    assert_elements_near(h_output, h_expected);
}

/*

Mat Vec Mul

*/

struct MatVecMulParams {
    CUDANet::DType dtype;
    const size_t   rows;
    const size_t   cols;
    std::string    matrix_path;
    std::string    vector_path;
    std::string    expected_path;
};

class MatVecMulTest : public ::testing::TestWithParam<MatVecMulParams> {};

template <typename T>
void run_mat_vec_mul_test(const MatVecMulParams params) {
    auto matrix_data   = load_binary<T>(params.matrix_path);
    auto vector_data   = load_binary<T>(params.vector_path);
    auto expected_data = load_binary<T>(params.expected_path);

    auto backend = create_backend();

    auto matrix_shape = CUDANet::Shape{params.rows, params.cols};
    auto matrix       = create_tensor<T>(
        matrix_shape, params.dtype, backend.get(), matrix_data
    );

    auto vector_shape = CUDANet::Shape{params.cols};
    auto vector       = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), vector_data
    );

    auto expected_shape = CUDANet::Shape{params.rows};
    auto expected       = create_tensor<T>(
        expected_shape, params.dtype, backend.get(), expected_data
    );

    auto output =
        create_output_tensor<T>(expected_shape, params.dtype, backend.get());

    auto grid_size = calc_grid_size(std::max(params.rows, params.cols));

    CUDANet::Kernels::mat_vec_mul<<<grid_size, BLOCK_SIZE>>>(
        static_cast<const T*>(matrix.device_ptr()),
        static_cast<const T*>(vector.device_ptr()),
        static_cast<T*>(output.device_ptr()), params.cols, params.rows
    );

    verify_output<T>(output, expected);
}

template void run_mat_vec_mul_test<float>(const MatVecMulParams params);

TEST_P(MatVecMulTest, MatrixVectorMultiplication) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_mat_vec_mul_test<float>(param);
    }
}

std::vector<MatVecMulParams> initialize_mat_vec_mul_params() {
    std::vector<std::vector<std::string>> rows =
        load_csv(FIXTURE_PATH + "/matmul/mat_vec_mul/metadata.csv");

    std::vector<MatVecMulParams> params;

    for (const auto& row : rows) {
        CUDANet::DType dtype = parse_dtype(row);

        size_t rows = std::stoul(row[1]);
        size_t cols = std::stoul(row[2]);

        params.push_back(
            MatVecMulParams{dtype, rows, cols, row[3], row[4], row[5]}
        );
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

struct VecVecAddParams {
    CUDANet::DType dtype;
    const size_t   size;
    std::string    vec_a_path;
    std::string    vec_b_path;
    std::string    expected_path;
};

class VecVecAddTest : public ::testing::TestWithParam<VecVecAddParams> {};

template <typename T>
void run_vec_vec_add_test(const VecVecAddParams params) {
    auto vector_a_data = load_binary<T>(params.vec_a_path);
    auto vector_b_data = load_binary<T>(params.vec_b_path);
    auto expected_data = load_binary<T>(params.expected_path);

    auto backend = create_backend();

    auto vector_shape = CUDANet::Shape{params.size};

    auto vector_a = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), vector_a_data
    );
    auto vector_b = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), vector_b_data
    );
    auto expected = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), expected_data
    );
    auto output =
        create_output_tensor<T>(vector_shape, params.dtype, backend.get());

    auto grid_size = calc_grid_size(params.size);

    CUDANet::Kernels::vec_vec_add<<<grid_size, BLOCK_SIZE>>>(
        static_cast<const T*>(vector_a.device_ptr()),
        static_cast<const T*>(vector_b.device_ptr()),
        static_cast<T*>(output.device_ptr()), params.size
    );

    verify_output<T>(output, expected);
}

template void run_vec_vec_add_test<float>(const VecVecAddParams params);

TEST_P(VecVecAddTest, VectorVectorAddition) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_vec_vec_add_test<float>(param);
    }
}

std::vector<VecVecAddParams> initialize_vec_vec_add_params() {
    std::vector<std::vector<std::string>> rows =
        load_csv(FIXTURE_PATH + "/matmul/vec_vec_add/metadata.csv");

    std::vector<VecVecAddParams> params;

    for (const auto& row : rows) {
        CUDANet::DType dtype = parse_dtype(row);

        size_t size = std::stoul(row[1]);

        params.push_back(VecVecAddParams{dtype, size, row[2], row[3], row[4]});
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

struct VecVecSubParams {
    CUDANet::DType dtype;
    const size_t   size;
    std::string    vec_a_path;
    std::string    vec_b_path;
    std::string    expected_path;
};

class VecVecSubTest : public ::testing::TestWithParam<VecVecSubParams> {};

template <typename T>
void run_vec_vec_sub_test(const VecVecSubParams params) {
    auto vector_a_data = load_binary<T>(params.vec_a_path);
    auto vector_b_data = load_binary<T>(params.vec_b_path);
    auto expected_data = load_binary<T>(params.expected_path);

    auto backend = create_backend();

    auto vector_shape = CUDANet::Shape{params.size};

    auto vector_a = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), vector_a_data
    );
    auto vector_b = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), vector_b_data
    );
    auto expected = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), expected_data
    );
    auto output =
        create_output_tensor<T>(vector_shape, params.dtype, backend.get());

    auto grid_size = calc_grid_size(params.size);

    CUDANet::Kernels::vec_vec_sub<<<grid_size, BLOCK_SIZE>>>(
        static_cast<const T*>(vector_a.device_ptr()),
        static_cast<const T*>(vector_b.device_ptr()),
        static_cast<T*>(output.device_ptr()), params.size
    );

    verify_output<T>(output, expected);
}

template void run_vec_vec_sub_test<float>(const VecVecSubParams params);

TEST_P(VecVecSubTest, VectorVectorSubtraction) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_vec_vec_sub_test<float>(param);
    }
}

std::vector<VecVecSubParams> initialize_vec_vec_sub_params() {
    std::vector<std::vector<std::string>> rows =
        load_csv(FIXTURE_PATH + "/matmul/vec_vec_sub/metadata.csv");

    std::vector<VecVecSubParams> params;

    for (const auto& row : rows) {
        CUDANet::DType dtype = parse_dtype(row);

        size_t size = std::stoul(row[1]);

        params.push_back(VecVecSubParams{dtype, size, row[2], row[3], row[4]});
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

struct VecVecMulParams {
    CUDANet::DType dtype;
    const size_t   size;
    std::string    vec_a_path;
    std::string    vec_b_path;
    std::string    expected_path;
};

class VecVecMulTest : public ::testing::TestWithParam<VecVecMulParams> {};

template <typename T>
void run_vec_vec_mul_test(const VecVecMulParams params) {
    auto vector_a_data = load_binary<T>(params.vec_a_path);
    auto vector_b_data = load_binary<T>(params.vec_b_path);
    auto expected_data = load_binary<T>(params.expected_path);

    auto backend = create_backend();

    auto vector_shape = CUDANet::Shape{params.size};

    auto vector_a = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), vector_a_data
    );
    auto vector_b = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), vector_b_data
    );
    auto expected = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), expected_data
    );
    auto output =
        create_output_tensor<T>(vector_shape, params.dtype, backend.get());

    auto grid_size = calc_grid_size(params.size);

    CUDANet::Kernels::vec_vec_mul<<<grid_size, BLOCK_SIZE>>>(
        static_cast<const T*>(vector_a.device_ptr()),
        static_cast<const T*>(vector_b.device_ptr()),
        static_cast<T*>(output.device_ptr()), params.size
    );

    verify_output<T>(output, expected);
}

template void run_vec_vec_mul_test<float>(const VecVecMulParams params);

TEST_P(VecVecMulTest, VectorVectorMultiplication) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_vec_vec_mul_test<float>(param);
    }
}

std::vector<VecVecMulParams> initialize_vec_vec_mul_params() {
    std::vector<std::vector<std::string>> rows =
        load_csv(FIXTURE_PATH + "/matmul/vec_vec_mul/metadata.csv");

    std::vector<VecVecMulParams> params;

    for (const auto& row : rows) {
        CUDANet::DType dtype = parse_dtype(row);

        size_t size = std::stoul(row[1]);

        params.push_back(VecVecMulParams{dtype, size, row[2], row[3], row[4]});
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    VecVecMulTestCases,
    VecVecMulTest,
    testing::ValuesIn(initialize_vec_vec_mul_params())
);

/*

Vec Scalar Sub

*/

struct VecScalarSubParams {
    CUDANet::DType dtype;
    const size_t   size;
    std::string    vector_path;
    std::string    scalar_path;
    std::string    expected_path;
};

class VecScalarSubTest : public ::testing::TestWithParam<VecScalarSubParams> {};

template <typename T>
void run_vec_scalar_sub_test(const VecScalarSubParams params) {
    auto vector_data   = load_binary<T>(params.vector_path);
    auto scalar_data   = load_binary<T>(params.scalar_path);
    auto expected_data = load_binary<T>(params.expected_path);

    auto backend = create_backend();

    auto vector_shape = CUDANet::Shape{params.size};

    auto vector = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), vector_data
    );
    auto scalar = create_tensor<T>(
        CUDANet::Shape{1}, params.dtype, backend.get(), scalar_data
    );
    auto expected = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), expected_data
    );
    auto output =
        create_output_tensor<T>(vector_shape, params.dtype, backend.get());

    auto grid_size = calc_grid_size(params.size);

    CUDANet::Kernels::vec_scalar_sub<<<grid_size, BLOCK_SIZE>>>(
        static_cast<const T*>(vector.device_ptr()),
        static_cast<T*>(output.device_ptr()),
        static_cast<const T*>(scalar.device_ptr()), params.size
    );

    verify_output<T>(output, expected);
}

template void run_vec_scalar_sub_test<float>(const VecScalarSubParams params);

TEST_P(VecScalarSubTest, VectorScalarSubtraction) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_vec_scalar_sub_test<float>(param);
    }
}

std::vector<VecScalarSubParams> initialize_vec_scalar_sub_params() {
    std::vector<std::vector<std::string>> rows =
        load_csv(FIXTURE_PATH + "/matmul/vec_scalar_sub/metadata.csv");

    std::vector<VecScalarSubParams> params;

    for (const auto& row : rows) {
        CUDANet::DType dtype = parse_dtype(row);

        size_t size = std::stoul(row[1]);

        params.push_back(
            VecScalarSubParams{dtype, size, row[2], row[3], row[4]}
        );
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    VecScalarSubTestCases,
    VecScalarSubTest,
    testing::ValuesIn(initialize_vec_scalar_sub_params())
);

/*

Vec Scalar Add

*/

struct VecScalarAddParams {
    CUDANet::DType dtype;
    const size_t   size;
    std::string    vector_path;
    std::string    scalar_path;
    std::string    expected_path;
};

class VecScalarAddTest : public ::testing::TestWithParam<VecScalarAddParams> {};

template <typename T>
void run_vec_scalar_add_test(const VecScalarAddParams params) {
    auto vector_data   = load_binary<T>(params.vector_path);
    auto scalar_data   = load_binary<T>(params.scalar_path);
    auto expected_data = load_binary<T>(params.expected_path);

    auto backend = create_backend();

    auto vector_shape = CUDANet::Shape{params.size};

    auto vector = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), vector_data
    );
    auto scalar = create_tensor<T>(
        CUDANet::Shape{1}, params.dtype, backend.get(), scalar_data
    );
    auto expected = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), expected_data
    );
    auto output =
        create_output_tensor<T>(vector_shape, params.dtype, backend.get());

    auto grid_size = calc_grid_size(params.size);

    CUDANet::Kernels::vec_scalar_add<<<grid_size, BLOCK_SIZE>>>(
        static_cast<const T*>(vector.device_ptr()),
        static_cast<T*>(output.device_ptr()),
        static_cast<const T*>(scalar.device_ptr()), params.size
    );

    verify_output<T>(output, expected);
}

template void run_vec_scalar_add_test<float>(const VecScalarAddParams params);

TEST_P(VecScalarAddTest, VectorScalarAddition) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_vec_scalar_add_test<float>(param);
    }
}

std::vector<VecScalarAddParams> initialize_vec_scalar_add_params() {
    std::vector<std::vector<std::string>> rows =
        load_csv(FIXTURE_PATH + "/matmul/vec_scalar_add/metadata.csv");

    std::vector<VecScalarAddParams> params;

    for (const auto& row : rows) {
        CUDANet::DType dtype = parse_dtype(row);

        size_t size = std::stoul(row[1]);

        params.push_back(
            VecScalarAddParams{dtype, size, row[2], row[3], row[4]}
        );
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    VecScalarAddTestCases,
    VecScalarAddTest,
    testing::ValuesIn(initialize_vec_scalar_add_params())
);

/*

Vec Scalar Div

*/

struct VecScalarDivParams {
    CUDANet::DType dtype;
    const size_t   size;
    std::string    vector_path;
    std::string    scalar_path;
    std::string    expected_path;
};

class VecScalarDivTest : public ::testing::TestWithParam<VecScalarDivParams> {};

template <typename T>
void run_vec_scalar_div_test(const VecScalarDivParams params) {
    auto vector_data   = load_binary<T>(params.vector_path);
    auto scalar_data   = load_binary<T>(params.scalar_path);
    auto expected_data = load_binary<T>(params.expected_path);

    auto backend = create_backend();

    auto vector_shape = CUDANet::Shape{params.size};

    auto vector = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), vector_data
    );
    auto scalar = create_tensor<T>(
        CUDANet::Shape{1}, params.dtype, backend.get(), scalar_data
    );
    auto expected = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), expected_data
    );
    auto output =
        create_output_tensor<T>(vector_shape, params.dtype, backend.get());

    auto grid_size = calc_grid_size(params.size);

    CUDANet::Kernels::vec_scalar_div<<<grid_size, BLOCK_SIZE>>>(
        static_cast<const T*>(vector.device_ptr()),
        static_cast<T*>(output.device_ptr()),
        static_cast<const T*>(scalar.device_ptr()), params.size
    );

    verify_output<T>(output, expected);
}

template void run_vec_scalar_div_test<float>(const VecScalarDivParams params);

TEST_P(VecScalarDivTest, VectorScalarDivision) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_vec_scalar_div_test<float>(param);
    }
}

std::vector<VecScalarDivParams> initialize_vec_scalar_div_params() {
    std::vector<std::vector<std::string>> rows =
        load_csv(FIXTURE_PATH + "/matmul/vec_scalar_div/metadata.csv");

    std::vector<VecScalarDivParams> params;

    for (const auto& row : rows) {
        CUDANet::DType dtype = parse_dtype(row);

        size_t size = std::stoul(row[1]);

        params.push_back(
            VecScalarDivParams{dtype, size, row[2], row[3], row[4]}
        );
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    VecScalarDivTestCases,
    VecScalarDivTest,
    testing::ValuesIn(initialize_vec_scalar_div_params())
);

/*

Vec Scalar Div

*/

struct VecScalarMulParams {
    CUDANet::DType dtype;
    const size_t   size;
    std::string    vector_path;
    std::string    scalar_path;
    std::string    expected_path;
};

class VecScalarMulTest : public ::testing::TestWithParam<VecScalarMulParams> {};

template <typename T>
void run_vec_scalar_mul_test(const VecScalarMulParams params) {
    auto vector_data   = load_binary<T>(params.vector_path);
    auto scalar_data   = load_binary<T>(params.scalar_path);
    auto expected_data = load_binary<T>(params.expected_path);

    auto backend = create_backend();

    auto vector_shape = CUDANet::Shape{params.size};

    auto vector = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), vector_data
    );
    auto scalar = create_tensor<T>(
        CUDANet::Shape{1}, params.dtype, backend.get(), scalar_data
    );
    auto expected = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), expected_data
    );
    auto output =
        create_output_tensor<T>(vector_shape, params.dtype, backend.get());

    auto grid_size = calc_grid_size(params.size);

    CUDANet::Kernels::vec_scalar_mul<<<grid_size, BLOCK_SIZE>>>(
        static_cast<const T*>(vector.device_ptr()),
        static_cast<T*>(output.device_ptr()),
        static_cast<const T*>(scalar.device_ptr()), params.size
    );

    verify_output<T>(output, expected);
}

template void run_vec_scalar_mul_test<float>(const VecScalarMulParams params);

TEST_P(VecScalarMulTest, VectorScalarMultiplication) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_vec_scalar_mul_test<float>(param);
    }
}

std::vector<VecScalarMulParams> initialize_vec_scalar_mul_params() {
    std::vector<std::vector<std::string>> rows =
        load_csv(FIXTURE_PATH + "/matmul/vec_scalar_mul/metadata.csv");

    std::vector<VecScalarMulParams> params;

    for (const auto& row : rows) {
        CUDANet::DType dtype = parse_dtype(row);

        size_t size = std::stoul(row[1]);

        params.push_back(
            VecScalarMulParams{dtype, size, row[2], row[3], row[4]}
        );
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    VecScalarMulTestCases,
    VecScalarMulTest,
    testing::ValuesIn(initialize_vec_scalar_mul_params())
);

/*

Vec Exp

*/

struct VecExpParams {
    CUDANet::DType dtype;
    const size_t   size;
    std::string    vector_path;
    std::string    expected_path;
};

class VecExpTest : public ::testing::TestWithParam<VecExpParams> {};

template <typename T>
void run_vec_exp_test(const VecExpParams params) {
    auto vector_data   = load_binary<T>(params.vector_path);
    auto expected_data = load_binary<T>(params.expected_path);

    auto backend = create_backend();

    auto vector_shape = CUDANet::Shape{params.size};

    auto vector = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), vector_data
    );
    auto expected = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), expected_data
    );
    auto output =
        create_output_tensor<T>(vector_shape, params.dtype, backend.get());

    auto grid_size = calc_grid_size(params.size);

    CUDANet::Kernels::vec_exp<<<grid_size, BLOCK_SIZE>>>(
        static_cast<const T*>(vector.device_ptr()),
        static_cast<T*>(output.device_ptr()),
        params.size
    );

    verify_output<T>(output, expected);
}

template void run_vec_exp_test<float>(const VecExpParams params);

TEST_P(VecExpTest, VectorExponentiation) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_vec_exp_test<float>(param);
    }
}

std::vector<VecExpParams> initialize_vec_exp_params() {
    std::vector<std::vector<std::string>> rows =
        load_csv(FIXTURE_PATH + "/matmul/vec_exp/metadata.csv");

    std::vector<VecExpParams> params;

    for (const auto& row : rows) {
        CUDANet::DType dtype = parse_dtype(row);

        size_t size = std::stoul(row[1]);

        params.push_back(
            VecExpParams{dtype, size, row[2], row[3]}
        );
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    VecExpTestCases,
    VecExpTest,
    testing::ValuesIn(initialize_vec_exp_params())
);

/*

Vec Sqrt

*/

struct VecSqrtParams {
    CUDANet::DType dtype;
    const size_t   size;
    std::string    vector_path;
    std::string    expected_path;
};

class VecSqrtTest : public ::testing::TestWithParam<VecSqrtParams> {};

template <typename T>
void run_vec_sqrt_test(const VecSqrtParams params) {
    auto vector_data   = load_binary<T>(params.vector_path);
    auto expected_data = load_binary<T>(params.expected_path);

    auto backend = create_backend();

    auto vector_shape = CUDANet::Shape{params.size};

    auto vector = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), vector_data
    );
    auto expected = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), expected_data
    );
    auto output =
        create_output_tensor<T>(vector_shape, params.dtype, backend.get());

    auto grid_size = calc_grid_size(params.size);

    CUDANet::Kernels::vec_sqrt<<<grid_size, BLOCK_SIZE>>>(
        static_cast<const T*>(vector.device_ptr()),
        static_cast<T*>(output.device_ptr()),
        params.size
    );

    verify_output<T>(output, expected);
}

template void run_vec_sqrt_test<float>(const VecSqrtParams params);

TEST_P(VecSqrtTest, VectorSquareRoot) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_vec_sqrt_test<float>(param);
    }
}

std::vector<VecSqrtParams> initialize_vec_sqrt_params() {
    std::vector<std::vector<std::string>> rows =
        load_csv(FIXTURE_PATH + "/matmul/vec_sqrt/metadata.csv");

    std::vector<VecSqrtParams> params;

    for (const auto& row : rows) {
        CUDANet::DType dtype = parse_dtype(row);

        size_t size = std::stoul(row[1]);

        params.push_back(
            VecSqrtParams{dtype, size, row[2], row[3]}
        );
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    VecSqrtTestCases,
    VecSqrtTest,
    testing::ValuesIn(initialize_vec_sqrt_params())
);