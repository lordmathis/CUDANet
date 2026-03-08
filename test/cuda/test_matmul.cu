#include "cudanet.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"
#include "cuda_test_utils.hpp"

/*

Mat Vec Mul

*/

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

INSTANTIATE_TEST_SUITE_P(
    MatVecMulTestCases,
    MatVecMulTest,
    testing::ValuesIn(initialize_mat_vec_mul_params("/matmul/mat_vec_mul/metadata.csv"))
);

/*

Vec Vec Add

*/

class VecVecAddTest : public ::testing::TestWithParam<BinaryOpParams> {};

template <typename T>
void run_vec_vec_add_test(const BinaryOpParams params) {
    auto vector_a_data = load_binary<T>(params.path_a);
    auto vector_b_data = load_binary<T>(params.path_b);
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

template void run_vec_vec_add_test<float>(const BinaryOpParams params);

TEST_P(VecVecAddTest, VectorVectorAddition) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_vec_vec_add_test<float>(param);
    }
}

INSTANTIATE_TEST_SUITE_P(
    VecVecAddTestCases,
    VecVecAddTest,
    testing::ValuesIn(initialize_binary_params("/matmul/vec_vec_add/metadata.csv"))
);

/*

Vec Vec Sub

*/

class VecVecSubTest : public ::testing::TestWithParam<BinaryOpParams> {};

template <typename T>
void run_vec_vec_sub_test(const BinaryOpParams params) {
    auto vector_a_data = load_binary<T>(params.path_a);
    auto vector_b_data = load_binary<T>(params.path_b);
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

template void run_vec_vec_sub_test<float>(const BinaryOpParams params);

TEST_P(VecVecSubTest, VectorVectorSubtraction) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_vec_vec_sub_test<float>(param);
    }
}

INSTANTIATE_TEST_SUITE_P(
    VecVecSubTestCases,
    VecVecSubTest,
    testing::ValuesIn(initialize_binary_params("/matmul/vec_vec_sub/metadata.csv"))
);

/*

Vec Vec Mul

*/

class VecVecMulTest : public ::testing::TestWithParam<BinaryOpParams> {};

template <typename T>
void run_vec_vec_mul_test(const BinaryOpParams params) {
    auto vector_a_data = load_binary<T>(params.path_a);
    auto vector_b_data = load_binary<T>(params.path_b);
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

template void run_vec_vec_mul_test<float>(const BinaryOpParams params);

TEST_P(VecVecMulTest, VectorVectorMultiplication) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_vec_vec_mul_test<float>(param);
    }
}

INSTANTIATE_TEST_SUITE_P(
    VecVecMulTestCases,
    VecVecMulTest,
    testing::ValuesIn(initialize_binary_params("/matmul/vec_vec_mul/metadata.csv"))
);

/*

Vec Scalar Sub

*/

class VecScalarSubTest : public ::testing::TestWithParam<BinaryOpParams> {};

template <typename T>
void run_vec_scalar_sub_test(const BinaryOpParams params) {
    auto vector_data   = load_binary<T>(params.path_a);
    auto scalar_data   = load_binary<T>(params.path_b);
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

template void run_vec_scalar_sub_test<float>(const BinaryOpParams params);

TEST_P(VecScalarSubTest, VectorScalarSubtraction) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_vec_scalar_sub_test<float>(param);
    }
}

INSTANTIATE_TEST_SUITE_P(
    VecScalarSubTestCases,
    VecScalarSubTest,
    testing::ValuesIn(initialize_binary_params("/matmul/vec_scalar_sub/metadata.csv"))
);

/*

Vec Scalar Add

*/

class VecScalarAddTest : public ::testing::TestWithParam<BinaryOpParams> {};

template <typename T>
void run_vec_scalar_add_test(const BinaryOpParams params) {
    auto vector_data   = load_binary<T>(params.path_a);
    auto scalar_data   = load_binary<T>(params.path_b);
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

template void run_vec_scalar_add_test<float>(const BinaryOpParams params);

TEST_P(VecScalarAddTest, VectorScalarAddition) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_vec_scalar_add_test<float>(param);
    }
}

INSTANTIATE_TEST_SUITE_P(
    VecScalarAddTestCases,
    VecScalarAddTest,
    testing::ValuesIn(initialize_binary_params("/matmul/vec_scalar_add/metadata.csv"))
);

/*

Vec Scalar Div

*/

class VecScalarDivTest : public ::testing::TestWithParam<BinaryOpParams> {};

template <typename T>
void run_vec_scalar_div_test(const BinaryOpParams params) {
    auto vector_data   = load_binary<T>(params.path_a);
    auto scalar_data   = load_binary<T>(params.path_b);
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

template void run_vec_scalar_div_test<float>(const BinaryOpParams params);

TEST_P(VecScalarDivTest, VectorScalarDivision) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_vec_scalar_div_test<float>(param);
    }
}

INSTANTIATE_TEST_SUITE_P(
    VecScalarDivTestCases,
    VecScalarDivTest,
    testing::ValuesIn(initialize_binary_params("/matmul/vec_scalar_div/metadata.csv"))
);

/*

Vec Scalar Mul

*/

class VecScalarMulTest : public ::testing::TestWithParam<BinaryOpParams> {};

template <typename T>
void run_vec_scalar_mul_test(const BinaryOpParams params) {
    auto vector_data   = load_binary<T>(params.path_a);
    auto scalar_data   = load_binary<T>(params.path_b);
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

template void run_vec_scalar_mul_test<float>(const BinaryOpParams params);

TEST_P(VecScalarMulTest, VectorScalarMultiplication) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_vec_scalar_mul_test<float>(param);
    }
}

INSTANTIATE_TEST_SUITE_P(
    VecScalarMulTestCases,
    VecScalarMulTest,
    testing::ValuesIn(initialize_binary_params("/matmul/vec_scalar_mul/metadata.csv"))
);

/*

Vec Exp

*/

class VecExpTest : public ::testing::TestWithParam<UnaryOpParams> {};

template <typename T>
void run_vec_exp_test(const UnaryOpParams params) {
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
        static_cast<T*>(output.device_ptr()), params.size
    );

    verify_output<T>(output, expected);
}

template void run_vec_exp_test<float>(const UnaryOpParams params);

TEST_P(VecExpTest, VectorExponentiation) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_vec_exp_test<float>(param);
    }
}

INSTANTIATE_TEST_SUITE_P(
    VecExpTestCases,
    VecExpTest,
    testing::ValuesIn(initialize_unary_params<UnaryOpParams>("/matmul/vec_exp/metadata.csv"))
);

/*

Vec Sqrt

*/

class VecSqrtTest : public ::testing::TestWithParam<UnaryOpParams> {};

template <typename T>
void run_vec_sqrt_test(const UnaryOpParams params) {
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
        static_cast<T*>(output.device_ptr()), params.size
    );

    verify_output<T>(output, expected);
}

template void run_vec_sqrt_test<float>(const UnaryOpParams params);

TEST_P(VecSqrtTest, VectorSquareRoot) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_vec_sqrt_test<float>(param);
    }
}

INSTANTIATE_TEST_SUITE_P(
    VecSqrtTestCases,
    VecSqrtTest,
    testing::ValuesIn(initialize_unary_params<UnaryOpParams>("/matmul/vec_sqrt/metadata.csv"))
);

/*

Vec Scale

*/

class VecScaleTest : public ::testing::TestWithParam<VecScaleParams> {};

template <typename T>
void run_vec_scale_test(const VecScaleParams params) {
    auto vector_data   = load_binary<T>(params.vector_path);
    auto scale_data    = load_binary<T>(params.scale_path);
    auto epsilon_data  = load_binary<T>(params.epsilon_path);
    auto expected_data = load_binary<T>(params.expected_path);

    auto backend = create_backend();

    auto vector_shape = CUDANet::Shape{params.size};

    auto vector = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), vector_data
    );
    auto scale = create_tensor<T>(
        CUDANet::Shape{1}, params.dtype, backend.get(), scale_data
    );
    auto epsilon = create_tensor<T>(
        CUDANet::Shape{1}, params.dtype, backend.get(), epsilon_data
    );
    auto expected = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), expected_data
    );
    auto output =
        create_output_tensor<T>(vector_shape, params.dtype, backend.get());

    auto grid_size = calc_grid_size(params.size);

    CUDANet::Kernels::vec_scale<<<grid_size, BLOCK_SIZE>>>(
        static_cast<const T*>(vector.device_ptr()),
        static_cast<T*>(output.device_ptr()),
        static_cast<const T*>(scale.device_ptr()),
        static_cast<const T*>(epsilon.device_ptr()), params.size
    );

    verify_output<T>(output, expected);
}

template void run_vec_scale_test<float>(const VecScaleParams params);

TEST_P(VecScaleTest, VectorScaling) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_vec_scale_test<float>(param);
    }
}

INSTANTIATE_TEST_SUITE_P(
    VecScaleTestCases,
    VecScaleTest,
    testing::ValuesIn(initialize_vec_scale_params("/matmul/vec_scale/metadata.csv"))
);

/*

Max Reduce

*/

class MaxReduceTest : public ::testing::TestWithParam<UnaryOpParams> {};

template <typename T>
void run_max_reduce_test(const UnaryOpParams params) {
    auto vector_data   = load_binary<T>(params.vector_path);
    auto expected_data = load_binary<T>(params.expected_path);

    auto backend = create_backend();

    auto vector_shape = CUDANet::Shape{params.size};
    auto vector       = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), vector_data
    );

    size_t num_blocks      = calc_grid_size(params.size);
    auto   expected_shape = CUDANet::Shape{num_blocks};
    auto   expected       = create_tensor<T>(
        expected_shape, params.dtype, backend.get(), expected_data
    );
    auto output =
        create_output_tensor<T>(expected_shape, params.dtype, backend.get());

    CUDANet::Kernels::max_reduce<<<num_blocks, BLOCK_SIZE>>>(
        static_cast<const T*>(vector.device_ptr()),
        static_cast<T*>(output.device_ptr()), params.size
    );

    verify_output<T>(output, expected);
}

template void run_max_reduce_test<float>(const UnaryOpParams params);

TEST_P(MaxReduceTest, MaximumReduction) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_max_reduce_test<float>(param);
    }
}

INSTANTIATE_TEST_SUITE_P(
    MaxReduceTestCases,
    MaxReduceTest,
    testing::ValuesIn(initialize_unary_params<UnaryOpParams>("/matmul/max_reduce/metadata.csv"))
);

/*

Sum Reduce

*/

class SumReduceTest : public ::testing::TestWithParam<UnaryOpParams> {};

template <typename T>
void run_sum_reduce_test(const UnaryOpParams params) {
    auto vector_data   = load_binary<T>(params.vector_path);
    auto expected_data = load_binary<T>(params.expected_path);

    auto backend = create_backend();

    auto vector_shape = CUDANet::Shape{params.size};
    auto vector       = create_tensor<T>(
        vector_shape, params.dtype, backend.get(), vector_data
    );

    size_t num_blocks      = calc_grid_size(params.size);
    auto   expected_shape = CUDANet::Shape{num_blocks};
    auto   expected       = create_tensor<T>(
        expected_shape, params.dtype, backend.get(), expected_data
    );
    auto output =
        create_output_tensor<T>(expected_shape, params.dtype, backend.get());

    CUDANet::Kernels::sum_reduce<<<num_blocks, BLOCK_SIZE>>>(
        static_cast<const T*>(vector.device_ptr()),
        static_cast<T*>(output.device_ptr()), params.size
    );

    verify_output<T>(output, expected);
}

template void run_sum_reduce_test<float>(const UnaryOpParams params);

TEST_P(SumReduceTest, SummationReduction) {
    auto param = GetParam();
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_sum_reduce_test<float>(param);
    }
}

INSTANTIATE_TEST_SUITE_P(
    SumReduceTestCases,
    SumReduceTest,
    testing::ValuesIn(initialize_unary_params<UnaryOpParams>("/matmul/sum_reduce/metadata.csv"))
);
