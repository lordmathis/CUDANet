#include "cudanet.hpp"
#include "gtest/gtest.h"

struct MatMulParams
{
    CUDANet::DType dtype;
    CUDANet::Shape matrix_shape;
    CUDANet::Shape vec_shape;
};

class MatVecMulTest : public ::testing::TestWithParam<MatMulParams> {};

TEST_P(MatVecMulTest, MatrixVectorMultiplication) {  
    auto param = GetParam(); // Get the current SumTestParams  

    // DType dispatch
    if (param.dtype == CUDANet::DType::FLOAT32) {
        run_matmul_test<float>(param.fixture_path, backend.get());
    }
    
//     CUDANet::Kernels::mat_vec_mul(
//     const T* __restrict__ d_matrix,
//     const T* __restrict__ d_vector,
//     T* __restrict__ d_output,
//     const unsigned int w,
//     const unsigned int h
// ); 

}

template<typename T>
void run_matmul_test(const std::string& path) {
    // Load binary data
    auto matrix_data = load_binary<T>(path + "/matrix.bin");
    auto vector_data = load_binary<T>(path + "/vector.bin");
    auto expected_data = load_binary<T>(path + "/expected.bin");
    
    // Run operation
    Tensor output = backend->matmul(matrix, vector);
    
    // Verify results
    auto actual = copy_to_host(output);
    assert_close(actual, expected_data);
}

// Instantiate with test cases  
INSTANTIATE_TEST_SUITE_P(  
    MatVecMulTestCases,  
    MatVecMulTest,  
    testing::Values(  
        MatMulParams{CUDANet::DType::FLOAT32, CUDANet::Shape{5,4}, CUDANet::Shape{4}}
    )
); 
