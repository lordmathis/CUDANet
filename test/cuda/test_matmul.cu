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
    
//     CUDANet::Kernels::mat_vec_mul(
//     const T* __restrict__ d_matrix,
//     const T* __restrict__ d_vector,
//     T* __restrict__ d_output,
//     const unsigned int w,
//     const unsigned int h
// ); 

}

// Instantiate with test cases  
INSTANTIATE_TEST_SUITE_P(  
    MatVecMulTestCases,  
    MatVecMulTest,  
    testing::Values(  
        MatMulParams{CUDANet::DType::FLOAT32, CUDANet::Shape{5,4}, CUDANet::Shape{4}}
    )
); 
