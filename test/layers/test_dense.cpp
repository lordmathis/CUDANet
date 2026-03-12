#include "cudanet.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"

#include <iostream>

struct DenseParams {
    CUDANet::DType dtype;
    size_t         input_size;
    size_t         output_size;
    std::string    weights_path;
    std::string    biases_path;
    std::string    input_path;
};


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
