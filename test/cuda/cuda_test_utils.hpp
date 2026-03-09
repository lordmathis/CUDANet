#pragma once

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

struct UnaryOpParams {
    CUDANet::DType dtype;
    size_t         size;
    std::string    vector_path;
    std::string    expected_path;
};

template <typename ParamType>
inline std::vector<ParamType> initialize_unary_params(
    const std::string& csv_relative_path
) {
    std::vector<std::vector<std::string>> rows =
        load_csv(FIXTURE_PATH + csv_relative_path);
    std::vector<ParamType> params;
    for (const auto& row : rows) {
        params.push_back(
            {parse_dtype(row), std::stoul(row[1]), row[2], row[3]}
        );
    }
    return params;
}

struct BinaryOpParams {
    CUDANet::DType dtype;
    size_t         size;
    std::string    path_a;
    std::string    path_b;
    std::string    expected_path;
};

inline std::vector<BinaryOpParams> initialize_binary_params(
    const std::string& csv_relative_path
) {
    std::vector<std::vector<std::string>> rows =
        load_csv(FIXTURE_PATH + csv_relative_path);
    std::vector<BinaryOpParams> params;
    for (const auto& row : rows) {
        params.push_back(
            {parse_dtype(row), std::stoul(row[1]), row[2], row[3], row[4]}
        );
    }
    return params;
}

struct MatVecMulParams {
    CUDANet::DType dtype;
    size_t         rows;
    size_t         cols;
    std::string    matrix_path;
    std::string    vector_path;
    std::string    expected_path;
};

inline std::vector<MatVecMulParams> initialize_mat_vec_mul_params(
    const std::string& csv_relative_path
) {
    std::vector<std::vector<std::string>> rows =
        load_csv(FIXTURE_PATH + csv_relative_path);
    std::vector<MatVecMulParams> params;
    for (const auto& row : rows) {
        params.push_back(
            {parse_dtype(row), std::stoul(row[1]), std::stoul(row[2]), row[3],
             row[4], row[5]}
        );
    }
    return params;
}

struct VecScaleParams {
    CUDANet::DType dtype;
    size_t         size;
    std::string    vector_path;
    std::string    scale_path;
    std::string    epsilon_path;
    std::string    expected_path;
};

inline std::vector<VecScaleParams> initialize_vec_scale_params(
    const std::string& csv_relative_path
) {
    std::vector<std::vector<std::string>> rows =
        load_csv(FIXTURE_PATH + csv_relative_path);
    std::vector<VecScaleParams> params;
    for (const auto& row : rows) {
        params.push_back(
            {parse_dtype(row), std::stoul(row[1]), row[2], row[3], row[4],
             row[5]}
        );
    }
    return params;
}

struct PoolParams {
    CUDANet::DType dtype;
    size_t         inH, inW, inC;
    size_t         outH, outW;
    size_t         kH, kW;
    size_t         sH, sW;
    size_t         pH, pW;
    std::string    input_path;
    std::string    expected_path;
};

inline std::vector<PoolParams> initialize_pool_params(
    const std::string& csv_relative_path
) {
    std::vector<std::vector<std::string>> rows =
        load_csv(FIXTURE_PATH + csv_relative_path);
    std::vector<PoolParams> params;
    for (const auto& row : rows) {
        params.push_back(
            {parse_dtype(row), std::stoul(row[1]), std::stoul(row[2]),
             std::stoul(row[3]), std::stoul(row[4]), std::stoul(row[5]),
             std::stoul(row[6]), std::stoul(row[7]), std::stoul(row[8]),
             std::stoul(row[9]), std::stoul(row[10]), std::stoul(row[11]),
             row[12], row[13]}
        );
    }
    return params;
}
