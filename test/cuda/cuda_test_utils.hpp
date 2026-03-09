#pragma once

#include "cudanet.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"

inline auto create_backend() {
    return CUDANet::BackendFactory::create(
        CUDANet::BackendType::CUDA_BACKEND, CUDANet::BackendConfig()
    );
}

inline size_t calc_grid_size(size_t size) {
    return (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
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

struct ConvolutionParams {
    CUDANet::DType dtype;
    size_t         inH, inW, inC;
    size_t         pH, pW;
    size_t         kH, kW;
    size_t         sH, sW;
    size_t         outH, outW, outC;
    std::string    input_path;
    std::string    kernel_path;
    std::string    bias_path;
    std::string    expected_path;
};

inline std::vector<ConvolutionParams> initialize_convolution_params(
    const std::string& csv_relative_path
) {
    std::vector<std::vector<std::string>> rows =
        load_csv(FIXTURE_PATH + csv_relative_path);
    std::vector<ConvolutionParams> params;
    for (const auto& row : rows) {
        params.push_back(
            {parse_dtype(row), std::stoul(row[1]), std::stoul(row[2]),
             std::stoul(row[3]), std::stoul(row[4]), std::stoul(row[5]),
             std::stoul(row[6]), std::stoul(row[7]), std::stoul(row[8]),
             std::stoul(row[9]), std::stoul(row[10]), std::stoul(row[11]),
             std::stoul(row[12]), row[13], row[14], row[15], row[16]}
        );
    }
    return params;
}
