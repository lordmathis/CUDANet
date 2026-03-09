#pragma once

#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "cudanet.hpp"
#include "gtest/gtest.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif  // BLOCK_SIZE

const std::string FIXTURE_PATH = "fixtures";

inline std::vector<std::vector<std::string>> load_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open metadata: " + path);

    std::vector<std::vector<std::string>> rows;
    std::string                           line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream       ss(line);
        std::vector<std::string> fields;
        std::string              field;
        while (std::getline(ss, field, ',')) fields.push_back(field);
        rows.push_back(std::move(fields));
    }
    return rows;
}

template <typename T>
std::vector<T> load_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Cannot open fixture: " + path);

    auto byte_count = static_cast<std::size_t>(file.tellg());
    if (byte_count % sizeof(T) != 0)
        throw std::runtime_error(
            "File size not a multiple of element size: " + path
        );

    file.seekg(0);
    std::vector<T> data(byte_count / sizeof(T));
    file.read(
        reinterpret_cast<char*>(data.data()),
        static_cast<std::streamsize>(byte_count)
    );
    return data;
}

inline CUDANet::DType parse_dtype(const std::vector<std::string>& row) {
    if (row[0] == "float32") return CUDANet::DType::FLOAT32;
    throw std::runtime_error("Unknown dtype: " + row[0]);
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

template <typename T>
inline void assert_elements_near(
    const std::vector<T>& actual,
    const std::vector<T>& expected
) {
    for (size_t i = 0; i < actual.size(); ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            EXPECT_NEAR(actual[i], expected[i], 1e-4f);
        } else {
            EXPECT_EQ(actual[i], expected[i]);
        }
    }
}

template <typename T>
inline void verify_output(CUDANet::Tensor& output, CUDANet::Tensor& expected) {
#ifdef USE_CUDA
    cudaDeviceSynchronize();
#endif
    auto h_output   = output.to_host<T>();
    auto h_expected = expected.to_host<T>();
    ASSERT_EQ(h_output.size(), h_expected.size());
    assert_elements_near(h_output, h_expected);
}
