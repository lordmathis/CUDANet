#pragma once

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#include <format>
#include <stdexcept>
#include <vector>

namespace CUDANet {

struct Shape {
    static constexpr size_t MAX_DIMS = 8;
    
    size_t dims[MAX_DIMS];
    size_t ndim;
    
    __host__ __device__ Shape() : ndim(0) {
        for (int i = 0; i < MAX_DIMS; i++) dims[i] = 0;
    }
    
    __host__ Shape(std::initializer_list<size_t> list) : ndim(list.size()) {
        if (ndim > MAX_DIMS) {
            throw std::runtime_error("Too many dimensions");
        }
        size_t i = 0;
        for (auto val : list) {
            dims[i++] = val;
        }
        for (; i < MAX_DIMS; i++) dims[i] = 0;
    }
    
    __host__ Shape(const std::vector<size_t>& vec) : ndim(vec.size()) {
        if (ndim > MAX_DIMS) {
            throw std::runtime_error("Too many dimensions");
        }
        for (size_t i = 0; i < ndim; i++) {
            dims[i] = vec[i];
        }
        for (size_t i = ndim; i < MAX_DIMS; i++) dims[i] = 0;
    }
    
    __host__ __device__ size_t operator[](size_t idx) const {
        return dims[idx];
    }
    
    __host__ __device__ size_t& operator[](size_t idx) {
        return dims[idx];
    }
    
    __host__ __device__ size_t size() const { return ndim; }
    
    __host__ bool operator==(const Shape& other) const {
        if (ndim != other.ndim) return false;
        for (size_t i = 0; i < ndim; i++) {
            if (dims[i] != other.dims[i]) return false;
        }
        return true;
    }
    
    __host__ bool operator!=(const Shape& other) const {
        return !(*this == other);
    }
};

std::string format_shape(const Shape& shape) {
    std::string result;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) result += ", ";
        result += std::to_string(shape[i]);
    }
    return result;
}

class InvalidShapeException : public std::runtime_error {
  public:
    InvalidShapeException(
        const std::string& param_name,
        size_t             expected,
        size_t             actual
    )
        : std::runtime_error(
              std::format(
                  "Invalid {} shape. Expected {}, actual {}",
                  param_name,
                  expected,
                  actual
              )
          ) {}

    InvalidShapeException(
        const std::string& message,
        const Shape&       shape_a,
        const Shape&       shape_b
    )
        : std::runtime_error(
              std::format(
                  "{}. Shape A: [{}], Shape B: [{}]",
                  message,
                  format_shape(shape_a),
                  format_shape(shape_b)
              )
          ) {}
};

}  // namespace CUDANet
