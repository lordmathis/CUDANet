#pragma once

#include <vector>

namespace CUDANet {

typedef std::vector<size_t> Shape;

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

  private:
    static std::string format_shape(const Shape& shape) {
        std::string result;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) result += ", ";
            result += std::to_string(shape[i]);
        }
        return result;
    }
};

}  // namespace CUDANet
