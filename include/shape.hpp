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
};

}  // namespace CUDANet
