#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "layer.hpp"

namespace CUDANet {

class Module {
  public:
    CUDANet::Shape input_shape();

    CUDANet::Shape output_shape();

    void register_layer(const std::string& name, Layer* layer);

    void register_module(Module& module);

    const std::vector<std::pair<std::string, Layer*>>& get_layers() const;

  protected:
    std::vector<std::pair<std::string, Layer*>> layers;

    CUDANet::Shape in_shape;
    CUDANet::Shape out_shape;
};

}  // namespace CUDANet
