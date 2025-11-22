#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "layer.hpp"
#include "module.hpp"

namespace CUDANet {

enum TensorType {
    WEIGHT,
    BIAS,
    RUNNING_MEAN,
    RUNNING_VAR
};

struct TensorInfo {
    std::string name;
    TensorType  type;
    int         size;
    int         offset;
};

class Model {
  public:
    Model(const CUDANet::Shape input_shape, const CUDANet::Shape output_shape);
    ~Model();

    virtual CUDANet::Tensor& predict(CUDANet::Tensor& input);

    CUDANet::Layer* get_layer(const std::string& name);

    void register_layer(const std::string& name, Layer* layer);

    void register_module(Module& module);

    void load_weights(const std::string& path);

    bool validate();

    void print_summary();

  protected:
    CUDANet::Shape in_shape;
    CUDANet::Shape out_shape;

    CUDANet::Tensor output;

    std::vector<std::pair<std::string, Layer*>> layers;
    std::unordered_map<std::string, Layer*>     layer_map;
};

}  // namespace CUDANet
