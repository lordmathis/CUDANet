#include "module.hpp"

#include <algorithm>

using namespace CUDANet;

CUDANet::Shape Module::input_shape() {
    return in_shape;
}

CUDANet::Shape Module::output_shape() {
    return out_shape;
}

size_t Module::input_size() {
    size_t count = 1;
    for (const auto& dim : in_shape) {
        count *= dim;
    }
    return sizeof(float) * count;
}

size_t Module::output_size() {
    size_t count = 1;
    for (const auto& dim : out_shape) {
        count *= dim;
    }
    return sizeof(float) * count;
}

void Module::register_layer(const std::string& name, Layer& layer) {
    layers.push_back({name, layer});
}

void Module::register_module(Module& module) {
    for (const auto& moduleLayer : module.get_layers()) {
        layers.push_back({moduleLayer.first, moduleLayer.second});
    }
}

const std::vector<std::pair<std::string, Layer&>>&
Module::get_layers() const {
    return layers;
}
