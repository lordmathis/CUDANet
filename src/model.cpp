#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <unordered_map>
#include <vector>

#include "layer.hpp"
#include "layers/batch_norm.hpp"

#include "model.hpp"

using namespace CUDANet;

Model::Model(
    const CUDANet::Shape input_shape,
    const CUDANet::Shape output_shape
)
    : in_shape(input_shape),
      out_shape(out_shape),
      layers(std::vector<std::pair<std::string, Layer*>>()),
      layer_map(std::unordered_map<std::string, Layer*>()) {};

Model::~Model() {};

CUDANet::Tensor& Model::predict(CUDANet::Tensor& input) {
    CUDANet::Tensor* current = &input;
    for (const auto& [name, layer_ptr] : layers) {
        current = &(layer_ptr->forward(*current));
    }
    return *current;
}

void Model::register_layer(const std::string& name, Layer* layer) {
    layers.push_back({name, layer});
    layer_map[name] = layer;
}

void Model::register_module(Module& module) {
    for (const auto& [name, layer_ptr] : module.get_layers()) {
        layer_map[name] = layer_ptr;
        layers.push_back({name, layer_ptr});
    }

    return;
}

Layer* Model::get_layer(const std::string& name) {
    return layer_map[name];
}

void Model::load_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return;
    }

    u_short version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (version != 1) {
        std::cerr << "Unsupported model version: " << version << std::endl;
        return;
    }

    auto get_tensor_type = [](const std::string& type_str) {
        if (type_str == "weight") return TensorType::WEIGHT;
        if (type_str == "bias") return TensorType::BIAS;
        if (type_str == "running_mean") return TensorType::RUNNING_MEAN;
        if (type_str == "running_var") return TensorType::RUNNING_VAR;
        throw std::runtime_error("Unknown tensor type: " + type_str);
    };

    u_int64_t header_size;
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));

    std::string header(header_size, '\0');
    file.read(&header[0], header_size);

    std::vector<TensorInfo> tensor_infos;
    size_t                  pos = 0;

    while (pos < header.size()) {
        size_t next_pos = header.find('\n', pos);
        if (next_pos == std::string::npos) break;

        std::string line = header.substr(pos, next_pos - pos);
        pos              = next_pos + 1;

        size_t comma_pos = line.find(',');
        if (comma_pos == std::string::npos) continue;

        // Parse tensor name into name and type
        std::string name_str = line.substr(0, comma_pos);
        size_t      dot_pos  = name_str.find_last_of('.');
        if (dot_pos == std::string::npos) continue;
        std::string name = name_str.substr(0, dot_pos);

        TensorType  type = get_tensor_type(name_str.substr(dot_pos + 1));

        line = line.substr(comma_pos + 1);

        comma_pos = line.find(',');
        if (comma_pos == std::string::npos) continue;

        int size   = std::stoi(line.substr(0, comma_pos));
        int offset = std::stoi(line.substr(comma_pos + 1));

        tensor_infos.push_back({name, type, size, offset});
    }

    for (const auto& tensor_info : tensor_infos) {
        std::vector<float> values(tensor_info.size);

        file.seekg(
            sizeof(version) + sizeof(header_size) + header.size() +
            tensor_info.offset
        );
        file.read(
            reinterpret_cast<char*>(values.data()),
            tensor_info.size * sizeof(float)
        );

        if (layer_map.find(tensor_info.name) != layer_map.end()) {

            Layer* layer = layer_map[tensor_info.name];

            if (tensor_info.type == TensorType::WEIGHT) {
                if (layer->get_weights_size() != values.size()) {
                    std::cerr << "Layer: " << tensor_info.name
                              << " has incorrect number of weights, expected "
                              << layer->get_weights_size() << " but got "
                              << values.size() << ", skipping" << std::endl;
                    continue;
                }

                layer->set_weights(values.data());
            } else if (tensor_info.type == TensorType::BIAS) {
                if (layer->get_biases_size() != values.size()) {
                    std::cerr << "Layer: " << tensor_info.name
                              << " has incorrect number of biases, expected "
                              << layer->get_biases_size() << " but got "
                              << values.size() << ", skipping" << std::endl;
                    continue;
                }

                layer->set_biases(values.data());
            }

            Layers::BatchNorm2d* bn_layer = dynamic_cast<Layers::BatchNorm2d*>(layer);
            if (bn_layer == nullptr) {
                continue;
            }

            if (tensor_info.type == TensorType::RUNNING_MEAN) {
                if (bn_layer->get_running_mean_size() != values.size()) {
                    std::cerr << "Layer: " << tensor_info.name << " has incorrect number of running mean values, expected "
                                << bn_layer->get_running_mean_size() << " but got " << values.size() << ", skipping" << std::endl;
                    continue;
                }
                bn_layer->set_running_mean(values.data());
            } else if (tensor_info.type == TensorType::RUNNING_VAR) {
                if (bn_layer->get_running_var_size() != values.size()) {
                    std::cerr << "Layer: " << tensor_info.name << " has incorrect number of running var values, expected "
                                << bn_layer->get_running_var_size() << " but got " << values.size() << ", skipping" << std::endl;
                    continue;
                }
                bn_layer->set_running_var(values.data());
            }


        } else {
            std::cerr << "Layer: " << tensor_info.name
                      << " does not exist, skipping" << std::endl;
        }
    }

    file.close();
}

bool Model::validate() {
    bool valid = true;
    CUDANet::Shape shape = in_shape;

    for (const auto& [name, layer_ptr] : layers) {
        if (layer_ptr->input_shape() != shape) {
            valid = false;
            std::cerr << "Layer: " << name
                      << " has incorrect input shape, expected " << format_shape(shape)
                      << " but got " << format_shape(layer_ptr->input_shape())
                      << std::endl;
            break;
        }

        shape = layer_ptr->output_shape();
    }

    return valid;
}

void Model::print_summary() {
    struct layer_info {
        std::string name;
        std::string input_shape;
        std::string output_shape;
    };

    std::vector<layer_info> layer_infos;

    int max_name_length   = 0;
    int max_input_length  = 0;
    int max_output_length = 0;

    for (const auto& [name, layer_ptr] : layers) {
        layer_info li = {
            name, format_shape(layer_ptr->input_shape()),
            format_shape(layer_ptr->output_shape())
        };
        layer_infos.push_back(li);

        max_name_length = std::max(max_name_length, (int)li.name.size());
        max_input_length =
            std::max(max_input_length, (int)li.input_shape.size());
        max_output_length =
            std::max(max_output_length, (int)li.output_shape.size());
    }

    int row_length = max_name_length + max_input_length + max_output_length + 6;

    std::cout << "Model Summary:" << std::endl              
              << std::string(row_length, '-') << std::endl;

    for (const auto& li : layer_infos) {
        std::cout << std::left
                  << std::setw(max_name_length) << li.name
                  << " | " << std::right
                  << std::setw(max_input_length) << li.input_shape
                  << " | "
                  << std::setw(max_output_length) << li.output_shape
                  << std::endl;
    }
}