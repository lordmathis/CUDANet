#pragma once

#include <vector>

#include "backend.hpp"
#include "layer.hpp"

namespace CUDANet::Layers {

/**
 * @brief Dense (fully connected) layer
 *
 */
class Dense : public Layer {
  public:

    Dense(CUDANet::Backend *backend, CUDANet::Shape input_shape, CUDANet::Shape output_shape);

    ~Dense();

    CUDANet::Tensor& forward(CUDANet::Tensor &input);

    CUDANet::Shape input_shape();

    CUDANet::Shape output_shape();

    size_t input_size();

    size_t output_size();

    void set_weights(CUDANet::Tensor &input);

    CUDANet::Tensor& get_weights();

    void set_biases(CUDANet::Tensor &input);

    CUDANet::Tensor& get_biases();

  private:
    CUDANet::Backend *backend;

    CUDANet::Shape in_shape;
    CUDANet::Shape out_shape;

    CUDANet::Tensor weights;
    CUDANet::Tensor biases;


    void init_weights();
    void init_biases();

// #ifdef USE_CUDA
//     float* d_output;

//     float* d_weights;
//     float* d_biases;

//     // Precompute kernel launch parameters
//     int forwardGridSize;
//     int biasGridSize;

//     /**
//      * @brief Copy the weights and biases to the device
//      *
//      */
//     void toCuda();

//     void initCUDA();
//     void delCUDA();

//     float* forwardCUDA(const float* d_input);
// #endif

};

}  // namespace CUDANet::Layers

