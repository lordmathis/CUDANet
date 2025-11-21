#include "add.hpp"

using namespace CUDANet::Layers;


Add::Add(CUDANet::Shape a_shape, CUDANet::Shape b_shape, CUDANet::Backend* backend) : backend(backend) {
    if (a_shape != b_shape) {
        throw InvalidShapeException(
            "Add requires matching dimensions", a_shape, b_shape
        );
    }

    out_shape = a_shape;
    output = CUDANet::Tensor(out_shape, CUDANet::DType::FLOAT32, backend);
}

Add::~Add() {}

CUDANet::Tensor&
Add::forward(CUDANet::Tensor& input_a, CUDANet::Tensor& input_b) {
    output.zero();
    backend->add(
        input_a,
        input_b,
        output
    );
    return output;
}
