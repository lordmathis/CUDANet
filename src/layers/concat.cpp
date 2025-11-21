#include "concat.hpp"

using namespace CUDANet::Layers;

Concat::Concat(const CUDANet::Shape a_shape, const CUDANet::Shape b_shape, CUDANet::Backend *backend)
    : a_shape(a_shape), b_shape(b_shape), backend(backend) {
    if (a_shape[0] != b_shape[0] || a_shape[1] != b_shape[1]) {
        throw InvalidShapeException(
            "Concat requires matching batch and height dimensions", a_shape,
            b_shape
        );
    }

    out_shape = {a_shape[0], a_shape[1], a_shape[2] + b_shape[2]};
    output = CUDANet::Tensor(out_shape, CUDANet::DType::FLOAT32, backend);
}

Concat::~Concat() {}

CUDANet::Tensor& Concat::forward(CUDANet::Tensor& input_a, CUDANet::Tensor& input_b) {
    output.zero();
    backend->concat(
        input_a,
        input_b,
        output
    );
    return output;
}

CUDANet::Shape Concat::output_shape() {
    return out_shape;
}