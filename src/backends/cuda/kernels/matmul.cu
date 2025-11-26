#include "backend/cuda/cuda.cuh"
#include "backend/cuda/kernels/matmul.cuh"

using namespace CUDANet;

template __global__ void Kernels::mat_vec_mul<float>(
    const float* __restrict__ d_matrix,
    const float* __restrict__ d_vector,
    float* __restrict__ d_output,
    const unsigned int w,
    const unsigned int h
);

template <typename T>
__global__ void Kernels::mat_vec_mul(
    const T* __restrict__ d_matrix,
    const T* __restrict__ d_vector,
    T* __restrict__ d_output,
    const unsigned int w,
    const unsigned int h
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < h) {
        T temp = static_cast<T>(0);

        for (unsigned int j = 0; j < w; j++) {
            temp += d_matrix[tid * w + j] * d_vector[j];
        }

        d_output[tid] = temp;
    }
}

template __global__ void Kernels::vec_vec_add<float>(
    const float* __restrict__ d_vector1,
    const float* __restrict__ d_vector2,
    float* __restrict__ d_output,
    const unsigned int w
);

template <typename T>
__global__ void Kernels::vec_vec_add(
    const T* __restrict__ d_vector1,
    const T* __restrict__ d_vector2,
    T* __restrict__ d_output,
    const unsigned int w
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= w) {
        return;
    }
    d_output[tid] = d_vector1[tid] + d_vector2[tid];
}

template __global__ void Kernels::vec_vec_sub<float>(
    const float* __restrict__ d_vector1,
    const float* __restrict__ d_vector2,
    float* __restrict__ d_output,
    const unsigned int w
);

template <typename T>
__global__ void Kernels::vec_vec_sub(
    const T* __restrict__ d_vector1,
    const T* __restrict__ d_vector2,
    T* __restrict__ d_output,
    const unsigned int w
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= w) {
        return;
    }
    d_output[tid] = d_vector1[tid] - d_vector2[tid];
}

template __global__ void Kernels::vec_vec_mul<float>(
    const float* __restrict__ d_vector1,
    const float* __restrict__ d_vector2,
    float* __restrict__ d_output,
    const unsigned int w
);

template <typename T>
__global__ void Kernels::vec_vec_mul(
    const T* __restrict__ d_vector1,
    const T* __restrict__ d_vector2,
    T* __restrict__ d_output,
    const unsigned int w
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= w) {
        return;
    }
    d_output[tid] = d_vector1[tid] * d_vector2[tid];
}

template __global__ void Kernels::vec_scalar_sub<float>(
    const float* __restrict__ d_src,
    float* __restrict__ d_out,
    const float* __restrict__ d_scalar,
    const unsigned int len
);

template <typename T>
__global__ void Kernels::vec_scalar_sub(
    const T* __restrict__ d_src,
    T* __restrict__ d_out,
    const T* __restrict__ d_scalar,
    const unsigned int len
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= len) {
        return;
    }
    d_out[tid] = d_src[tid] - *d_scalar;
}

template __global__ void Kernels::vec_scalar_add<float>(
    const float* __restrict__ d_src,
    float* __restrict__ d_out,
    const float* __restrict__ d_scalar,
    const unsigned int len
);

template <typename T>
__global__ void Kernels::vec_scalar_add(
    const T* __restrict__ d_src,
    T* __restrict__ d_out,
    const T* __restrict__ d_scalar,
    const unsigned int len
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= len) {
        return;
    }
    d_out[tid] = d_src[tid] + *d_scalar;
}

template __global__ void Kernels::vec_scalar_div<float>(
    const float* __restrict__ d_src,
    float* __restrict__ d_out,
    const float* __restrict__ d_scalar,
    const unsigned int len
);

template <typename T>
__global__ void Kernels::vec_scalar_div(
    const T* __restrict__ d_src,
    T* __restrict__ d_out,
    const T* __restrict__ d_scalar,
    const unsigned int len
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= len) {
        return;
    }
    d_out[tid] = d_src[tid] / *d_scalar;
}

template __global__ void Kernels::vec_scalar_mul<float>(
    const float* __restrict__ d_src,
    float* __restrict__ d_out,
    const float* __restrict__ d_scalar,
    const unsigned int len
);

template <typename T>
__global__ void Kernels::vec_scalar_mul(
    const T* __restrict__ d_src,
    T* __restrict__ d_out,
    const T* __restrict__ d_scalar,
    const unsigned int len
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= len) {
        return;
    }
    d_out[tid] = d_src[tid] * *d_scalar;
}

template __global__ void Kernels::vec_exp<float>(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const unsigned int len
);

template <typename T>
__global__ void Kernels::vec_exp(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const unsigned int len
) {
    int stride = gridDim.x * blockDim.x;
    int tid    = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = tid; i < len; i += stride) {
        // TODO: separate implementation for __half
        dst[i] = expf(src[i]);
    }
}

template __global__ void Kernels::vec_sqrt<float>(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const unsigned int len
);

template <typename T>
__global__ void Kernels::vec_sqrt(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const unsigned int len
) {
    int stride = gridDim.x * blockDim.x;
    int tid    = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = tid; i < len; i += stride) {
        // TODO: separate implementation for __half
        dst[i] = sqrtf(src[i]);
    }
}

template __global__ void Kernels::vec_scale<float>(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const float* __restrict__ scale,
    const float* epsilon,
    const unsigned int len
);

template <typename T>
__global__ void Kernels::vec_scale(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const T* __restrict__ scale,
    const T* epsilon,
    const unsigned int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        // TODO: separate implementation for __half
        float inv_std = rsqrtf(*scale + *epsilon);
        dst[idx] = src[idx] * inv_std;
    }
}

template __global__ void Kernels::max_reduce<float>(
    const float* __restrict__ d_vector,
    float* __restrict__ d_output,
    const unsigned int len
);

template <typename T>
__global__ void Kernels::max_reduce(
    const T* __restrict__ d_vector,
    T* __restrict__ d_output,
    const unsigned int len
) {
    __shared__ T shared_max[BLOCK_SIZE];
    int i       = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < len) {
        shared_max[threadIdx.x] = d_vector[i];
    } else {
        shared_max[threadIdx.x] = -INFINITY;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            // TODO: separate implementation for __half
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_output[blockIdx.x] = shared_max[0];
    }
}

template __global__ void Kernels::sum_reduce<float>(
    const float* __restrict__ d_vector,
    float* __restrict__ d_output,
    const unsigned int len
);

template <typename T>
__global__ void Kernels::sum_reduce(
    const T* __restrict__ d_vector,
    T* __restrict__ d_output,
    const unsigned int len
) {
    __shared__ T partial_sum[BLOCK_SIZE];
    int              i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < len) {
        partial_sum[threadIdx.x] = d_vector[i];
    } else {
        partial_sum[threadIdx.x] = static_cast<T>(0);
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_output[blockIdx.x] = partial_sum[0];
    }
}
