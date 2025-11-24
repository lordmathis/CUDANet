#pragma once

// CUDA Backend Implementation
#include "backend/cuda/cuda.cuh"

// CUDA Kernels
#include "backend/cuda/kernels/activation_functions.cuh"
#include "backend/cuda/kernels/convolution.cuh"
#include "backend/cuda/kernels/matmul.cuh"
#include "backend/cuda/kernels/pool.cuh"

