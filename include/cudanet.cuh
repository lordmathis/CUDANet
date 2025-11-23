#pragma once

// ============================================================================
// Core Data Structures & Abstractions (BACKEND-INDEPENDENT)
// ============================================================================

#include "shape.hpp"
#include "backend.hpp"
#include "tensor.hpp"
#include "layer.hpp"

// ============================================================================
// Container Classes
// ============================================================================

#include "module.hpp"
#include "model.hpp"

// ============================================================================
// Layer Implementations
// ============================================================================

// Activation
#include "layers/activation.hpp"

// Normalization
#include "layers/batch_norm.hpp"

// Linear
#include "layers/dense.hpp"

// Convolutional
#include "layers/conv2d.hpp"

// Pooling
#include "layers/max_pool.hpp"
#include "layers/avg_pool.hpp"

// Composition (element-wise operations)
#include "layers/add.hpp"
#include "layers/concat.hpp"

// ============================================================================
// Utilities
// ============================================================================

#include "utils/imagenet.hpp"

// ============================================================================
// Backend-Specific Includes (conditionally compiled)
// ============================================================================

#ifdef USE_CUDA
#include "backend/cuda/cuda_backend.cuh"
#endif
