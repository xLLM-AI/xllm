/* Copyright 2025-2026 The xLLM Authors. */
#pragma once

#if defined(USE_CUDA) || defined(USE_MUSA)
#include "core/kernels/cuda/cuda_ops_library.h"
#endif

namespace xllm {

inline void ensure_xllm_torch_ops_registered() {
#if defined(USE_CUDA) || defined(USE_MUSA)
  ensure_xllm_ops_registered();
#endif
}

}  // namespace xllm
