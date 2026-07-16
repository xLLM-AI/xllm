/* Copyright 2025-2026 The xLLM Authors. */
#pragma once

#if defined(USE_CUDA)
#include "core/kernels/cuda/cuda_ops_library.h"
#endif

namespace xllm {

inline void ensure_xllm_torch_ops_registered() {
#if defined(USE_CUDA)
  ensure_xllm_ops_registered();
#endif
}

}  // namespace xllm
