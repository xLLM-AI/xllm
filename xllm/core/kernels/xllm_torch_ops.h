/* Copyright 2025-2026 The xLLM Authors. */
#pragma once

#if defined(USE_CUDA) || defined(USE_MUSA)
#include "core/kernels/cuda/cuda_ops_library.h"
#elif defined(USE_NPU)
#include "core/kernels/npu/npu_ops_library.h"
#endif

namespace xllm {

inline void ensure_xllm_torch_ops_registered() {
#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_NPU)
  ensure_xllm_ops_registered();
#endif
}

}  // namespace xllm
