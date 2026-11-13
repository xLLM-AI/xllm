/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>
#include <string>

#include "models/dit/utils/minimax_h3_rainfusion_policy.h"

#if defined(USE_NPU)
#include "core/kernels/npu/tilelang/h3_rainfusion_runtime.h"
#endif

namespace xllm {

struct MiniMaxH3RainfusionExecutionContext {
#if defined(USE_NPU)
  const kernel::npu::tilelang::H3RainfusionGeometry* geometry = nullptr;
  kernel::npu::tilelang::H3RainfusionWorkspace* workspace = nullptr;
  torch::Tensor* output = nullptr;
  kernel::npu::tilelang::H3RainfusionLaunchCounters* counters = nullptr;
#endif
  H3RainfusionPolicy policy = kH3SparseNone;
  int64_t launches = 0;
};

inline bool h3_rainfusion_context_enabled(
    const MiniMaxH3RainfusionExecutionContext* context) {
#if defined(USE_NPU)
  return context != nullptr && context->geometry != nullptr &&
         context->workspace != nullptr && context->output != nullptr &&
         context->counters != nullptr;
#else
  return false;
#endif
}

struct H3RainfusionRequestState {
  H3RainfusionPolicy policy = kH3SparseNone;
  bool ready = false;
  std::string fallback_reason;
#if defined(USE_NPU)
  kernel::npu::tilelang::H3RainfusionLaunchCounters kernel_launches;
#endif
  MiniMaxH3RainfusionExecutionContext borrowed_kernel_context;
};

inline MiniMaxH3RainfusionExecutionContext* active_h3_rainfusion_context(
    H3RainfusionRequestState& state,
    int64_t step) {
  return state.ready && state.policy.includes_step(step)
             ? &state.borrowed_kernel_context
             : nullptr;
}

}  // namespace xllm
