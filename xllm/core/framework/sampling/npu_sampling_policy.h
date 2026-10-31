/* Copyright 2026 The xLLM Authors.

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

// This symbol exists only in an explicitly instrumented benchmark build.
// The offline controller changes it while all owned, idle workers are stopped.
#if defined(XLLM_NPU_SAMPLING_ABLATION)
extern "C" {
inline volatile uint8_t xllm_npu_sampling_ablation_mask = 63;
}
#endif

namespace xllm {

enum class NpuSamplingOptimization : uint8_t {
  BATCHED = 1,
  RANDOM_ROW_FILTER = 2,
  FUSED_LOGITS = 4,
  COMPACT_NORMALIZATION = 8,
  SPARSE_DFLASH2 = 16,
  VERIFICATION = 32,
};

// Production dispatch is automatic; only benchmark builds permit ablation.
#if defined(XLLM_NPU_SAMPLING_ABLATION)
inline bool npu_sampling_enabled(NpuSamplingOptimization optimization) {
  return (xllm_npu_sampling_ablation_mask &
          static_cast<uint8_t>(optimization)) != 0;
}
#else
constexpr bool npu_sampling_enabled(NpuSamplingOptimization /*optimization*/) {
  return true;
}
#endif

}  // namespace xllm
