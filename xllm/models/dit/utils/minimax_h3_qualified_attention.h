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

#include <torch/torch.h>

#include <stdexcept>

#include "models/dit/transformers/minimax_h3_blocks.h"
#include "models/dit/utils/minimax_h3_rainfusion_context.h"

namespace xllm {

// Both serial and overlapped UAA converge here after all Q/K/V are ready.
// The inverse UAA consumes the same original packed order on either path.
inline torch::Tensor minimax_h3_qualified_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& cu_seqlens,
    MiniMaxH3RainfusionExecutionContext* rainfusion = nullptr) {
  if (rainfusion == nullptr) {
    return minimax_h3_segmented_sdpa(query, key, value, cu_seqlens);
  }
#if defined(USE_NPU)
  if (!h3_rainfusion_context_enabled(rainfusion)) {
    throw std::logic_error("H3 active RainFusion context is incomplete");
  }
  const auto& geometry = *rainfusion->geometry;
  const int64_t width = kernel::npu::tilelang::kH3RainfusionHeads *
                        kernel::npu::tilelang::kH3RainfusionHeadDim;
  kernel::npu::tilelang::h3_rainfusion_attention_out(
      query.view({geometry.total_rows, width}),
      key.view({geometry.total_rows, width}),
      value.view({geometry.total_rows, width}),
      geometry,
      *rainfusion->workspace,
      *rainfusion->output,
      rainfusion->counters);
  // Counts completed host submissions; request completion still synchronizes
  // the device and rejects asynchronous kernel failures.
  ++rainfusion->launches;
  return rainfusion->output->view(query.sizes());
#else
  throw std::logic_error("H3 RainFusion requires the NPU backend");
#endif
}

}  // namespace xllm
