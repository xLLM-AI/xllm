/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Thin adapter around the flash-mla native extension.

#pragma once

#include <torch/torch.h>

#include <cstdint>

namespace xllm {
namespace kernel {
namespace dcu {
namespace flash_mla {

enum class DenseDecodeKind {
  kQNopePe,
};

// Inputs for flash-mla dense MLA decode.
//
//   q_nope:      [B, S_q, H_q, kv_lora_rank]
//   q_pe:        [B, S_q, H_q, qk_rope_head_dim]
//   k_cache:     [num_blocks, 64, 1, kv_lora_rank + qk_rope_head_dim]
//   seqlens_k:   [B] int32
//   block_table: [B, max_blocks] int32
struct DenseDecodeParams {
  torch::Tensor q_nope;
  torch::Tensor q_pe;
  torch::Tensor k_cache;
  torch::Tensor seqlens_k;
  torch::Tensor block_table;
  int64_t head_size_v = 0;
  float softmax_scale = -1.0F;
  bool is_causal = false;
  DenseDecodeKind kind = DenseDecodeKind::kQNopePe;
};

// Returns [B, S_q, H_q, head_size_v] on the DCU device.
torch::Tensor dense_decode(DenseDecodeParams& params);

}  // namespace flash_mla
}  // namespace dcu
}  // namespace kernel
}  // namespace xllm
