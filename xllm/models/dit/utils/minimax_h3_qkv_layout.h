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

#include <cstdint>
#include <stdexcept>

namespace xllm {

// Per-rank grouped [head,QKV,dim,input] -> split [QKV,head,dim,input].
// The caller copies this temporary back into its one registered weight.
inline torch::Tensor minimax_h3_split_grouped_qkv_shard(
    const torch::Tensor& grouped,
    int64_t heads,
    int64_t head_dim) {
  if (heads <= 0 || head_dim <= 0 || !grouped.defined() || grouped.dim() != 2 ||
      grouped.size(0) != 3 * heads * head_dim) {
    throw std::invalid_argument("H3 grouped QKV shard metadata mismatch");
  }
  return grouped.view({heads, 3, head_dim, grouped.size(1)})
      .permute({1, 0, 2, 3})
      .reshape(grouped.sizes())
      .contiguous();
}

}  // namespace xllm
