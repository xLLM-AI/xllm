/* Copyright 2025-2026 The xLLM Authors.

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

#pragma once

#include <torch/torch.h>

#include <optional>
#include <tuple>
#include <utility>

#include "framework/kv_cache/kv_cache.h"
#include "layers/common/attention_metadata.h"
#include "layers/dcu/base_attention_impl.h"

namespace xllm {
namespace layer {

// TorchAttentionImpl implements attention computation using PyTorch's
// aten::scaled_dot_product_attention operator. This provides a pure
// PyTorch implementation that works across different hardware backends.
class TorchAttentionImpl final : public BaseAttentionImpl {
 public:
  TorchAttentionImpl(int64_t num_heads,
                     int64_t head_size,
                     float scale,
                     int64_t num_kv_heads,
                     int64_t sliding_window);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      torch::Tensor& output,
      KVCache& kv_cache) override;

 private:
  // Helper function to compute scaled dot product attention.
  torch::Tensor compute_attention(
      const torch::Tensor& query,  // [seq_len, num_heads, head_size]
      const torch::Tensor& key,    // [seq_len, num_kv_heads, head_size]
      const torch::Tensor& value,  // [seq_len, num_kv_heads, head_size]
      bool is_causal,
      const std::optional<torch::Tensor>& attn_mask = std::nullopt);

  // Helper function to expand kv heads to match query heads.
  std::pair<torch::Tensor, torch::Tensor> expand_kv_for_mqa(
      const torch::Tensor& key,
      const torch::Tensor& value);
};

}  // namespace layer
}  // namespace xllm
