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
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "layers/common/attention_metadata.h"
#include "layers/dcu/base_attention_impl.h"

namespace xllm {
namespace layer {

torch::Tensor dense_varlen_flash_attention(torch::Tensor query,
                                           torch::Tensor key,
                                           torch::Tensor value,
                                           const torch::Tensor& cu_seqlens_q,
                                           const torch::Tensor& cu_seqlens_k,
                                           double softmax_scale,
                                           bool is_causal);

std::vector<torch::Tensor> flash_attention_varlen_forward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    int64_t num_heads,
    int64_t num_kv_heads,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    float softmax_scale,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    bool is_bf16_output,
    std::optional<torch::Tensor> output = std::nullopt,
    std::optional<torch::Tensor> seqused_k = std::nullopt,
    std::optional<torch::Tensor> alibi_slopes = std::nullopt,
    std::optional<torch::Tensor> q_descale = std::nullopt,
    std::optional<torch::Tensor> k_descale = std::nullopt,
    std::optional<torch::Tensor> v_descale = std::nullopt);

// FlashAttentionImpl uses the mha_fwd_kvcache_bshd HIP kernel from
// libflash_attention to compute paged attention directly from KV cache,
// with zero intermediate tensor copies.
class FlashAttentionImpl final : public BaseAttentionImpl {
 public:
  FlashAttentionImpl(int64_t num_heads,
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
  // Prefill: variable-length queries against contiguous KV.
  // Uses k_/v_ (new KV) + optional existing kcache.
  void prefill_forward(const AttentionMetadata& attn_metadata,
                       torch::Tensor& query,
                       torch::Tensor& key,
                       torch::Tensor& value,
                       torch::Tensor& output,
                       torch::Tensor k_cache,
                       torch::Tensor v_cache);

  // Decode or chunked prefill: single/multiple query tokens against paged KV.
  void paged_forward(const AttentionMetadata& attn_metadata,
                     torch::Tensor& query,
                     torch::Tensor& output,
                     torch::Tensor k_cache,
                     torch::Tensor v_cache,
                     bool is_chunked_prefill);
};

}  // namespace layer
}  // namespace xllm
