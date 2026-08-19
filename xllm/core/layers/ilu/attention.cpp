/* Copyright 2025-2026 The xLLM Authors.

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

#include "attention.h"

#include "kernels/ilu/ilu_ops_api.h"
#include "kernels/ops_api.h"

namespace xllm {
namespace layer {
AttentionImpl::AttentionImpl(int64_t num_heads,
                             int64_t head_size,
                             float scale,
                             int64_t num_kv_heads,
                             int64_t sliding_window)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      v_head_dim_(head_size),
      use_fused_mla_qkv_(false),
      enable_lighting_indexer_(false),
      enable_mla_(false),
      sliding_window_(sliding_window) {
  if (sliding_window_ > -1) {
    sliding_window_ = sliding_window_ - 1;
  }
}

AttentionImpl::AttentionImpl(int64_t num_heads,
                             int64_t head_size,
                             int64_t num_kv_heads,
                             int64_t v_head_dim,
                             int64_t sliding_window,
                             float scale,
                             bool use_fused_mla_qkv,
                             bool enable_lighting_indexer,
                             bool enable_mla)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      v_head_dim_(v_head_dim),
      use_fused_mla_qkv_(use_fused_mla_qkv),
      enable_lighting_indexer_(enable_lighting_indexer),
      enable_mla_(enable_mla),
      sliding_window_(sliding_window) {
  if (sliding_window_ > -1) {
    sliding_window_ = sliding_window_ - 1;
  }
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> AttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  std::optional<torch::Tensor> output_lse = std::nullopt;
  torch::Tensor output;
  if (enable_mla_) {
    output = torch::empty({query.size(0), num_heads_ * v_head_dim_},
                          query.options());
  } else {
    output = torch::empty_like(query);
  }
  if (attn_metadata.is_dummy) {
    return std::make_tuple(output, output_lse);
  }

  bool only_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;
  int64_t num_kv_heads = (enable_mla_ && !only_prefill) ? 1 : num_kv_heads_;
  torch::Tensor k_cache = kv_cache.get_k_cache();
  std::optional<torch::Tensor> v_cache;
  std::optional<torch::Tensor> v;
  if (!enable_mla_) {
    v = value.view({-1, num_kv_heads, head_size_});
    v_cache = kv_cache.get_v_cache();
  }

  bool skip_process_cache = enable_mla_ && (only_prefill || use_fused_mla_qkv_);
  if (!skip_process_cache) {
    xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
    reshape_paged_cache_params.key = key.view({-1, num_kv_heads, head_size_});
    reshape_paged_cache_params.value = v;
    reshape_paged_cache_params.k_cache = k_cache;
    reshape_paged_cache_params.v_cache = v_cache;
    reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
    xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);
  }

  if (enable_lighting_indexer_ || !only_prefill) {
    decoder_forward(query, output, k_cache, v_cache, attn_metadata);
  } else {
    prefill_forward(query, key, value, output, k_cache, v_cache, attn_metadata);
  }

  int64_t head_size = enable_mla_ ? v_head_dim_ : head_size_;
  output = output.view({-1, num_heads_ * head_size});
  return {output, output_lse};
}

void AttentionImpl::prefill_forward(torch::Tensor& query,
                                    torch::Tensor& key,
                                    torch::Tensor& value,
                                    torch::Tensor& output,
                                    const torch::Tensor& k_cache,
                                    const std::optional<torch::Tensor>& v_cache,
                                    const AttentionMetadata& attn_metadata) {
  int64_t head_size_v = enable_mla_ ? v_head_dim_ : head_size_;
  std::optional<torch::Tensor> output_lse = std::nullopt;
  query = query.view({-1, num_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_v});
  // torch::Tensor k_cache_ = k_cache;
  // torch::Tensor v_cache_ = v_cache.value();
  xllm::kernel::ilu::batch_prefill(query,
                                   k_cache,
                                   v_cache,
                                   output,
                                   output_lse,
                                   attn_metadata.q_cu_seq_lens,
                                   attn_metadata.kv_cu_seq_lens,
                                   /*alibi_slope=*/std::nullopt,
                                   /*attn_bias=*/std::nullopt,
                                   /*q_quant_scale=*/std::nullopt,
                                   /*k_quant_scale=*/std::nullopt,
                                   /*v_quant_scale=*/std::nullopt,
                                   attn_metadata.block_table,
                                   attn_metadata.max_query_len,
                                   attn_metadata.max_seq_len,
                                   scale_,
                                   attn_metadata.is_causal,
                                   sliding_window_,
                                   /*window_size_right=*/-1,
                                   attn_metadata.compute_dtype,
                                   /*return_lse=*/false);
}

void AttentionImpl::decoder_forward(torch::Tensor& query,
                                    torch::Tensor& output,
                                    const torch::Tensor& k_cache,
                                    const std::optional<torch::Tensor>& v_cache,
                                    const AttentionMetadata& attn_metadata) {
  int64_t head_size_v = enable_mla_ ? v_head_dim_ : head_size_;
  query = query.view({-1, 1, num_heads_, head_size_});
  output = output.view({-1, 1, num_heads_, head_size_v});
  std::optional<torch::Tensor> output_lse = std::nullopt;

  int64_t block_aligned_max_seq_len =
      attn_metadata.block_table.size(-1) * k_cache.size(2);

  xllm::kernel::ilu::batch_decode(query,
                                  k_cache,
                                  output,
                                  attn_metadata.block_table,
                                  attn_metadata.kv_seq_lens,
                                  v_cache,
                                  output_lse,
                                  /*q_quant_scale=*/std::nullopt,
                                  /*k_quant_scale=*/std::nullopt,
                                  /*v_quant_scale=*/std::nullopt,
                                  /*out_quant_scale=*/std::nullopt,
                                  /*alibi_slope=*/std::nullopt,
                                  attn_metadata.attn_mask,
                                  attn_metadata.compute_dtype,
                                  block_aligned_max_seq_len,
                                  sliding_window_,
                                  /*window_size_right=*/-1,
                                  scale_,
                                  /*return_lse=*/false,
                                  attn_metadata.is_causal,
                                  /*kv_cache_quant_bit_size=*/-1);
}

}  // namespace layer
}  // namespace xllm
