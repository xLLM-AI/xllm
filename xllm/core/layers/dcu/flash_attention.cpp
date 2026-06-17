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

#include "layers/dcu/flash_attention.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <optional>

#include "framework/kv_cache/kv_cache.h"
#include "kernels/dcu/attention_runner.h"
#include "kernels/dcu/dcu_ops_api.h"
#include "kernels/ops_api.h"
#include "layers/common/attention_metadata.h"

// Forward declarations for prefix prefill/decode kernels defined in
// flash_api.cpp
// and linked via libflash_attention.so.
//
// Packed layout (layout=1, "BSHD" in flash_api convention):
//   q:              (total_q, num_heads, head_dim), packed, no padding
//   kcache:         (num_blocks, page_block_size, num_heads_k, head_dim)
//   vcache:         (num_blocks, page_block_size, num_heads_k, head_dim)
//   cu_seqlens_q:   (batch_size + 1) int32, cumulative Q sequence lengths
//   seqused_k:      (batch_size)     int32, per-sequence KV lengths in cache
//   block_table:    (batch_size, max_num_blocks_per_seq) int32, -1 padded
//   output:         same shape as q
//
// prefix_prefill_varlen_fwd is used for the prefill phase.
// Q may contain many tokens per sequence. Uses the general fwd kernel.
std::vector<torch::Tensor> prefix_prefill_varlen_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    std::optional<torch::Tensor>& out_,
    const torch::Tensor& cu_seqlens_q,
    std::optional<torch::Tensor>& cu_seqlens_k,
    torch::Tensor& seqused_k,
    std::optional<torch::Tensor>& alibi_slopes_,
    torch::Tensor& block_table,
    const int32_t max_seqlen_q,
    const int32_t max_seqlen_k,
    const float p_dropout,
    const float softmax_scale,
    const bool zero_tensors,
    const bool is_causal,
    int32_t window_size_left,
    int32_t window_size_right,
    const float softcap,
    const bool return_softmax,
    const int32_t layout,
    std::optional<torch::Tensor> scales_q_ = std::nullopt,
    std::optional<torch::Tensor> scales_k_ = std::nullopt,
    std::optional<torch::Tensor> scales_v_ = std::nullopt,
    const bool is_bf16_output = false);

// prefix_decode_varlen_fwd is used for the decode phase and chunked prefill.
// Q is typically short (often 1 token per sequence). Uses the KV-cache kernel
// with GQA to MQA ngroups optimization and split parallelism for small batches.
std::vector<torch::Tensor> prefix_decode_varlen_fwd(
    torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    std::optional<torch::Tensor>& out_,
    const torch::Tensor& cu_seqlens_q,
    std::optional<torch::Tensor>& cu_seqlens_k,
    torch::Tensor& seqused_k,
    std::optional<torch::Tensor>& alibi_slopes_,
    torch::Tensor& block_table,
    const int32_t max_seqlen_q,
    const int32_t max_seqlen_k,
    const float p_dropout,
    const float softmax_scale,
    const bool zero_tensors,
    const bool is_causal,
    int32_t window_size_left,
    int32_t window_size_right,
    const float softcap,
    const bool return_softmax,
    const int32_t layout);

namespace xllm {
namespace layer {

namespace {

// The flash attention kernel expects -1 for invalid page slots, while xLLM's
// block table can contain a valid block_id=0. Mask only columns beyond each
// sequence's real page count, never values by block id.
torch::Tensor mask_block_table_padding(const torch::Tensor& block_table,
                                       const torch::Tensor& kv_seq_lens,
                                       int64_t page_block_size) {
  CHECK(block_table.defined()) << "block_table must be defined";
  CHECK(kv_seq_lens.defined()) << "kv_seq_lens must be defined";
  CHECK_GT(page_block_size, 0) << "page_block_size must be positive";

  const torch::TensorOptions index_options =
      torch::TensorOptions().dtype(torch::kInt64).device(block_table.device());
  torch::Tensor page_counts =
      (kv_seq_lens.to(index_options) + page_block_size - 1) / page_block_size;
  page_counts = page_counts.view({-1, 1});

  torch::Tensor col_indices =
      torch::arange(block_table.size(1), index_options).view({1, -1});
  torch::Tensor padding_mask = col_indices >= page_counts;

  torch::Tensor result = block_table.clone();
  result.masked_fill_(padding_mask, -1);
  return result;
}

// Convert kv_seq_lens / q_seq_lens from int64 (from torch::diff) to int32
// as required by the kernel.
torch::Tensor to_int32_seqlens(const torch::Tensor& t) {
  return t.to(torch::kInt32).contiguous();
}

// Get or compute per-sequence seqlens from cu_seq_lens.
torch::Tensor get_or_compute_seqlens(const torch::Tensor& per_seq,
                                     const torch::Tensor& cu_seq) {
  if (per_seq.defined()) {
    return to_int32_seqlens(per_seq);
  }
  return to_int32_seqlens(torch::diff(cu_seq));
}

// Get or fix block_table, building from paged KV metadata when undefined.
torch::Tensor get_or_build_block_table(const torch::Tensor& block_table,
                                       const torch::Tensor& paged_kv_indptr,
                                       const torch::Tensor& paged_kv_indices,
                                       const torch::Tensor& kv_seq_lens,
                                       int64_t page_block_size) {
  if (block_table.defined()) {
    return mask_block_table_padding(block_table, kv_seq_lens, page_block_size);
  }
  return xllm::kernel::dcu::build_block_table_from_paged_kv_cuda(
      paged_kv_indptr, paged_kv_indices);
}

}  // namespace

FlashAttentionImpl::FlashAttentionImpl(int64_t num_heads,
                                       int64_t head_size,
                                       float scale,
                                       int64_t num_kv_heads,
                                       int64_t sliding_window)
    : BaseAttentionImpl(num_heads,
                        head_size,
                        scale,
                        num_kv_heads,
                        sliding_window) {}

void FlashAttentionImpl::prefill_forward(const AttentionMetadata& attn_metadata,
                                         torch::Tensor& query,
                                         torch::Tensor& key,
                                         torch::Tensor& value,
                                         torch::Tensor& output,
                                         torch::Tensor k_cache,
                                         torch::Tensor v_cache) {
  // Prefill: KV has been stored into paged cache by reshape_paged_cache.
  // Q is already in packed format [total_tokens, nh, hd], no padding needed.
  // prefix_prefill_varlen_fwd reads Q, K/V cache, cu_seqlens_q, seqused_k,
  // and block_table directly.
  torch::Tensor kv_seq_lens = get_or_compute_seqlens(
      attn_metadata.kv_seq_lens, attn_metadata.kv_cu_seq_lens);
  torch::Tensor block_table =
      get_or_build_block_table(attn_metadata.block_table,
                               attn_metadata.paged_kv_indptr,
                               attn_metadata.paged_kv_indices,
                               kv_seq_lens,
                               k_cache.size(1));

  torch::Tensor cu_seqlens_q =
      attn_metadata.q_cu_seq_lens.to(torch::kInt32).contiguous();

  std::optional<torch::Tensor> out_opt = std::nullopt;
  std::optional<torch::Tensor> cu_seqlens_k_opt = std::nullopt;
  std::optional<torch::Tensor> alibi_opt = std::nullopt;

  std::vector<torch::Tensor> result = prefix_prefill_varlen_fwd(
      query,
      k_cache,
      v_cache,
      out_opt,
      cu_seqlens_q,
      cu_seqlens_k_opt,
      kv_seq_lens,
      alibi_opt,
      block_table,
      /*max_seqlen_q=*/static_cast<int32_t>(attn_metadata.max_query_len),
      /*max_seqlen_k=*/static_cast<int32_t>(attn_metadata.max_seq_len),
      /*p_dropout=*/0.0f,
      /*softmax_scale=*/scale_,
      /*zero_tensors=*/false,
      /*is_causal=*/attn_metadata.is_causal,
      /*window_size_left=*/sliding_window_,
      /*window_size_right=*/-1,
      /*softcap=*/0.0f,
      /*return_softmax=*/false,
      /*layout=*/1);

  // Output is already packed [total_tokens, nh, hd], matching query shape.
  output.copy_(result[0]);
}

void FlashAttentionImpl::paged_forward(const AttentionMetadata& attn_metadata,
                                       torch::Tensor& query,
                                       torch::Tensor& output,
                                       torch::Tensor k_cache,
                                       torch::Tensor v_cache,
                                       bool is_chunked_prefill) {
  int64_t batch_size = attn_metadata.q_seq_lens.size(0);
  int64_t max_kv_len = attn_metadata.max_seq_len;

  torch::Tensor kv_seq_lens = to_int32_seqlens(attn_metadata.kv_seq_lens);
  torch::Tensor block_table = mask_block_table_padding(
      attn_metadata.block_table, kv_seq_lens, k_cache.size(1));

  // Build cu_seqlens_q: cumulative sequence lengths for Q.
  // For decode, Q is [B, nh, hd] (1 token per seq), so cu is
  // [0, 1, 2, ..., B].
  // For chunked prefill, Q is packed [total_tokens, nh, hd] from q_cu_seq_lens.
  torch::Tensor cu_seqlens_q;
  int64_t max_q_len;
  if (is_chunked_prefill) {
    cu_seqlens_q = attn_metadata.q_cu_seq_lens.to(torch::kInt32).contiguous();
    max_q_len = attn_metadata.max_query_len;
  } else {
    cu_seqlens_q = torch::arange(
        0,
        batch_size + 1,
        torch::TensorOptions().dtype(torch::kInt32).device(query.device()));
    max_q_len = 1;
  }

  // For prefix_decode_varlen_fwd, the window semantics are:
  //   - window_left = -1 means infinite lookback (converted to seqlen_k
  //   internally)
  //   - window_left >= 0 enables a sliding window of that size
  //   - window_right = 0 with window_left < 0 produces is_causal = true
  // The "both negative" special case (line 3621 of flash_api.cpp) skips causal
  // entirely, so we must ensure window_right is not negative for causal decode.
  int64_t window_left = sliding_window_ > 0 ? sliding_window_ : -1;
  int64_t window_right = attn_metadata.is_causal ? 0 : -1;

  std::optional<torch::Tensor> out_opt = std::nullopt;
  std::optional<torch::Tensor> cu_seqlens_k_opt = std::nullopt;
  std::optional<torch::Tensor> alibi_opt = std::nullopt;

  std::vector<torch::Tensor> result = prefix_decode_varlen_fwd(
      query,
      k_cache,
      v_cache,
      out_opt,
      cu_seqlens_q,
      cu_seqlens_k_opt,
      kv_seq_lens,
      alibi_opt,
      block_table,
      /*max_seqlen_q=*/static_cast<int32_t>(max_q_len),
      /*max_seqlen_k=*/static_cast<int32_t>(max_kv_len),
      /*p_dropout=*/0.0f,
      /*softmax_scale=*/scale_,
      /*zero_tensors=*/false,
      /*is_causal=*/attn_metadata.is_causal,
      /*window_size_left=*/static_cast<int32_t>(window_left),
      /*window_size_right=*/static_cast<int32_t>(window_right),
      /*softcap=*/0.0f,
      /*return_softmax=*/false,
      /*layout=*/1);

  // Output is already packed, matching query shape.
  output.copy_(result[0]);
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
FlashAttentionImpl::forward(const AttentionMetadata& attn_metadata,
                            torch::Tensor& query,
                            torch::Tensor& key,
                            torch::Tensor& value,
                            torch::Tensor& output,
                            KVCache& kv_cache) {
  std::optional<torch::Tensor> output_lse = std::nullopt;

  if (attn_metadata.max_seq_len == 0) {
    output = output.view({-1, num_heads_ * head_size_});
    return std::make_tuple(output, output_lse);
  }

  // Reshape inputs
  query = query.view({-1, num_heads_, head_size_});
  key = key.view({-1, num_kv_heads_, head_size_});
  value = value.view({-1, num_kv_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});

  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v_cache = kv_cache.get_v_cache();

  // Store current KV into paged cache
  if (k_cache.defined() && k_cache.dim() >= 2) {
    xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
    reshape_paged_cache_params.key = key;
    reshape_paged_cache_params.value = value;
    reshape_paged_cache_params.k_cache = k_cache;
    reshape_paged_cache_params.v_cache = v_cache;
    reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
    xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);
  }

  if (attn_metadata.is_prefill) {
    AttentionMetadata attn_metadata_copy = attn_metadata;
    torch::Tensor query_saved = query;
    torch::Tensor key_saved = key;
    torch::Tensor value_saved = value;
    torch::Tensor output_3d_saved = output;
    torch::Tensor k_cache_saved = k_cache;
    torch::Tensor v_cache_saved = v_cache;
    torch::Tensor output_flat = output.view({-1, num_heads_ * head_size_});

    return ::xllm::kernel::dcu::prefill_with_optional_piecewise_capture(
        [this,
         attn_metadata_copy,
         query_saved,
         key_saved,
         value_saved,
         output_3d_saved,
         k_cache_saved,
         v_cache_saved](
            const ::xllm::kernel::dcu::AttentionReplayParams& params) mutable {
          const AttentionMetadata& replay_metadata =
              params.attn_metadata ? *params.attn_metadata : attn_metadata_copy;

          torch::Tensor query = query_saved;
          torch::Tensor key = key_saved;
          torch::Tensor value = value_saved;
          torch::Tensor output = output_3d_saved;

          prefill_forward(replay_metadata,
                          query,
                          key,
                          value,
                          output,
                          k_cache_saved,
                          v_cache_saved);

          torch::Tensor output_flat =
              output.view({-1, num_heads_ * head_size_});
          return std::make_tuple(output_flat, std::nullopt);
        },
        output_flat);
  } else {
    paged_forward(attn_metadata,
                  query,
                  output,
                  k_cache,
                  v_cache,
                  attn_metadata.is_chunked_prefill);
  }

  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

}  // namespace layer
}  // namespace xllm
