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

#include "layers/dcu/flash_attention.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

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

// Dense varlen forward kernel from flash_api.cpp. This is the non-KV-cache
// path used by full-sequence DiT attention.
std::vector<torch::Tensor> varlen_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const int32_t num_heads,
    const int32_t num_heads_k,
    std::optional<torch::Tensor>& out_,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    std::optional<torch::Tensor>& seqused_k,
    std::optional<torch::Tensor>& alibi_slopes_,
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
    std::optional<at::Generator> gen_,
    const int32_t layout,
    std::optional<torch::Tensor> q_descale_ = std::nullopt,
    std::optional<torch::Tensor> k_descale_ = std::nullopt,
    std::optional<torch::Tensor> v_descale_ = std::nullopt,
    const bool is_bf16_output = true);

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

struct VarlenSeqlenInfo {
  int64_t batch_size = 0;
  int64_t total_len = 0;
  int64_t max_seq_len = 0;
  std::vector<int64_t> seq_lens;
};

VarlenSeqlenInfo validate_cu_seqlens(const torch::Tensor& cu_seqlens,
                                     const char* name) {
  CHECK(cu_seqlens.defined()) << name << " must be defined";
  CHECK_EQ(cu_seqlens.dim(), 1) << name << " must be 1D";
  CHECK_GE(cu_seqlens.numel(), 2) << name << " must include batch + 1 items";

  auto cu_cpu = cu_seqlens.to(torch::kCPU).to(torch::kInt64).contiguous();
  const auto* data = cu_cpu.data_ptr<int64_t>();
  VarlenSeqlenInfo info;
  info.batch_size = cu_cpu.numel() - 1;
  info.seq_lens.reserve(static_cast<size_t>(info.batch_size));
  CHECK_EQ(data[0], 0) << name << " must start at 0";

  for (int64_t i = 0; i < info.batch_size; ++i) {
    const int64_t current_len = data[i + 1] - data[i];
    CHECK_GT(current_len, 0)
        << name << " sequence length must be positive, batch index=" << i;
    info.seq_lens.push_back(current_len);
    info.max_seq_len = std::max(info.max_seq_len, current_len);
  }
  info.total_len = data[info.batch_size];
  return info;
}

int32_t checked_int32(int64_t value, const char* name) {
  CHECK_LE(value, static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << name << " exceeds int32 range: " << value;
  CHECK_GE(value, static_cast<int64_t>(std::numeric_limits<int32_t>::min()))
      << name << " is below int32 range: " << value;
  return static_cast<int32_t>(value);
}

}  // namespace

torch::Tensor dense_varlen_flash_attention(torch::Tensor query,
                                           torch::Tensor key,
                                           torch::Tensor value,
                                           const torch::Tensor& cu_seqlens_q,
                                           const torch::Tensor& cu_seqlens_k,
                                           double softmax_scale,
                                           bool is_causal) {
  CHECK_EQ(query.dim(), 3) << "query must be [total, heads, dim]";
  CHECK_EQ(key.dim(), 3) << "key must be [total, heads, dim]";
  CHECK_EQ(value.dim(), 3) << "value must be [total, heads, dim]";
  CHECK(query.device() == key.device() && query.device() == value.device())
      << "query/key/value must be on the same device";
  CHECK(query.scalar_type() == key.scalar_type() &&
        query.scalar_type() == value.scalar_type())
      << "query/key/value must have the same dtype";
  CHECK_EQ(key.size(0), value.size(0)) << "key/value token counts must match";
  CHECK_EQ(key.size(1), value.size(1)) << "key/value head counts must match";
  CHECK_EQ(query.size(2), key.size(2))
      << "query/key head dimensions must match";
  CHECK_EQ(query.size(2), value.size(2))
      << "query/value head dimensions must match";
  CHECK_GT(query.size(0), 0) << "query token count must be positive";
  CHECK_GT(key.size(0), 0) << "key token count must be positive";

  const VarlenSeqlenInfo q_info =
      validate_cu_seqlens(cu_seqlens_q, "cu_seqlens_q");
  const VarlenSeqlenInfo k_info =
      validate_cu_seqlens(cu_seqlens_k, "cu_seqlens_k");
  CHECK_EQ(q_info.batch_size, k_info.batch_size)
      << "q/k batch sizes must match";
  CHECK_EQ(query.size(0), q_info.total_len)
      << "query length does not match cu_seqlens_q";
  CHECK_EQ(key.size(0), k_info.total_len)
      << "key length does not match cu_seqlens_k";
  checked_int32(q_info.total_len, "cu_seqlens_q total length");
  checked_int32(k_info.total_len, "cu_seqlens_k total length");
  if (is_causal) {
    CHECK(q_info.seq_lens == k_info.seq_lens)
        << "causal dense flash-attn C API requires per-sequence q_len == "
           "kv_len";
  }

  const int64_t num_heads = query.size(1);
  const int64_t num_heads_k = key.size(1);

  const int32_t num_heads_i32 = checked_int32(num_heads, "num_heads");
  const int32_t num_heads_k_i32 = checked_int32(num_heads_k, "num_heads_k");
  const int32_t max_seqlen_q_i32 =
      checked_int32(q_info.max_seq_len, "max_seqlen_q");
  const int32_t max_seqlen_k_i32 =
      checked_int32(k_info.max_seq_len, "max_seqlen_k");

  std::optional<torch::Tensor> out_opt = std::nullopt;
  std::optional<torch::Tensor> seqused_k_opt = std::nullopt;
  std::optional<torch::Tensor> alibi_opt = std::nullopt;
  std::optional<at::Generator> gen_opt = std::nullopt;

  std::vector<torch::Tensor> result =
      varlen_fwd(query.contiguous(),
                 key.contiguous(),
                 value.contiguous(),
                 num_heads_i32,
                 num_heads_k_i32,
                 out_opt,
                 cu_seqlens_q.to(query.device()).to(torch::kInt32).contiguous(),
                 cu_seqlens_k.to(query.device()).to(torch::kInt32).contiguous(),
                 seqused_k_opt,
                 alibi_opt,
                 max_seqlen_q_i32,
                 max_seqlen_k_i32,
                 /*p_dropout=*/0.0f,
                 /*softmax_scale=*/static_cast<float>(softmax_scale),
                 /*zero_tensors=*/false,
                 /*is_causal=*/is_causal,
                 /*window_size_left=*/-1,
                 /*window_size_right=*/is_causal ? 0 : -1,
                 /*softcap=*/0.0f,
                 /*return_softmax=*/false,
                 gen_opt,
                 /*layout=*/1,
                 /*q_descale_=*/std::nullopt,
                 /*k_descale_=*/std::nullopt,
                 /*v_descale_=*/std::nullopt,
                 /*is_bf16_output=*/query.scalar_type() == at::kBFloat16);

  return result[0];
}

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
  const bool has_paged_kv_cache = k_cache.defined() && k_cache.dim() >= 2 &&
                                  v_cache.defined() && v_cache.dim() >= 2;
  CHECK(has_paged_kv_cache)
      << "FlashAttentionImpl requires paged KV cache for prefill. "
      << "Set use_dense_flash_attention for uncached dense prefill.";

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

  if (attn_metadata.use_dense_flash_attention) {
    CHECK(attn_metadata.is_prefill)
        << "dense flash-attn C API is only supported for prefill";
    auto dense_output =
        dense_varlen_flash_attention(query,
                                     key,
                                     value,
                                     attn_metadata.q_cu_seq_lens,
                                     attn_metadata.kv_cu_seq_lens,
                                     /*softmax_scale=*/scale_,
                                     attn_metadata.is_causal);
    output.copy_(dense_output);
    output = output.view({-1, num_heads_ * head_size_});
    return {output, output_lse};
  }

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
