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

#include "attention.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <vector>

#include "framework/parallel_state/parallel_state.h"
#include "kernels/npu/npu_ops_api.h"
#include "kernels/ops_api.h"

namespace {

std::vector<int64_t> compute_dcp_local_kv_seq_lens(
    const std::vector<int64_t>& global_kv_seq_lens,
    int32_t dcp_size,
    int32_t dcp_rank,
    int64_t block_size) {
  CHECK_GT(dcp_size, 1);
  CHECK_GE(dcp_rank, 0);
  CHECK_LT(dcp_rank, dcp_size);
  CHECK_GT(block_size, 0);

  std::vector<int64_t> local_kv_seq_lens;
  local_kv_seq_lens.reserve(global_kv_seq_lens.size());
  for (const int64_t global_kv_seq_len : global_kv_seq_lens) {
    CHECK_GE(global_kv_seq_len, 0);
    const int64_t base = global_kv_seq_len / block_size / dcp_size * block_size;
    const int64_t remainder = global_kv_seq_len - base * dcp_size;
    const int64_t rank_offset = static_cast<int64_t>(dcp_rank) * block_size;
    const int64_t local_remainder =
        std::clamp(remainder - rank_offset, int64_t{0}, block_size);
    local_kv_seq_lens.emplace_back(base + local_remainder);
  }
  return local_kv_seq_lens;
}

void validate_dcp_decode_lengths(const std::vector<int64_t>& q_cu_seq_lens,
                                 const std::vector<int64_t>& global_kv_seq_lens,
                                 int64_t token_count) {
  CHECK(!q_cu_seq_lens.empty())
      << "DCP decode requires host cumulative query lengths.";
  CHECK_EQ(q_cu_seq_lens.size(), global_kv_seq_lens.size())
      << "DCP decode requires one query and KV length per request.";
  CHECK_EQ(token_count, static_cast<int64_t>(global_kv_seq_lens.size()))
      << "DCP supports only one-token decode requests.";

  int64_t previous_q_end = 0;
  for (const int64_t q_end : q_cu_seq_lens) {
    CHECK_EQ(q_end - previous_q_end, 1)
        << "DCP supports only one-token decode requests.";
    previous_q_end = q_end;
  }
  CHECK_EQ(previous_q_end, token_count)
      << "DCP cumulative query lengths do not match query tokens.";
}

void normalize_zero_dcp_partials(
    torch::Tensor& partial_out,
    torch::Tensor& partial_lse,
    const std::vector<int64_t>& local_kv_seq_lens) {
  CHECK_EQ(partial_out.scalar_type(), torch::kFloat32);
  CHECK_EQ(partial_lse.scalar_type(), torch::kFloat32);
  CHECK_EQ(partial_out.dim(), 3);
  CHECK_EQ(partial_lse.dim(), 3);
  CHECK_EQ(partial_out.size(0), partial_lse.size(0));
  CHECK_EQ(partial_out.size(1), partial_lse.size(1));
  CHECK_EQ(partial_lse.size(2), 1);
  CHECK_EQ(partial_out.size(0), static_cast<int64_t>(local_kv_seq_lens.size()));

  for (int64_t request_index = 0;
       request_index < static_cast<int64_t>(local_kv_seq_lens.size());
       ++request_index) {
    if (local_kv_seq_lens[request_index] == 0) {
      partial_out.select(0, request_index).zero_();
      partial_lse.select(0, request_index)
          .fill_(-std::numeric_limits<float>::infinity());
    }
  }
}

torch::Tensor merge_dcp_partials(const torch::Tensor& all_partial_out,
                                 const torch::Tensor& all_partial_lse) {
  CHECK_EQ(all_partial_out.scalar_type(), torch::kFloat32);
  CHECK_EQ(all_partial_lse.scalar_type(), torch::kFloat32);
  CHECK_EQ(all_partial_out.dim(), 4);
  CHECK_EQ(all_partial_lse.dim(), 4);
  CHECK_EQ(all_partial_out.size(0), all_partial_lse.size(0));
  CHECK_EQ(all_partial_out.size(1), all_partial_lse.size(1));
  CHECK_EQ(all_partial_out.size(2), all_partial_lse.size(2));
  CHECK_EQ(all_partial_lse.size(3), 1);

  const torch::Tensor max_lse = std::get<0>(all_partial_lse.max(0));
  const torch::Tensor max_lse_is_finite = torch::isfinite(max_lse);
  const torch::Tensor safe_max_lse =
      torch::where(max_lse_is_finite, max_lse, torch::zeros_like(max_lse));
  const torch::Tensor weights =
      torch::where(torch::isfinite(all_partial_lse),
                   torch::exp(all_partial_lse - safe_max_lse),
                   torch::zeros_like(all_partial_lse));
  const torch::Tensor denominator = weights.sum(0);
  const torch::Tensor safe_denominator = torch::where(
      denominator.gt(0), denominator, torch::ones_like(denominator));
  const torch::Tensor merged_out =
      (weights * all_partial_out).sum(0) / safe_denominator;
  return torch::where(max_lse_is_finite.expand_as(merged_out),
                      merged_out,
                      torch::zeros_like(merged_out));
}

}  // namespace

namespace xllm {
namespace layer {

AttentionImpl::AttentionImpl(int64_t num_heads,
                             int64_t head_size,
                             float scale,
                             int64_t num_kv_heads,
                             int64_t sliding_window,
                             int32_t dcp_size,
                             int32_t dcp_rank,
                             ProcessGroup* dcp_group)
    : num_heads_(num_heads),
      head_size_(head_size),
      num_kv_heads_(num_kv_heads),
      sliding_window_(sliding_window),
      scale_(scale),
      dcp_size_(dcp_size),
      dcp_rank_(dcp_rank),
      dcp_group_(dcp_group) {
  CHECK_GT(dcp_size_, 0) << "dcp_size must be positive.";
  CHECK_GE(dcp_rank_, 0) << "dcp_rank must be non-negative.";
  CHECK_LT(dcp_rank_, dcp_size_) << "dcp_rank must be smaller than dcp_size.";
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
  torch::Tensor output = torch::empty_like(query);

  if (attn_metadata.is_dummy) {
    return std::make_tuple(output, output_lse);
  }

  bool only_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;

  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v = value.view({-1, num_kv_heads_, head_size_});
  std::optional<torch::Tensor> v_cache = kv_cache.get_v_cache();

  // Reshape and cache key/value
  xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
  reshape_paged_cache_params.key = key.view({-1, num_kv_heads_, head_size_});
  reshape_paged_cache_params.value = v;
  reshape_paged_cache_params.k_cache = k_cache;
  reshape_paged_cache_params.v_cache = v_cache;
  reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
  xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);

  if (attn_metadata.use_expanded_decode_for_spec_verify_attention) {
    decoder_forward(query, output, k_cache, v_cache, attn_metadata);
  } else if (only_prefill) {
    prefill_forward(query, key, value, output, k_cache, v_cache, attn_metadata);
  } else {
    decoder_forward(query, output, k_cache, v_cache, attn_metadata);
  }

  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

void AttentionImpl::prefill_forward(torch::Tensor& query,
                                    torch::Tensor& key,
                                    torch::Tensor& value,
                                    torch::Tensor& output,
                                    const torch::Tensor& k_cache,
                                    const std::optional<torch::Tensor>& v_cache,
                                    const AttentionMetadata& attn_metadata) {
  query = query.view({-1, num_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});

  if (attn_metadata.is_prefill) {
    key = key.view({-1, num_kv_heads_, head_size_});
    value = value.view({-1, num_kv_heads_, head_size_});

    auto fia_result = xllm::kernel::npu::npu_fused_infer_attention(
        query,
        key,
        value,
        attn_metadata.fia_attn_mask.defined()
            ? std::make_optional(attn_metadata.fia_attn_mask)
            : std::nullopt,
        std::nullopt,
        attn_metadata.q_cu_seq_lens_host_vec,
        attn_metadata.kv_cu_seq_lens_host_vec,
        num_heads_,
        num_kv_heads_,
        scale_,
        /*block_size=*/0,
        /*sparse_mode=*/3,
        "TND");
    output.copy_(std::get<0>(fia_result).view_as(output));
  } else if (attn_metadata.is_chunked_prefill) {
    torch::Tensor k = k_cache.view({k_cache.size(0), k_cache.size(1), -1});
    torch::Tensor v = v_cache.value().view(
        {v_cache.value().size(0), v_cache.value().size(1), -1});
    auto fia_result = xllm::kernel::npu::npu_fused_infer_attention(
        query,
        k,
        v,
        attn_metadata.fia_attn_mask.defined()
            ? std::make_optional(attn_metadata.fia_attn_mask)
            : std::nullopt,
        attn_metadata.block_table.defined()
            ? std::make_optional(attn_metadata.block_table)
            : std::nullopt,
        attn_metadata.q_cu_seq_lens_host_vec,
        attn_metadata.kv_seq_lens_host_vec,
        num_heads_,
        num_kv_heads_,
        scale_,
        /*block_size=*/k_cache.size(1),
        /*sparse_mode=*/3,
        "TND");
    output.copy_(std::get<0>(fia_result).view_as(output));
  }
}

void AttentionImpl::dcp_decoder_forward(
    torch::Tensor& query,
    torch::Tensor& output,
    const torch::Tensor& k_cache,
    const std::optional<torch::Tensor>& v_cache,
    const AttentionMetadata& attn_metadata) {
  CHECK(dcp_group_ != nullptr) << "DCP decode requires a DCP process group.";
  CHECK_EQ(dcp_group_->world_size(), dcp_size_)
      << "DCP process group size does not match attention DCP size.";
  CHECK_EQ(dcp_group_->rank(), dcp_rank_)
      << "DCP process group rank does not match attention DCP rank.";
  CHECK(!attn_metadata.is_prefill);
  CHECK(!attn_metadata.is_chunked_prefill);
  CHECK(!attn_metadata.is_spec_verify)
      << "DCP-2 does not support speculative decode attention.";
  CHECK(!attn_metadata.use_expanded_decode_for_spec_verify_attention)
      << "DCP-2 does not support speculative decode attention.";
  CHECK(!attn_metadata.paged_attention_tiling_data.defined())
      << "DCP-2 does not support graph-captured decode attention.";
  CHECK(v_cache.has_value() && v_cache.value().defined())
      << "DCP decode requires a defined V cache.";
  CHECK(attn_metadata.block_table.defined())
      << "DCP decode requires a paged KV block table.";

  const int64_t token_count = query.size(0);
  const std::vector<int64_t>& q_cu_seq_lens =
      attn_metadata.q_cu_seq_lens_host_vec;
  const std::vector<int64_t>& global_kv_seq_lens =
      attn_metadata.kv_seq_lens_host_vec;
  validate_dcp_decode_lengths(q_cu_seq_lens, global_kv_seq_lens, token_count);

  const int64_t block_size = k_cache.size(1);
  const std::vector<int64_t> local_kv_seq_lens = compute_dcp_local_kv_seq_lens(
      global_kv_seq_lens, dcp_size_, dcp_rank_, block_size);
  const torch::Tensor local_block_table =
      parallel_state::select_dcp_local_block_table(
          attn_metadata.block_table, dcp_size_, dcp_rank_);
  CHECK_EQ(local_block_table.size(0), token_count)
      << "DCP local block table batch size does not match decode tokens.";

  const torch::Tensor query_group =
      parallel_state::gather(query, dcp_group_, 1);
  const int64_t group_num_heads = num_heads_ * static_cast<int64_t>(dcp_size_);
  CHECK_EQ(query_group.dim(), 3);
  CHECK_EQ(query_group.size(0), token_count);
  CHECK_EQ(query_group.size(1), group_num_heads);
  CHECK_EQ(query_group.size(2), head_size_);
  CHECK_EQ(group_num_heads % num_kv_heads_, 0)
      << "DCP gathered Q heads must preserve the GQA ratio.";

  const torch::Tensor k = k_cache.view({k_cache.size(0), k_cache.size(1), -1});
  const torch::Tensor v = v_cache.value().view(
      {v_cache.value().size(0), v_cache.value().size(1), -1});
  const std::optional<torch::Tensor> no_mask = std::nullopt;
  const std::optional<torch::Tensor> local_block_table_opt = local_block_table;
  const auto fia_result =
      xllm::kernel::npu::npu_fused_infer_attention(query_group,
                                                   k,
                                                   v,
                                                   no_mask,
                                                   local_block_table_opt,
                                                   q_cu_seq_lens,
                                                   local_kv_seq_lens,
                                                   group_num_heads,
                                                   num_kv_heads_,
                                                   scale_,
                                                   block_size,
                                                   0,
                                                   "TND",
                                                   true);
  torch::Tensor partial_out = std::get<0>(fia_result).to(torch::kFloat32);
  torch::Tensor partial_lse = std::get<1>(fia_result).to(torch::kFloat32);
  normalize_zero_dcp_partials(partial_out, partial_lse, local_kv_seq_lens);

  const torch::Tensor all_partial_out =
      dcp_group_->allgather_base_sync(partial_out);
  const torch::Tensor all_partial_lse =
      dcp_group_->allgather_base_sync(partial_lse);
  const torch::Tensor merged_out =
      merge_dcp_partials(all_partial_out, all_partial_lse);
  const int64_t head_begin = static_cast<int64_t>(dcp_rank_) * num_heads_;
  const torch::Tensor local_out =
      merged_out.slice(1, head_begin, head_begin + num_heads_);
  output.copy_(local_out.to(output.scalar_type()));
}

void AttentionImpl::decoder_forward(torch::Tensor& query,
                                    torch::Tensor& output,
                                    const torch::Tensor& k_cache,
                                    const std::optional<torch::Tensor>& v_cache,
                                    const AttentionMetadata& attn_metadata) {
  query = query.view({-1, num_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});

  if (dcp_size_ > 1) {
    dcp_decoder_forward(query, output, k_cache, v_cache, attn_metadata);
    return;
  }

  query = query.view({-1, 1, num_heads_, head_size_});
  output = output.view({-1, 1, num_heads_, head_size_});

  torch::Tensor kv_seq_lens;
  torch::Tensor block_table = attn_metadata.block_table;
  torch::Tensor tiling_data = attn_metadata.paged_attention_tiling_data;
  if (attn_metadata.use_expanded_decode_for_spec_verify_attention) {
    block_table = attn_metadata.expanded_block_table;
    tiling_data = attn_metadata.expanded_paged_attention_tiling_data;
    if (attn_metadata.expanded_kv_seq_lens_host.defined()) {
      kv_seq_lens = attn_metadata.expanded_kv_seq_lens_host;
    } else {
      kv_seq_lens = attn_metadata.expanded_kv_seq_lens;
    }
  } else if (attn_metadata.kv_seq_lens_host.defined()) {
    kv_seq_lens = attn_metadata.kv_seq_lens_host;
  } else {
    // Fallback if host tensor isn't prepared.
    kv_seq_lens = attn_metadata.kv_seq_lens;
  }

  if (tiling_data.defined()) {
    // Use CustomPagedAttention for ACL graph mode to avoid .to(kCPU) operations

    xllm::kernel::npu::batch_decode_acl_graph(query,
                                              k_cache,
                                              v_cache.value_or(torch::Tensor()),
                                              scale_,
                                              block_table,
                                              kv_seq_lens,
                                              tiling_data,
                                              output);
  } else {
    // Standard PagedAttention path
    xllm::kernel::npu::batch_decode(query,
                                    k_cache,
                                    v_cache.value_or(torch::Tensor()),
                                    scale_,
                                    block_table,
                                    kv_seq_lens,
                                    output);
  }
}

}  // namespace layer
}  // namespace xllm
