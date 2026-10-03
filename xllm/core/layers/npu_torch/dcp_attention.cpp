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

#include "layers/npu_torch/dcp_attention.h"

#include <glog/logging.h>

#include <limits>
#include <optional>
#include <tuple>
#include <vector>

#include "framework/parallel_state/parallel_state.h"
#include "framework/parallel_state/process_group.h"
#include "kernels/npu/npu_ops_api.h"
#include "layers/common/attention_metadata.h"
#include "layers/npu_torch/fused_infer_attention_graph.h"
#include "layers/npu_torch/fused_infer_attention_utils.h"
#include "platform/npu/acl_graph_task_update_context.h"

namespace xllm::layer {
namespace detail {

DcpAttentionResult merge_dcp_tnd_attention_shards(
    const torch::Tensor& partial_outputs,
    const torch::Tensor& partial_lse) {
  CHECK_EQ(partial_outputs.dim(), 4)
      << "partial outputs must have shape [dcp, tokens, heads, value]";
  CHECK_EQ(partial_lse.dim(), 4)
      << "partial LSE must have shape [dcp, tokens, heads, 1]";
  CHECK_EQ(partial_lse.size(-1), 1);
  for (int64_t dim = 0; dim < 3; ++dim) {
    CHECK_EQ(partial_outputs.size(dim), partial_lse.size(dim));
  }

  // FIA may leave non-finite output in an empty shard. Sanitize it before
  // applying the shared reduction's [batch, query, heads, value] layout.
  const torch::Tensor finite_lse = torch::isfinite(partial_lse);
  const torch::Tensor sanitized_lse = torch::where(
      finite_lse,
      partial_lse,
      torch::full_like(partial_lse, -std::numeric_limits<float>::infinity()));
  const torch::Tensor safe_outputs =
      torch::where(finite_lse.expand_as(partial_outputs),
                   partial_outputs,
                   torch::zeros_like(partial_outputs));
  DcpAttentionResult result = merge_dcp_attention_shards(
      safe_outputs.unsqueeze(/*dim=*/2), sanitized_lse);
  return {result.output.squeeze(/*dim=*/1), result.lse.squeeze(/*dim=*/-1)};
}

}  // namespace detail

namespace {

using detail::merge_dcp_tnd_attention_shards;
using detail::resolve_fia_sparse_mode;

void mask_empty_shard_lse(torch::Tensor& partial_lse,
                          const std::vector<int64_t>& local_lengths,
                          const std::vector<int64_t>& query_end_offsets) {
  int64_t previous_query_end = 0;
  for (size_t request_index = 0; request_index < local_lengths.size();
       ++request_index) {
    const int64_t query_end = query_end_offsets[request_index];
    if (local_lengths[request_index] == 0) {
      const int64_t query_length = query_end - previous_query_end;
      partial_lse.narrow(/*dim=*/0, previous_query_end, query_length)
          .fill_(-std::numeric_limits<float>::infinity());
    }
    previous_query_end = query_end;
  }
}

void copy_rank_local_heads(const torch::Tensor& gathered_output,
                           torch::Tensor& output,
                           int32_t dcp_rank) {
  const int64_t local_num_heads = output.size(1);
  const int64_t head_begin = static_cast<int64_t>(dcp_rank) * local_num_heads;
  output.copy_(gathered_output.slice(
      /*dim=*/1, head_begin, head_begin + local_num_heads));
}

DcpAttentionResult run_local_dcp_cache_attention(
    const torch::Tensor& gathered_query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const AttentionMetadata& attention_metadata,
    const std::vector<int64_t>& query_end_offsets,
    const std::vector<int64_t>& local_lengths,
    int64_t num_kv_heads,
    double scale,
    ProcessGroup& dcp_group) {
  const int32_t dcp_size = dcp_group.world_size();
  const int32_t dcp_rank = dcp_group.rank();
  const int64_t block_size = key_cache.size(1);
  const torch::Tensor key_view =
      key_cache.view({key_cache.size(0), block_size, -1});
  const torch::Tensor value_view =
      value_cache.view({value_cache.size(0), block_size, -1});

  torch::Tensor partial_output;
  torch::Tensor partial_lse;
  if (attention_metadata.paged_attention_tiling_data.defined()) {
    partial_output = torch::empty_like(gathered_query);
    partial_lse = detail::run_fused_infer_attention_graph(
        attention_metadata.acl_graph_task_update_context,
        gathered_query,
        key_view,
        value_view,
        attention_metadata.block_table,
        query_end_offsets,
        local_lengths,
        num_kv_heads,
        scale,
        dcp_size,
        dcp_rank,
        npu::FusedInferAttentionGraphBranch::kDecode,
        partial_output);
  } else {
    std::tie(partial_output, partial_lse) =
        kernel::npu::npu_fused_infer_attention(
            gathered_query,
            key_view,
            value_view,
            /*atten_mask=*/std::nullopt,
            std::make_optional(attention_metadata.block_table),
            query_end_offsets,
            local_lengths,
            gathered_query.size(1),
            num_kv_heads,
            scale,
            block_size,
            /*sparse_mode=*/0,
            "TND",
            /*softmax_lse_flag=*/true);
  }
  if (attention_metadata.paged_attention_tiling_data.defined()) {
    const torch::Tensor& empty_shards =
        attention_metadata.kv_shard_batch_metadata->empty_shard_mask;
    partial_lse.copy_(torch::where(
        empty_shards,
        torch::full_like(partial_lse, -std::numeric_limits<float>::infinity()),
        partial_lse));
  } else {
    mask_empty_shard_lse(partial_lse, local_lengths, query_end_offsets);
  }
  return {partial_output, partial_lse};
}

}  // namespace

void dcp_decode(const torch::Tensor& query,
                torch::Tensor& output,
                const torch::Tensor& key_cache,
                const torch::Tensor& value_cache,
                const AttentionMetadata& attention_metadata,
                int64_t num_kv_heads,
                double scale,
                ProcessGroup& dcp_group) {
  CHECK(attention_metadata.kv_shard_batch_metadata != nullptr)
      << "DCP decode requires shared cache-shard batch metadata";
  const KVShardBatchMetadata& shard_metadata =
      *attention_metadata.kv_shard_batch_metadata;
  const auto& query_end_offsets = shard_metadata.query_end_offsets;
  const torch::Tensor gathered_query =
      parallel_state::gather(query, &dcp_group, /*dim=*/1);
  const DcpAttentionResult local =
      run_local_dcp_cache_attention(gathered_query,
                                    key_cache,
                                    value_cache,
                                    attention_metadata,
                                    query_end_offsets,
                                    shard_metadata.local_kv_lengths,
                                    num_kv_heads,
                                    scale,
                                    dcp_group);
  const DcpAttentionResult merged = merge_dcp_tnd_attention_shards(
      dcp_group.allgather_base_sync(local.output.contiguous()),
      dcp_group.allgather_base_sync(local.lse.contiguous()));
  copy_rank_local_heads(merged.output, output, dcp_group.rank());
}

void dcp_chunked_prefill(const torch::Tensor& query,
                         const torch::Tensor& key,
                         const torch::Tensor& value,
                         torch::Tensor& output,
                         const torch::Tensor& key_cache,
                         const torch::Tensor& value_cache,
                         const AttentionMetadata& attention_metadata,
                         int64_t num_kv_heads,
                         double scale,
                         ProcessGroup& dcp_group) {
  const int64_t token_count = query.size(0);
  CHECK(attention_metadata.kv_shard_batch_metadata != nullptr)
      << "DCP prefill requires shared cache-shard batch metadata";
  const KVShardBatchMetadata& shard_metadata =
      *attention_metadata.kv_shard_batch_metadata;
  const auto& query_end_offsets = shard_metadata.query_end_offsets;
  const bool has_context = shard_metadata.has_context;
  const int64_t head_size = query.size(2);
  const auto [current_output_raw, current_lse_raw] =
      kernel::npu::npu_fused_infer_attention(
          query,
          key.view({token_count, num_kv_heads, head_size}),
          value.view({token_count, num_kv_heads, head_size}),
          attention_metadata.fia_attn_mask.defined()
              ? std::make_optional(attention_metadata.fia_attn_mask)
              : std::nullopt,
          /*block_table=*/std::nullopt,
          query_end_offsets,
          query_end_offsets,
          query.size(1),
          num_kv_heads,
          scale,
          /*block_size=*/0,
          resolve_fia_sparse_mode(attention_metadata),
          "TND",
          /*softmax_lse_flag=*/has_context,
          attention_metadata.is_causal,
          attention_metadata.fia_pre_tokens,
          attention_metadata.fia_next_tokens);
  if (!has_context) {
    output.copy_(current_output_raw);
    return;
  }

  const torch::Tensor gathered_query =
      parallel_state::gather(query, &dcp_group, /*dim=*/1);
  const DcpAttentionResult context =
      run_local_dcp_cache_attention(gathered_query,
                                    key_cache,
                                    value_cache,
                                    attention_metadata,
                                    query_end_offsets,
                                    shard_metadata.local_context_lengths,
                                    num_kv_heads,
                                    scale,
                                    dcp_group);
  const int64_t head_begin = dcp_group.rank() * query.size(1);
  const DcpAttentionResult merged = merge_dcp_tnd_attention_shards(
      torch::cat({dcp_group.allgather_base_sync(context.output.contiguous())
                      .narrow(/*dim=*/2, head_begin, query.size(1)),
                  current_output_raw.unsqueeze(/*dim=*/0)},
                 /*dim=*/0),
      torch::cat({dcp_group.allgather_base_sync(context.lse.contiguous())
                      .narrow(/*dim=*/2, head_begin, query.size(1)),
                  current_lse_raw.unsqueeze(/*dim=*/0)},
                 /*dim=*/0));
  output.copy_(merged.output);
}

}  // namespace xllm::layer
