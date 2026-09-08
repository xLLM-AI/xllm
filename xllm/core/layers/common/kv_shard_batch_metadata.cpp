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

#include "layers/common/kv_shard_batch_metadata.h"

#include <glog/logging.h>

#include "layers/common/attention_metadata.h"

namespace xllm::layer {

torch::Tensor localize_kv_shard_slots(const torch::Tensor& logical_slots,
                                      const KVShardLayout& layout) {
  CHECK(logical_slots.scalar_type() == torch::kInt32 ||
        logical_slots.scalar_type() == torch::kInt64)
      << "cache-shard slot mapping must use int32 or int64";
  torch::Tensor valid_slots = logical_slots >= 0;
  torch::Tensor safe_slots = torch::clamp_min(logical_slots, 0);
  torch::Tensor logical_offsets =
      torch::remainder(safe_slots, layout.logical_block_size());
  torch::Tensor owner_ranks =
      torch::floor_divide(logical_offsets, layout.physical_block_size());
  torch::Tensor owned_slots =
      torch::logical_and(valid_slots, owner_ranks == layout.dcp_rank());
  torch::Tensor logical_block_ids =
      torch::floor_divide(safe_slots, layout.logical_block_size());
  torch::Tensor local_offsets =
      torch::remainder(logical_offsets, layout.physical_block_size());
  torch::Tensor local_slots =
      logical_block_ids * layout.physical_block_size() + local_offsets;
  return torch::where(
      owned_slots,
      local_slots,
      torch::full_like(local_slots, KVShardLayout::kInvalidSlot));
}

torch::Tensor localize_kv_shard_context_lens(
    const torch::Tensor& global_context_lens,
    const KVShardLayout& layout) {
  CHECK(global_context_lens.scalar_type() == torch::kInt32 ||
        global_context_lens.scalar_type() == torch::kInt64)
      << "cache-shard context lengths must use int32 or int64";
  torch::Tensor nonnegative_lens = torch::clamp_min(global_context_lens, 0);
  const int64_t logical_block_size = layout.logical_block_size();
  const int64_t physical_block_size = layout.physical_block_size();
  const int64_t rank_start =
      static_cast<int64_t>(layout.dcp_rank()) * physical_block_size;
  torch::Tensor full_logical_blocks =
      torch::floor_divide(nonnegative_lens, logical_block_size);
  torch::Tensor logical_block_remainder =
      torch::remainder(nonnegative_lens, logical_block_size);
  torch::Tensor owned_remainder =
      torch::clamp(logical_block_remainder - rank_start,
                   /*min=*/0,
                   /*max=*/physical_block_size);
  return full_logical_blocks * physical_block_size + owned_remainder;
}

KVShardCausalSelectorMetadata build_kv_shard_causal_selector_metadata(
    const AttentionMetadata& attention_metadata,
    const KVShardLayout& layout) {
  CHECK(attention_metadata.q_cu_seq_lens.defined())
      << "cache-shard causal selector requires query cumulative lengths";
  CHECK(attention_metadata.kv_cu_seq_lens.defined())
      << "cache-shard causal selector requires KV cumulative lengths";
  CHECK(attention_metadata.block_table.defined())
      << "cache-shard causal selector requires a block table";
  CHECK(attention_metadata.slot_mapping.defined())
      << "cache-shard causal selector requires slot mapping";
  CHECK_EQ(attention_metadata.q_cu_seq_lens.dim(), 1)
      << "cache-shard causal selector query lengths must be one-dimensional";
  CHECK_EQ(attention_metadata.kv_cu_seq_lens.dim(), 1)
      << "cache-shard causal selector KV lengths must be one-dimensional";
  CHECK_EQ(attention_metadata.q_cu_seq_lens.numel(),
           attention_metadata.kv_cu_seq_lens.numel())
      << "cache-shard causal selector query and KV batches must match";
  CHECK_EQ(attention_metadata.block_table.size(0),
           attention_metadata.q_cu_seq_lens.numel() - 1)
      << "cache-shard causal selector block-table batch must match lengths";
  CHECK_EQ(attention_metadata.q_cu_seq_lens.scalar_type(), torch::kInt32)
      << "cache-shard causal selector query lengths must be int32";
  CHECK_EQ(attention_metadata.kv_cu_seq_lens.scalar_type(), torch::kInt32)
      << "cache-shard causal selector KV lengths must be int32";

  torch::Tensor query_lens = torch::diff(attention_metadata.q_cu_seq_lens);
  torch::Tensor kv_lens = torch::diff(attention_metadata.kv_cu_seq_lens);
  torch::Tensor prefix_lens = kv_lens - query_lens;
  const int64_t token_count = attention_metadata.slot_mapping.numel();

  torch::Tensor token_prefix_lens =
      torch::repeat_interleave(prefix_lens, query_lens, /*dim=*/0);
  torch::Tensor token_query_starts = torch::repeat_interleave(
      attention_metadata.q_cu_seq_lens.slice(/*dim=*/0,
                                             /*start=*/0,
                                             /*end=*/-1),
      query_lens,
      /*dim=*/0);
  torch::Tensor token_offsets =
      torch::arange(token_count, attention_metadata.q_cu_seq_lens.options());
  torch::Tensor global_context_lens =
      token_prefix_lens + token_offsets - token_query_starts + 1;
  torch::Tensor query_block_table = attention_metadata.block_table
                                        .repeat_interleave(query_lens,
                                                           /*dim=*/0)
                                        .contiguous();
  torch::Tensor selector_q_cu_seq_lens = torch::arange(
      token_count + 1, attention_metadata.q_cu_seq_lens.options());
  return KVShardCausalSelectorMetadata{
      std::move(query_block_table),
      localize_kv_shard_context_lens(global_context_lens, layout),
      std::move(selector_q_cu_seq_lens)};
}

torch::Tensor expand_kv_shard_indexer_block_table(
    const torch::Tensor& logical_block_table,
    const KVShardLayout& layout) {
  CHECK_EQ(logical_block_table.dim(), 2)
      << "cache-shard indexer block table must be two-dimensional";
  torch::Tensor shard_offsets =
      torch::arange(layout.dcp_size(), logical_block_table.options());
  torch::Tensor expanded =
      logical_block_table.unsqueeze(-1) * layout.dcp_size() + shard_offsets;
  expanded = torch::where(logical_block_table.unsqueeze(-1) >= 0,
                          expanded,
                          torch::full_like(expanded, -1));
  return expanded.flatten(/*start_dim=*/1).contiguous();
}

std::shared_ptr<const KVShardBatchMetadata> build_kv_shard_batch_metadata(
    const AttentionMetadata& attention_metadata,
    const KVShardLayout& layout) {
  CHECK(attention_metadata.slot_mapping.defined())
      << "cache-shard batch metadata requires slot mapping";
  auto metadata = std::make_shared<KVShardBatchMetadata>();
  metadata->kv_split_size = layout.dcp_size();
  metadata->kv_split_rank = layout.dcp_rank();
  metadata->local_slot_mapping =
      localize_kv_shard_slots(attention_metadata.slot_mapping, layout);
  if (attention_metadata.block_table.defined()) {
    metadata->expanded_indexer_block_table =
        expand_kv_shard_indexer_block_table(attention_metadata.block_table,
                                            layout);
  }
  return metadata;
}

}  // namespace xllm::layer
