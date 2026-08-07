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
