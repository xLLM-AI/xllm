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

#include "layers/mlu/dcp_decode_context.h"

#include <glog/logging.h>

#include <utility>

#include "framework/parallel_state/process_group.h"
#include "layers/common/kv_shard_batch_metadata.h"

namespace xllm::layer {

DcpDecodeContext::DcpDecodeContext(KVShardLayout layout,
                                   ProcessGroup* dcp_group)
    : layout_(std::move(layout)), dcp_group_(dcp_group) {}

torch::Tensor DcpDecodeContext::localize_slots(
    const torch::Tensor& global_slots) const {
  return localize_kv_shard_slots(global_slots, layout_);
}

torch::Tensor DcpDecodeContext::expand_indexer_block_table(
    const torch::Tensor& logical_block_table) const {
  return expand_kv_shard_indexer_block_table(logical_block_table, layout_);
}

DsaTopkState DcpDecodeContext::localize_topk(
    const DsaTopkState& global_state) const {
  const torch::Tensor& global_table = global_state.block_tables();
  const torch::Tensor& global_context_lens = global_state.context_lens();
  CHECK_EQ(global_table.dim(), 2)
      << "DCP decode requires a two-dimensional sparse block table";
  CHECK_EQ(global_table.size(0), global_context_lens.numel());

  const int64_t width = global_table.size(1);
  torch::Tensor columns =
      torch::arange(width, global_table.options().dtype(torch::kInt64));
  torch::Tensor within_context =
      columns.unsqueeze(0) < global_context_lens.unsqueeze(1);
  torch::Tensor local_table = localize_slots(global_table);
  torch::Tensor owned_entries =
      torch::logical_and(within_context, local_table >= 0);
  torch::Tensor sort_keys =
      torch::where(owned_entries,
                   columns.unsqueeze(0).expand_as(local_table),
                   torch::full_like(local_table, width).to(torch::kInt64));
  torch::Tensor gather_indices = std::get<1>(torch::sort(sort_keys, /*dim=*/1));
  torch::Tensor packed_table = local_table.gather(/*dim=*/1, gather_indices);
  torch::Tensor local_context_lens =
      owned_entries.sum(/*dim=*/1).to(global_context_lens.scalar_type());
  torch::Tensor packed_columns = columns.unsqueeze(0).expand_as(local_table);
  packed_table = torch::where(packed_columns < local_context_lens.unsqueeze(1),
                              packed_table,
                              torch::zeros_like(packed_table));
  return DsaTopkState(packed_table.contiguous(),
                      local_context_lens.contiguous());
}

torch::Tensor DcpDecodeContext::gather_topk_cache(
    const torch::Tensor& global_slots,
    const torch::Tensor& local_cache) const {
  CHECK_EQ(global_slots.dim(), 2)
      << "DCP global top-k slots must be two-dimensional";
  CHECK_EQ(local_cache.dim(), 4)
      << "DCP MLA cache must have shape [blocks, heads, block, latent]";
  CHECK_EQ(local_cache.size(1), 1)
      << "DCP MLA cache must use a single latent head";
  CHECK_EQ(local_cache.size(2), layout_.physical_block_size());

  torch::Tensor local_slots = localize_slots(global_slots);
  torch::Tensor safe_local_slots =
      torch::clamp_min(local_slots, 0).to(torch::kLong);
  torch::Tensor flat_cache = local_cache.flatten(/*start_dim=*/0,
                                                 /*end_dim=*/2);
  torch::Tensor selected = flat_cache.index_select(
      /*dim=*/0, safe_local_slots.flatten());
  selected = selected.view(
      {global_slots.size(0), global_slots.size(1), local_cache.size(3)});
  selected = torch::where(
      (local_slots >= 0).unsqueeze(-1), selected, torch::zeros_like(selected));
  if (dcp_group_ != nullptr && dcp_group_->world_size() > 1) {
    dcp_group_->allreduce(selected);
  }
  return selected;
}

DcpAttentionResult DcpDecodeContext::merge(
    const torch::Tensor& local_output,
    const torch::Tensor& local_lse) const {
  if (dcp_group_ == nullptr || dcp_group_->world_size() == 1) {
    return {local_output, local_lse};
  }
  return all_gather_and_merge_dcp_attention(
      local_output, local_lse, *dcp_group_);
}

}  // namespace xllm::layer
