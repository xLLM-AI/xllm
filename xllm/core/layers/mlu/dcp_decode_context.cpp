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

#include "layers/mlu/dcp_decode_context.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <utility>

#include "framework/parallel_state/process_group.h"
#include "kernels/mlu/mlu_ops_api.h"
#include "layers/common/kv_shard_batch_metadata.h"

namespace xllm::layer {

DcpDecodeContext::DcpDecodeContext(KVShardLayout layout,
                                   ProcessGroup* dcp_group)
    : layout_(std::move(layout)), dcp_group_(dcp_group) {}

int32_t DcpDecodeContext::world_size() const {
  return dcp_group_ == nullptr ? 1 : dcp_group_->world_size();
}

torch::Tensor DcpDecodeContext::localize_slots(
    const torch::Tensor& global_slots) const {
  return localize_kv_shard_slots(global_slots, layout_);
}

DsaTopkState DcpDecodeContext::localize_topk(
    const DsaTopkState& global_state) const {
  const torch::Tensor& global_table = global_state.block_tables();
  const torch::Tensor& global_context_lens = global_state.context_lens();
  CHECK_EQ(global_table.dim(), 2) << "Flatten MTP TopK before localization";
  const int64_t rows = global_table.size(0);
  const int64_t width = global_table.size(1);
  torch::Tensor local_table =
      torch::empty({rows, width}, global_table.options());
  torch::Tensor local_lens =
      torch::empty({rows}, global_context_lens.options());
  kernel::mlu::dcp_localize_topk(global_table,
                                 global_context_lens,
                                 layout_.dcp_rank(),
                                 layout_.dcp_size(),
                                 layout_.physical_block_size(),
                                 local_table,
                                 local_lens);

  return DsaTopkState(std::move(local_table), std::move(local_lens));
}

DcpIndexerGatherAsyncCtx DcpDecodeContext::launch_indexer_candidate_gather(
    const torch::Tensor& local_scores,
    const torch::Tensor& local_global_slots) const {
  return launch_dcp_indexer_candidate_gather(
      local_scores, local_global_slots, dcp_group_);
}

DsaTopkState DcpDecodeContext::finish_indexer_candidate_merge(
    DcpIndexerGatherAsyncCtx&& gather,
    int64_t topk,
    const torch::Tensor& slot_mapping) const {
  const DcpIndexerGatheredCandidates gathered =
      finish_dcp_indexer_candidate_gather(std::move(gather));
  const int64_t query_count = gathered.scores.size(1);
  torch::Tensor output_slots =
      torch::empty({query_count, topk}, gathered.global_slots.options());
  torch::Tensor context_lens =
      torch::empty({query_count}, gathered.global_slots.options());
  kernel::mlu::dcp_merge_topk(gathered.scores,
                              gathered.global_slots,
                              slot_mapping,
                              topk,
                              output_slots,
                              context_lens);
  return DsaTopkState(std::move(output_slots), std::move(context_lens));
}

torch::Tensor DcpDecodeContext::merge(const torch::Tensor& local_output,
                                      const torch::Tensor& local_lse,
                                      const torch::Tensor& slot_mapping,
                                      bool head_sharded) const {
  if (world_size() <= 1) {
    return local_output;
  }
  ProcessGroup& dcp_group = *dcp_group_;
  const int64_t world_size = dcp_group.world_size();

  torch::Tensor lses =
      dcp_group.allgather_base_sync(local_lse.contiguous());  // [N, B, H, 1]
  lses = lses.squeeze(-1).contiguous();                       // [N, B, H]

  torch::Tensor out = local_output.squeeze(1).contiguous();  // [B, H, V]
  const int64_t batch_size = out.size(0);
  const int64_t heads = out.size(1);
  const int64_t head_dim = out.size(2);
  CHECK_EQ(slot_mapping.numel(), batch_size)
      << "DCP attention slot mappings must match output rows";

  if (head_sharded) {
    // DCP spans the TP ranks, so heads are distributed: reduce-scatter over the
    // head dim (dim 0 of the fused output) and keep this rank's own H/dcp
    // heads, avoiding the full all-gather of `out`.
    CHECK_EQ(heads % world_size, 0)
        << "DCP head-sharded merge requires heads divisible by dcp size";
    torch::Tensor head_major =
        torch::empty({heads, batch_size, head_dim}, out.options());
    kernel::mlu::dcp_correct_attn_transpose(
        out, lses, slot_mapping, dcp_group.rank(), head_major);
    const int64_t chunk_heads = head_major.size(0) / world_size;
    torch::Tensor reduced =
        torch::empty({chunk_heads, batch_size, head_dim}, out.options());
    dcp_group.reduce_scatter(head_major, reduced);
    return reduced.movedim(0, 1).contiguous().unsqueeze(/*dim=*/1);
  }
  kernel::mlu::dcp_correct_attn_out(out, lses, slot_mapping, dcp_group.rank());
  dcp_group.allreduce(out);
  return out.unsqueeze(/*dim=*/1);
}

}  // namespace xllm::layer
