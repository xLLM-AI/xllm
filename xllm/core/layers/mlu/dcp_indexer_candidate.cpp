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

#include "layers/mlu/dcp_indexer_candidate.h"

#include <cstdint>
#include <utility>

#include "framework/parallel_state/process_group.h"

namespace xllm::layer {

DcpIndexerGatherAsyncCtx launch_dcp_indexer_candidate_gather(
    const torch::Tensor& local_scores,
    const torch::Tensor& local_global_slots,
    ProcessGroup* dcp_group) {
  if (dcp_group == nullptr || dcp_group->world_size() == 1) {
    DcpIndexerGatherAsyncCtx ctx;
    ctx.gathered_scores = local_scores.unsqueeze(0);
    ctx.gathered_global_slots = local_global_slots.unsqueeze(0);
    return ctx;
  }

  const int32_t world_size = dcp_group->world_size();
  std::vector<int64_t> gathered_shape = local_scores.sizes().vec();
  gathered_shape.insert(gathered_shape.begin(),
                        static_cast<int64_t>(world_size));
  DcpIndexerGatherAsyncCtx ctx;
  ctx.gathered_scores = torch::empty(gathered_shape, local_scores.options());
  ctx.gathered_global_slots =
      torch::empty(gathered_shape, local_global_slots.options());
  ctx.score_work =
      dcp_group->allgather_base_async(local_scores, ctx.gathered_scores);
  ctx.slot_work = dcp_group->allgather_base_async(local_global_slots,
                                                  ctx.gathered_global_slots);
  return ctx;
}

DcpIndexerGatheredCandidates finish_dcp_indexer_candidate_gather(
    DcpIndexerGatherAsyncCtx&& ctx) {
  if (ctx.score_work.defined()) {
    ctx.score_work->wait();
  }
  if (ctx.slot_work.defined()) {
    ctx.slot_work->wait();
  }
  return DcpIndexerGatheredCandidates{std::move(ctx.gathered_scores),
                                      std::move(ctx.gathered_global_slots)};
}

}  // namespace xllm::layer
