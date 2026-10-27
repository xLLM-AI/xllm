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

#pragma once

#include <torch/types.h>

#include "core/kv_cache/layout/kv_shard_layout.h"
#include "layers/mlu/dcp_indexer_candidate.h"
#include "layers/mlu/dsa_topk_state.h"

namespace xllm {

class ProcessGroup;

namespace layer {

class DcpDecodeContext final {
 public:
  DcpDecodeContext(KVShardLayout layout, ProcessGroup* dcp_group);

  torch::Tensor localize_slots(const torch::Tensor& global_slots) const;
  DsaTopkState localize_topk(const DsaTopkState& global_state) const;
  // Defer the wait to overlap candidate exchange with independent projections.
  DcpIndexerGatherAsyncCtx launch_indexer_candidate_gather(
      const torch::Tensor& local_scores,
      const torch::Tensor& local_global_slots) const;
  DsaTopkState finish_indexer_candidate_merge(
      DcpIndexerGatherAsyncCtx&& gather,
      int64_t topk,
      const torch::Tensor& slot_mapping) const;
  torch::Tensor merge(const torch::Tensor& local_output,
                      const torch::Tensor& local_lse,
                      const torch::Tensor& slot_mapping,
                      bool head_sharded) const;

  // DCP group size (1 when no DCP group is configured).
  int32_t world_size() const;
  const KVShardLayout& layout() const { return layout_; }

 private:
  KVShardLayout layout_;
  ProcessGroup* dcp_group_ = nullptr;
};

}  // namespace layer
}  // namespace xllm
