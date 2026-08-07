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

#pragma once

#include <torch/torch.h>

#include "framework/kv_cache/kv_shard_layout.h"
#include "layers/mlu/dcp_attention_merge.h"
#include "layers/mlu/dsa_topk_state.h"

namespace xllm {

class ProcessGroup;

namespace layer {

class DcpDecodeContext final {
 public:
  DcpDecodeContext(KVShardLayout layout, ProcessGroup* dcp_group);

  torch::Tensor localize_slots(const torch::Tensor& global_slots) const;
  torch::Tensor expand_indexer_block_table(
      const torch::Tensor& logical_block_table) const;
  DsaTopkState localize_topk(const DsaTopkState& global_state) const;
  torch::Tensor gather_topk_cache(const torch::Tensor& global_slots,
                                  const torch::Tensor& local_cache) const;
  DcpAttentionResult merge(const torch::Tensor& local_output,
                           const torch::Tensor& local_lse) const;

 private:
  KVShardLayout layout_;
  ProcessGroup* dcp_group_ = nullptr;
};

}  // namespace layer
}  // namespace xllm
