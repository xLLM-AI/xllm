/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/runtime/spec_runtime_helpers.h"

#include <torch/torch.h>

#include "core/framework/parallel_state/process_group.h"

namespace xllm::spec_detail {

// DeepSeek-V4 TORCH publishes orthogonal TP and CP groups; other backends
// retain single-group speculative broadcast behavior.
bool uses_orthogonal_cp_consensus(const ParallelArgs& parallel_args) {
  return parallel_args.cp_size() > 1 && parallel_args.tp_group_ != nullptr &&
         parallel_args.cp_group_ != nullptr &&
         parallel_args.cp_group_ != parallel_args.tp_group_;
}

void broadcast_consensus_tokens(torch::Tensor& tokens,
                                const ParallelArgs& parallel_args) {
  if (!uses_orthogonal_cp_consensus(parallel_args)) {
    sync_sampling_tensor(tokens, spec_sampling_group(parallel_args));
    return;
  }

  sync_sampling_tensor(tokens, parallel_args.tp_group_);
  sync_sampling_tensor(tokens, parallel_args.cp_group_);
}

}  // namespace xllm::spec_detail
