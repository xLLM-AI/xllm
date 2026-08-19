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

#include "mlu_ops_api.h"

namespace xllm::kernel::mlu {

torch::Tensor rejection_sample(const torch::Tensor& draft_token_ids,
                               const torch::Tensor& num_draft_tokens,
                               const torch::Tensor& cu_num_draft_tokens,
                               const std::optional<torch::Tensor>& draft_probs,
                               const torch::Tensor& target_probs,
                               const torch::Tensor& bonus_token_ids,
                               const torch::Tensor& uniform_rand,
                               const torch::Tensor& uniform_probs,
                               int64_t max_spec_len) {
  // Output shape should be 1D: number of draft tokens + number of bonus/recover
  // tokens
  auto output = torch::empty(
      {num_draft_tokens.size(0) + draft_token_ids.size(0)},
      torch::dtype(torch::kInt32).device(draft_token_ids.device()));
  // we only support high accuracy mode for now
  tmo::torch_api::rejection_sample(output,
                                   draft_token_ids,
                                   num_draft_tokens,
                                   cu_num_draft_tokens,
                                   draft_probs,
                                   target_probs,
                                   bonus_token_ids,
                                   uniform_rand,
                                   uniform_probs,
                                   max_spec_len,
                                   /*high_acc=*/true);
  return output;
}
}  // namespace xllm::kernel::mlu
