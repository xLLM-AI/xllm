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

#include "core/framework/speculative/draft_proposal.h"

#include <glog/logging.h>

namespace xllm {

DraftProposal DraftProposal::from_token_buffer(
    torch::Tensor token_ids,
    int64_t vocab_size,
    bool is_probabilistic_draft,
    const torch::TensorOptions& logits_options) {
  CHECK(token_ids.defined());
  CHECK_EQ(token_ids.dim(), 2);
  // Greedy drafts carry tokens only (no logits buffer), so callers pass
  // vocab_size = 0; the vocabulary bound is only meaningful when q is
  // reported.
  DraftProposal proposal(std::move(token_ids));
  if (is_probabilistic_draft) {
    CHECK_GT(vocab_size, 0);
    proposal.logits_ = torch::empty(
        {proposal.token_ids_.size(0), proposal.token_ids_.size(1), vocab_size},
        logits_options);
  }
  return proposal;
}

DraftProposal DraftProposal::preallocate(
    int64_t batch,
    int64_t num_speculative_tokens,
    int64_t vocab_size,
    bool is_probabilistic_draft,
    const torch::TensorOptions& token_options,
    const torch::TensorOptions& logits_options) {
  CHECK_GT(batch, 0);
  CHECK_GT(num_speculative_tokens, 0);
  return from_token_buffer(
      torch::empty({batch, num_speculative_tokens}, token_options),
      vocab_size,
      is_probabilistic_draft,
      logits_options);
}

}  // namespace xllm
