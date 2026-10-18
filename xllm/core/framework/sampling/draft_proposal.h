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

#include <glog/logging.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace xllm {

// Rejection-sampling input: drafted tokens plus the draft distribution q
// for probabilistic drafts (greedy drafts carry tokens only, q = 1). -1
// sentinels in token_ids are force-rejected.
class DraftProposal final {
 public:
  DraftProposal() = default;

  explicit DraftProposal(torch::Tensor token_ids,
                         std::optional<torch::Tensor> logits = std::nullopt)
      : token_ids_(std::move(token_ids)),
        logits_(logits.has_value() && logits->defined() ? std::move(logits)
                                                        : std::nullopt) {}

  const torch::Tensor& token_ids() const { return token_ids_; }

  const std::optional<torch::Tensor>& logits() const { return logits_; }

  void validate(int64_t expected_batch_size,
                int64_t expected_vocab_size,
                int64_t expected_num_speculative_tokens) const {
    CHECK(token_ids_.defined()) << "draft proposal token_ids must be defined";
    CHECK_EQ(token_ids_.dim(), 2)
        << "draft proposal token_ids must be [batch, n_spec]";
    CHECK_EQ(token_ids_.size(0), expected_batch_size)
        << "draft proposal batch size mismatch";
    CHECK_EQ(token_ids_.size(1), expected_num_speculative_tokens)
        << "draft proposal speculative length mismatch";

    if (!logits_.has_value()) {
      return;
    }
    CHECK_EQ(logits_->dim(), 3)
        << "proposal logits must be [batch, n_spec, vocab]";
    CHECK_EQ(logits_->size(0), token_ids_.size(0))
        << "logits and token_ids must share the batch dim";
    CHECK_EQ(logits_->size(1), token_ids_.size(1))
        << "logits and token_ids must share the speculative length";
    CHECK_EQ(logits_->size(-1), expected_vocab_size)
        << "proposal logits and target logits must share the vocab dim";
  }

  DraftProposal slice_speculative(int64_t start, int64_t end) const;

 private:
  torch::Tensor token_ids_;
  std::optional<torch::Tensor> logits_;
};

inline DraftProposal DraftProposal::slice_speculative(int64_t start,
                                                      int64_t end) const {
  using ISlice = torch::indexing::Slice;
  const auto slice_dim1 = [&](const torch::Tensor& tensor) {
    return tensor.index({ISlice(), ISlice(start, end)}).contiguous();
  };
  return DraftProposal(slice_dim1(token_ids_),
                       logits_.has_value()
                           ? std::make_optional(slice_dim1(*logits_))
                           : std::nullopt);
}

// Stacks per-step draft logits; nullopt when no step carried logits.
inline std::optional<torch::Tensor> stack_draft_logits(
    const std::vector<torch::Tensor>& draft_logits_steps) {
  if (draft_logits_steps.empty() || !draft_logits_steps.front().defined()) {
    return std::nullopt;
  }
  CHECK(std::all_of(draft_logits_steps.begin(),
                    draft_logits_steps.end(),
                    [](const torch::Tensor& step) { return step.defined(); }))
      << "draft logits must be defined for every step";
  return torch::stack(draft_logits_steps, /*dim=*/1);
}

inline DraftProposal stack_draft_proposal(
    const std::vector<torch::Tensor>& draft_token_ids_steps,
    const std::vector<torch::Tensor>& draft_logits_steps) {
  CHECK(!draft_token_ids_steps.empty()) << "draft steps must not be empty";
  return DraftProposal(
      torch::stack(draft_token_ids_steps, /*dim=*/1).to(torch::kLong),
      stack_draft_logits(draft_logits_steps));
}

}  // namespace xllm
