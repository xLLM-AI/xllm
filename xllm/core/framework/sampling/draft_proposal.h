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
#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <utility>

namespace xllm {

// Rejection-sampling input: drafted tokens, plus the draft distribution q for
// probabilistic drafts. Greedy drafts carry tokens only (accepted with q=1).
class DraftProposal final {
 public:
  DraftProposal() = default;

  explicit DraftProposal(
      torch::Tensor token_ids,
      std::optional<torch::Tensor> draft_probs = std::nullopt)
      : token_ids_(std::move(token_ids)),
        draft_probs_(std::move(draft_probs)) {}

  const torch::Tensor& token_ids() const { return token_ids_; }

  const std::optional<torch::Tensor>& draft_probs() const {
    return draft_probs_;
  }

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

    if (!draft_probs_.has_value()) {
      return;
    }
    CHECK(draft_probs_->defined())
        << "draft_probs must be defined when present";
    CHECK_EQ(draft_probs_->dim(), 3)
        << "draft_probs must be [batch, n_spec, vocab]";
    CHECK_EQ(draft_probs_->size(0), token_ids_.size(0))
        << "draft_probs and draft_token_ids must share the batch dim";
    CHECK_EQ(draft_probs_->size(1), token_ids_.size(1))
        << "draft_probs and draft_token_ids must share the speculative "
           "length";
    CHECK_EQ(draft_probs_->size(-1), expected_vocab_size)
        << "draft_probs and target_probs must share the vocab dim";
  }

  DraftProposal coerce_to(const torch::Tensor& reference_token_ids) const;

  DraftProposal slice_speculative(int64_t start, int64_t end) const;

 private:
  torch::Tensor token_ids_;
  std::optional<torch::Tensor> draft_probs_;
};

inline DraftProposal DraftProposal::coerce_to(
    const torch::Tensor& reference_token_ids) const {
  const auto device = reference_token_ids.device();
  return DraftProposal(token_ids_.to(reference_token_ids),
                       draft_probs_.has_value()
                           ? std::make_optional(draft_probs_->to(device))
                           : std::nullopt);
}

inline DraftProposal DraftProposal::slice_speculative(int64_t start,
                                                      int64_t end) const {
  using ISlice = torch::indexing::Slice;
  const auto slice_dim1 = [&](const torch::Tensor& tensor) {
    return tensor.index({ISlice(), ISlice(start, end)}).contiguous();
  };
  return DraftProposal(slice_dim1(token_ids_),
                       draft_probs_.has_value()
                           ? std::make_optional(slice_dim1(*draft_probs_))
                           : std::nullopt);
}

}  // namespace xllm
