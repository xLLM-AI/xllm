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

  // Lifetime owner for no-sync draft inputs that must outlive the verify
  // step's stream sync; never read by the rejection sampler.
  std::vector<std::shared_ptr<const void>>& keep_alive() { return keep_alive_; }

  void validate(int64_t expected_batch_size,
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
    // The logits' vocab is the DRAFT model's own (an EAGLE3-style small head
    // differs from the target's), so it is intentionally NOT compared to the
    // target vocab: the rejection sampler only gathers at draft_token_ids,
    // which index the draft domain.
    CHECK_EQ(logits_->dim(), 3)
        << "proposal logits must be [batch, n_spec, draft_vocab]";
    CHECK_EQ(logits_->size(0), token_ids_.size(0))
        << "logits and token_ids must share the batch dim";
    CHECK_EQ(logits_->size(1), token_ids_.size(1))
        << "logits and token_ids must share the speculative length";
  }

  // Preallocates token_ids [batch, n_spec] plus the q buffer when
  // probabilistic; write via tokens_at()/logits_at().
  static DraftProposal preallocate(int64_t batch,
                                   int64_t num_speculative_tokens,
                                   int64_t vocab_size,
                                   bool is_probabilistic_draft,
                                   const torch::TensorOptions& token_options,
                                   const torch::TensorOptions& logits_options);

  static DraftProposal from_token_buffer(
      torch::Tensor token_ids,
      int64_t vocab_size,
      bool is_probabilistic_draft,
      const torch::TensorOptions& logits_options);

  torch::Tensor tokens_at(int64_t step) {
    return token_ids_.select(/*dim=*/1, /*index=*/step);
  }
  torch::Tensor logits_at(int64_t step) {
    CHECK(logits_.has_value()) << "greedy proposal carries no logits buffer";
    return logits_->select(/*dim=*/1, /*index=*/step);
  }

 private:
  torch::Tensor token_ids_;
  std::optional<torch::Tensor> logits_;
  std::vector<std::shared_ptr<const void>> keep_alive_;
};

}  // namespace xllm
