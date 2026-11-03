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

#include <memory>
#include <variant>
#include <vector>

namespace xllm {

// Completed draft tokens and their distribution q. The proposal owns q and
// retained asynchronous inputs through verification; consumers cannot replace
// its tokens or access representation-specific buffers. Verification tokens may
// contain -1 sentinels; dense/sparse q retains the original draft distribution.
class DraftProposal final {
 public:
  explicit DraftProposal(torch::Tensor token_ids);
  // Undefined logits represent a point mass; otherwise logits are dense.
  DraftProposal(torch::Tensor token_ids, torch::Tensor logits);
  // Candidate IDs are unique within each row and index the target vocabulary.
  DraftProposal(torch::Tensor token_ids,
                torch::Tensor candidate_ids,
                torch::Tensor logits);

  DraftProposal(const DraftProposal&) = delete;
  DraftProposal& operator=(const DraftProposal&) = delete;
  DraftProposal(DraftProposal&&) = default;
  DraftProposal& operator=(DraftProposal&&) = default;

  const torch::Tensor& token_ids() const { return token_ids_; }
  // FP32 [batch, draft tokens] normalizers; undefined for a point mass.
  torch::Tensor logsumexp() const;
  // Replace log_p with log(p/q) at nonnegative proposal gather indices.
  // The verifier separately rejects negative sentinel positions.
  void apply_acceptance(torch::Tensor& log_p,
                        const torch::Tensor& token_ids,
                        const torch::Tensor& lse) const;
  // Replace log_p with log((p - q)+) at the selected steps.
  // Negative rejected tokens disable q, recovering directly from p.
  void apply_residual(torch::Tensor& log_p,
                      const torch::Tensor& steps,
                      const torch::Tensor& rejected_tokens,
                      const torch::Tensor& lse) const;

  // No-sync draft inputs must outlive the verify step's stream sync.
  std::vector<std::shared_ptr<const void>>& keep_alive() { return keep_alive_; }

 private:
  struct SparseData {
    torch::Tensor candidate_ids;
    torch::Tensor logits;
  };
  // No data for point masses; a single logits tensor for dense distributions.
  using Representation =
      std::variant<std::monostate, torch::Tensor, SparseData>;

  void validate() const;

  torch::Tensor token_ids_;
  Representation representation_;
  std::vector<std::shared_ptr<const void>> keep_alive_;
};

}  // namespace xllm
