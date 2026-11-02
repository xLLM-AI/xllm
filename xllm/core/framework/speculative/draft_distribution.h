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
#include <variant>

namespace xllm {

// Verification operations for the draft distribution q. Producers finish
// populating proposal tokens and distribution tensors before publishing them.
class DraftDistribution final {
 public:
  explicit DraftDistribution(const torch::Tensor& token_ids);
  // Undefined logits represent a point mass; otherwise logits are dense.
  DraftDistribution(const torch::Tensor& token_ids, torch::Tensor logits);
  // Candidate IDs are unique within each row and index the target vocabulary.
  DraftDistribution(const torch::Tensor& token_ids,
                    torch::Tensor candidate_ids,
                    torch::Tensor logits);

  DraftDistribution(const DraftDistribution&) = delete;
  DraftDistribution& operator=(const DraftDistribution&) = delete;
  DraftDistribution(DraftDistribution&&) = default;
  DraftDistribution& operator=(DraftDistribution&&) = default;

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

  // Check the token/distribution pairing when a proposal is built.
  void validate(const torch::Tensor& token_ids) const;

 private:
  class PointMass final {
   public:
    torch::Tensor logsumexp() const;
    void apply_acceptance(torch::Tensor& log_p,
                          const torch::Tensor& token_ids,
                          const torch::Tensor& lse) const;
    void apply_residual(torch::Tensor& log_p,
                        const torch::Tensor& steps,
                        const torch::Tensor& rejected_tokens,
                        const torch::Tensor& lse) const;
  };
  class Dense final {
   public:
    explicit Dense(torch::Tensor logits);
    torch::Tensor logsumexp() const;
    void apply_acceptance(torch::Tensor& log_p,
                          const torch::Tensor& token_ids,
                          const torch::Tensor& lse) const;
    void apply_residual(torch::Tensor& log_p,
                        const torch::Tensor& steps,
                        const torch::Tensor& rejected_tokens,
                        const torch::Tensor& lse) const;

   private:
    torch::Tensor logits_;
  };
  class Sparse final {
   public:
    Sparse(torch::Tensor candidate_ids, torch::Tensor logits);
    torch::Tensor logsumexp() const;
    void apply_acceptance(torch::Tensor& log_p,
                          const torch::Tensor& token_ids,
                          const torch::Tensor& lse) const;
    void apply_residual(torch::Tensor& log_p,
                        const torch::Tensor& steps,
                        const torch::Tensor& rejected_tokens,
                        const torch::Tensor& lse) const;

   private:
    torch::Tensor candidate_ids_;
    torch::Tensor logits_;
  };

  DraftDistribution(int64_t rank, const torch::Tensor& values);

  int64_t batch_size_;
  int64_t num_draft_tokens_;
  torch::Device device_{torch::kCPU};
  std::variant<PointMass, Dense, Sparse> representation_;
};

}  // namespace xllm
