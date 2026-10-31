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

namespace xllm {

// Verification operations for the draft distribution q. Producers finish
// populating proposal tokens and distribution tensors before publishing them.
class DraftDistribution {
 public:
  virtual ~DraftDistribution() = default;

  // FP32 [batch, steps] normalizers; undefined for a point mass.
  virtual torch::Tensor logsumexp() const = 0;
  // Replace log_p [batch, steps] with log(p/q) at the proposed tokens.
  // token_ids are their nonnegative gather indices; the verifier separately
  // rejects negative sentinel positions. This is a proposal operation, not
  // an arbitrary-token probability query.
  virtual void apply_acceptance(torch::Tensor& log_p,
                                const torch::Tensor& token_ids,
                                const torch::Tensor& lse) const = 0;
  // Replace log_p [batch, vocab] with log((p - q)+) at the selected steps.
  // Negative rejected tokens disable q, recovering directly from p.
  virtual void apply_residual(torch::Tensor& log_p,
                              const torch::Tensor& steps,
                              const torch::Tensor& rejected_tokens,
                              const torch::Tensor& lse) const = 0;

  // Metadata-only compatibility check, performed when a proposal is built.
  void validate(const torch::Tensor& token_ids) const;

 protected:
  DraftDistribution(const torch::Tensor& values, int64_t rank);

 private:
  int64_t batch_size_;
  int64_t num_steps_;
  torch::Device device_{torch::kCPU};
};

class PointMassDraftDistribution final : public DraftDistribution {
 public:
  // q is concentrated at each proposal token. The proposal owns those tokens;
  // this representation needs only their metadata, with no probability buffer.
  explicit PointMassDraftDistribution(const torch::Tensor& token_ids);

  torch::Tensor logsumexp() const override;
  void apply_acceptance(torch::Tensor& log_p,
                        const torch::Tensor& token_ids,
                        const torch::Tensor& lse) const override;
  void apply_residual(torch::Tensor& log_p,
                      const torch::Tensor& steps,
                      const torch::Tensor& rejected_tokens,
                      const torch::Tensor& lse) const override;
};

class DenseDraftDistribution final : public DraftDistribution {
 public:
  explicit DenseDraftDistribution(torch::Tensor logits);

  torch::Tensor logsumexp() const override;
  void apply_acceptance(torch::Tensor& log_p,
                        const torch::Tensor& token_ids,
                        const torch::Tensor& lse) const override;
  void apply_residual(torch::Tensor& log_p,
                      const torch::Tensor& steps,
                      const torch::Tensor& rejected_tokens,
                      const torch::Tensor& lse) const override;

 private:
  torch::Tensor logits_;
};

class SparseDraftDistribution final : public DraftDistribution {
 public:
  // IDs are unique within each support row and index the target vocabulary.
  SparseDraftDistribution(torch::Tensor candidate_ids, torch::Tensor logits);

  torch::Tensor logsumexp() const override;
  void apply_acceptance(torch::Tensor& log_p,
                        const torch::Tensor& token_ids,
                        const torch::Tensor& lse) const override;
  void apply_residual(torch::Tensor& log_p,
                      const torch::Tensor& steps,
                      const torch::Tensor& rejected_tokens,
                      const torch::Tensor& lse) const override;

 private:
  torch::Tensor candidate_ids_;
  torch::Tensor logits_;
};

}  // namespace xllm
