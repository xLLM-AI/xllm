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

#include "core/framework/speculative/draft_distribution.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <limits>
#include <utility>

#include "core/util/tensor_helper.h"

namespace xllm {
namespace {

void validate_logits(const torch::Tensor& logits) {
  CHECK_GT(logits.size(-1), 0);
  CHECK(logits.is_floating_point());
}

void subtract_log_mass(torch::Tensor& log_q, const torch::Tensor& log_p) {
  log_q.sub_(log_p)
      .nan_to_num_(/*nan=*/0.0)
      .clamp_max_(0)
      .exp_()
      .neg_()
      .log1p_();
}

}  // namespace

DraftDistribution::DraftDistribution(int64_t rank,
                                     const torch::Tensor& values) {
  CHECK(values.defined());
  CHECK_EQ(values.dim(), rank);
  batch_size_ = values.size(0);
  num_draft_tokens_ = values.size(1);
  device_ = values.device();
}

void DraftDistribution::validate(const torch::Tensor& token_ids) const {
  CHECK(token_ids.defined());
  CHECK_EQ(token_ids.dim(), 2);
  CHECK(token_ids.scalar_type() == torch::kInt64 ||
        token_ids.scalar_type() == torch::kInt32);
  CHECK_EQ(token_ids.size(0), batch_size_);
  CHECK_EQ(token_ids.size(1), num_draft_tokens_);
  CHECK_EQ(token_ids.device(), device_);
}

DraftDistribution::DraftDistribution(const torch::Tensor& token_ids)
    : DraftDistribution(/*rank=*/2, token_ids) {
  validate(token_ids);
}

DraftDistribution::DraftDistribution(const torch::Tensor& token_ids,
                                     torch::Tensor logits)
    : DraftDistribution(logits.defined() ? 3 : 2,
                        logits.defined() ? logits : token_ids) {
  validate(token_ids);
  if (logits.defined()) {
    representation_.emplace<Dense>(std::move(logits));
  }
}

DraftDistribution::DraftDistribution(const torch::Tensor& token_ids,
                                     torch::Tensor candidate_ids,
                                     torch::Tensor logits)
    : DraftDistribution(/*rank=*/3, logits) {
  validate(token_ids);
  representation_.emplace<Sparse>(std::move(candidate_ids), std::move(logits));
}

torch::Tensor DraftDistribution::logsumexp() const {
  return std::visit(
      [](const auto& distribution) { return distribution.logsumexp(); },
      representation_);
}

void DraftDistribution::apply_acceptance(torch::Tensor& log_p,
                                         const torch::Tensor& token_ids,
                                         const torch::Tensor& lse) const {
  std::visit(
      [&](const auto& distribution) {
        distribution.apply_acceptance(log_p, token_ids, lse);
      },
      representation_);
}

void DraftDistribution::apply_residual(torch::Tensor& log_p,
                                       const torch::Tensor& steps,
                                       const torch::Tensor& rejected_tokens,
                                       const torch::Tensor& lse) const {
  std::visit(
      [&](const auto& distribution) {
        distribution.apply_residual(log_p, steps, rejected_tokens, lse);
      },
      representation_);
}

torch::Tensor DraftDistribution::PointMass::logsumexp() const { return {}; }

void DraftDistribution::PointMass::apply_acceptance(
    torch::Tensor& /*log_p*/,
    const torch::Tensor& /*token_ids*/,
    const torch::Tensor& /*lse*/) const {
  // q(proposed token) = 1, so log(p/q) = log(p). Sentinel positions are
  // rejected by the verifier independently of this correction.
}

void DraftDistribution::PointMass::apply_residual(
    torch::Tensor& log_p,
    const torch::Tensor& /*steps*/,
    const torch::Tensor& rejected_tokens,
    const torch::Tensor& /*lse*/) const {
  const torch::Tensor ids = rejected_tokens.clamp_min(0);
  const torch::Tensor remaining = log_p.gather(/*dim=*/-1, ids);
  remaining.masked_fill_(rejected_tokens.ge(0),
                         -std::numeric_limits<float>::infinity());
  log_p.scatter_(/*dim=*/-1, ids, remaining);
}

DraftDistribution::Dense::Dense(torch::Tensor logits)
    : logits_(std::move(logits)) {
  validate_logits(logits_);
}

torch::Tensor DraftDistribution::Dense::logsumexp() const {
  return torch::logsumexp(logits_.to(torch::kFloat32), /*dim=*/-1);
}

void DraftDistribution::Dense::apply_acceptance(
    torch::Tensor& log_p,
    const torch::Tensor& token_ids,
    const torch::Tensor& lse) const {
  log_p.sub_(index_select_2d(logits_, /*dim=*/-1, token_ids)
                 .to(torch::kFloat32)
                 .sub(lse));
}

void DraftDistribution::Dense::apply_residual(
    torch::Tensor& log_p,
    const torch::Tensor& steps,
    const torch::Tensor& rejected_tokens,
    const torch::Tensor& lse) const {
  const torch::Tensor rows = steps.expand({log_p.size(0), logits_.size(-1)});
  torch::Tensor log_q = index_select_2d(logits_, /*dim=*/1, rows)
                            .to(torch::kFloat32)
                            .sub_(lse.gather(/*dim=*/1, steps));
  log_q.masked_fill_(rejected_tokens.lt(0),
                     -std::numeric_limits<float>::infinity());
  subtract_log_mass(log_q, log_p);
  log_p.add_(log_q);
}

DraftDistribution::Sparse::Sparse(torch::Tensor candidate_ids,
                                  torch::Tensor logits)
    : candidate_ids_(std::move(candidate_ids)), logits_(std::move(logits)) {
  validate_logits(logits_);
  CHECK(candidate_ids_.defined());
  CHECK_EQ(candidate_ids_.sizes(), logits_.sizes());
  CHECK_EQ(candidate_ids_.scalar_type(), torch::kInt64);
  CHECK_EQ(candidate_ids_.device(), logits_.device());
}

torch::Tensor DraftDistribution::Sparse::logsumexp() const {
  return torch::logsumexp(logits_.to(torch::kFloat32), /*dim=*/-1);
}

void DraftDistribution::Sparse::apply_acceptance(
    torch::Tensor& log_p,
    const torch::Tensor& token_ids,
    const torch::Tensor& lse) const {
  const torch::Tensor matches = candidate_ids_.eq(token_ids.unsqueeze(-1));
  const torch::Tensor selected =
      logits_.to(torch::kFloat32)
          .masked_fill(matches.logical_not(),
                       -std::numeric_limits<float>::infinity());
  log_p.sub_(torch::logsumexp(selected, /*dim=*/-1).sub(lse));
}

void DraftDistribution::Sparse::apply_residual(
    torch::Tensor& log_p,
    const torch::Tensor& steps,
    const torch::Tensor& rejected_tokens,
    const torch::Tensor& lse) const {
  const torch::Tensor rows = steps.expand({log_p.size(0), logits_.size(-1)});
  const torch::Tensor ids = index_select_2d(candidate_ids_, /*dim=*/1, rows);
  torch::Tensor log_q = index_select_2d(logits_, /*dim=*/1, rows)
                            .to(torch::kFloat32)
                            .sub_(lse.gather(/*dim=*/1, steps));
  log_q.masked_fill_(rejected_tokens.lt(0),
                     -std::numeric_limits<float>::infinity());
  const torch::Tensor support_log_p = log_p.gather(/*dim=*/-1, ids);
  subtract_log_mass(log_q, support_log_p);
  log_q.add_(support_log_p);
  // Only q support is corrected. Outside it the residual remains p.
  log_p.scatter_(/*dim=*/-1, ids, log_q);
}

}  // namespace xllm
