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
#include <torch/torch.h>

#include <limits>
#include <type_traits>
#include <utility>

#include "core/util/tensor_helper.h"

namespace xllm {
namespace {

void subtract_log_mass(torch::Tensor& log_q, const torch::Tensor& log_p) {
  log_q.sub_(log_p)
      .nan_to_num_(/*nan=*/0.0)
      .clamp_max_(0)
      .exp_()
      .neg_()
      .log1p_();
}

}  // namespace

DraftProposal::DraftProposal(torch::Tensor token_ids)
    : token_ids_(std::move(token_ids)) {
  validate();
}

DraftProposal::DraftProposal(torch::Tensor token_ids, torch::Tensor logits)
    : token_ids_(std::move(token_ids)) {
  if (logits.defined()) {
    representation_.emplace<torch::Tensor>(std::move(logits));
  }
  validate();
}

DraftProposal::DraftProposal(torch::Tensor token_ids,
                             torch::Tensor candidate_ids,
                             torch::Tensor logits)
    : token_ids_(std::move(token_ids)),
      representation_(SparseData{std::move(candidate_ids), std::move(logits)}) {
  validate();
}

void DraftProposal::validate() const {
  CHECK(token_ids_.defined());
  CHECK_EQ(token_ids_.dim(), 2);
  CHECK(token_ids_.scalar_type() == torch::kInt64 ||
        token_ids_.scalar_type() == torch::kInt32);
  const int64_t batch_size = token_ids_.size(0);
  const int64_t num_draft_tokens = token_ids_.size(1);
  const torch::Device device = token_ids_.device();
  std::visit(
      [&](const auto& data) {
        using T = std::decay_t<decltype(data)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          return;
        } else if constexpr (std::is_same_v<T, torch::Tensor> ||
                             std::is_same_v<T, SparseData>) {
          const torch::Tensor& logits = [&]() -> const torch::Tensor& {
            if constexpr (std::is_same_v<T, torch::Tensor>) {
              return data;
            } else {
              return data.logits;
            }
          }();
          CHECK(logits.defined());
          CHECK_EQ(logits.dim(), 3);
          CHECK(logits.is_floating_point());
          CHECK_GT(logits.size(-1), 0);
          CHECK_EQ(logits.size(0), batch_size);
          CHECK_EQ(logits.size(1), num_draft_tokens);
          CHECK_EQ(logits.device(), device);
          if constexpr (std::is_same_v<T, SparseData>) {
            CHECK(data.candidate_ids.defined());
            CHECK_EQ(data.candidate_ids.sizes(), logits.sizes());
            CHECK_EQ(data.candidate_ids.scalar_type(), torch::kInt64);
            CHECK_EQ(data.candidate_ids.device(), device);
          }
        } else {
          static_assert(std::is_same_v<T, void>,
                        "Unsupported draft distribution representation");
        }
      },
      representation_);
}

torch::Tensor DraftProposal::logsumexp() const {
  return std::visit(
      [](const auto& data) -> torch::Tensor {
        using T = std::decay_t<decltype(data)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          return {};
        } else if constexpr (std::is_same_v<T, torch::Tensor>) {
          return torch::logsumexp(data.to(torch::kFloat32), /*dim=*/-1);
        } else if constexpr (std::is_same_v<T, SparseData>) {
          return torch::logsumexp(data.logits.to(torch::kFloat32), /*dim=*/-1);
        } else {
          static_assert(std::is_same_v<T, void>,
                        "Unsupported draft distribution representation");
        }
      },
      representation_);
}

void DraftProposal::apply_acceptance(torch::Tensor& log_p,
                                     const torch::Tensor& token_ids,
                                     const torch::Tensor& lse) const {
  std::visit(
      [&](const auto& data) {
        using T = std::decay_t<decltype(data)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          // q(proposed token) = 1. The verifier rejects sentinels separately.
          return;
        } else if constexpr (std::is_same_v<T, torch::Tensor>) {
          log_p.sub_(index_select_2d(data, /*dim=*/-1, token_ids)
                         .to(torch::kFloat32)
                         .sub(lse));
        } else if constexpr (std::is_same_v<T, SparseData>) {
          const torch::Tensor matches =
              data.candidate_ids.eq(token_ids.unsqueeze(-1));
          const torch::Tensor selected =
              data.logits.to(torch::kFloat32)
                  .masked_fill(matches.logical_not(),
                               -std::numeric_limits<float>::infinity());
          log_p.sub_(torch::logsumexp(selected, /*dim=*/-1).sub(lse));
        } else {
          static_assert(std::is_same_v<T, void>,
                        "Unsupported draft distribution representation");
        }
      },
      representation_);
}

void DraftProposal::apply_residual(torch::Tensor& log_p,
                                   const torch::Tensor& steps,
                                   const torch::Tensor& rejected_tokens,
                                   const torch::Tensor& lse) const {
  std::visit(
      [&](const auto& data) {
        using T = std::decay_t<decltype(data)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          const torch::Tensor ids = rejected_tokens.clamp_min(0);
          const torch::Tensor remaining = log_p.gather(/*dim=*/-1, ids);
          remaining.masked_fill_(rejected_tokens.ge(0),
                                 -std::numeric_limits<float>::infinity());
          log_p.scatter_(/*dim=*/-1, ids, remaining);
        } else if constexpr (std::is_same_v<T, torch::Tensor>) {
          const torch::Tensor rows =
              steps.expand({log_p.size(0), data.size(-1)});
          torch::Tensor log_q = index_select_2d(data, /*dim=*/1, rows)
                                    .to(torch::kFloat32)
                                    .sub_(lse.gather(/*dim=*/1, steps));
          log_q.masked_fill_(rejected_tokens.lt(0),
                             -std::numeric_limits<float>::infinity());
          subtract_log_mass(log_q, log_p);
          log_p.add_(log_q);
        } else if constexpr (std::is_same_v<T, SparseData>) {
          const torch::Tensor rows =
              steps.expand({log_p.size(0), data.logits.size(-1)});
          const torch::Tensor ids =
              index_select_2d(data.candidate_ids, /*dim=*/1, rows);
          torch::Tensor log_q = index_select_2d(data.logits, /*dim=*/1, rows)
                                    .to(torch::kFloat32)
                                    .sub_(lse.gather(/*dim=*/1, steps));
          log_q.masked_fill_(rejected_tokens.lt(0),
                             -std::numeric_limits<float>::infinity());
          const torch::Tensor support_log_p = log_p.gather(/*dim=*/-1, ids);
          subtract_log_mass(log_q, support_log_p);
          log_q.add_(support_log_p);
          // Outside q's support the residual remains p.
          log_p.scatter_(/*dim=*/-1, ids, log_q);
        } else {
          static_assert(std::is_same_v<T, void>,
                        "Unsupported draft distribution representation");
        }
      },
      representation_);
}

}  // namespace xllm
