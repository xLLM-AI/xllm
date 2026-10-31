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

#include <utility>

namespace xllm {

DraftProposal::DraftProposal(torch::Tensor token_ids, torch::Tensor logits)
    : DraftProposal(
          token_ids,
          logits.defined()
              ? std::unique_ptr<const DraftDistribution>(
                    std::make_unique<DenseDraftDistribution>(std::move(logits)))
              : std::unique_ptr<const DraftDistribution>(
                    std::make_unique<PointMassDraftDistribution>(token_ids))) {}

DraftProposal::DraftProposal(
    torch::Tensor token_ids,
    std::unique_ptr<const DraftDistribution> distribution)
    : token_ids_(std::move(token_ids)), distribution_(std::move(distribution)) {
  CHECK(distribution_ != nullptr);
  distribution_->validate(token_ids_);
}

}  // namespace xllm
