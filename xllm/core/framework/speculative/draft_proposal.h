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
#include <vector>

#include "core/framework/speculative/draft_distribution.h"

namespace xllm {

// Completed draft tokens and their distribution q. The proposal owns q and
// retained asynchronous inputs through verification; consumers cannot replace
// its tokens or access representation-specific buffers.
class DraftProposal final {
 public:
  explicit DraftProposal(torch::Tensor token_ids, torch::Tensor logits = {});
  DraftProposal(torch::Tensor token_ids,
                std::unique_ptr<const DraftDistribution> distribution);

  DraftProposal(const DraftProposal&) = delete;
  DraftProposal& operator=(const DraftProposal&) = delete;
  DraftProposal(DraftProposal&&) = default;
  DraftProposal& operator=(DraftProposal&&) = default;

  const torch::Tensor& token_ids() const { return token_ids_; }
  const DraftDistribution& distribution() const { return *distribution_; }

  // No-sync draft inputs must outlive the verify step's stream sync.
  std::vector<std::shared_ptr<const void>>& keep_alive() { return keep_alive_; }

 private:
  torch::Tensor token_ids_;
  std::unique_ptr<const DraftDistribution> distribution_;
  std::vector<std::shared_ptr<const void>> keep_alive_;
};

}  // namespace xllm
