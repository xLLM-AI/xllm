/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/sampling/dspark_sampling.h"

#include <vector>

#include "core/framework/sampling/sampler.h"

namespace xllm {

DraftProposal sample_dspark_block(
    const torch::Tensor& base_logits,
    const torch::Tensor& anchor_token_ids,
    const SamplingParameters& sampling,
    const std::function<torch::Tensor(const torch::Tensor&)>& markov_bias,
    const std::function<void(torch::Tensor&)>& synchronize_tokens,
    torch::Tensor tokens) {
  CHECK_EQ(base_logits.dim(), 3);
  CHECK_EQ(anchor_token_ids.dim(), 1);
  CHECK_EQ(anchor_token_ids.size(0), base_logits.size(0));
  CHECK_EQ(anchor_token_ids.scalar_type(), torch::kInt64);
  CHECK_EQ(anchor_token_ids.device(), base_logits.device());
  const int64_t rows = base_logits.size(0);
  const int64_t steps = base_logits.size(1);
  const int64_t vocab = base_logits.size(2);
  CHECK_GT(steps, 0);
  if (!tokens.defined()) {
    tokens = torch::empty({rows, steps}, anchor_token_ids.options());
  } else {
    CHECK_EQ(tokens.sizes(), torch::IntArrayRef({rows, steps}));
    CHECK_EQ(tokens.scalar_type(), torch::kInt64);
    CHECK_EQ(tokens.device(), base_logits.device());
  }
  std::vector<torch::Tensor> draft_logits_steps;
  draft_logits_steps.reserve(steps);
  Sampler sampler;
  auto previous = anchor_token_ids;
  for (int64_t step = 0; step < steps; ++step) {
    auto bias = markov_bias(previous);
    CHECK_EQ(bias.sizes(), torch::IntArrayRef({rows, vocab}))
        << "DSpark reduced vocabulary requires draft-to-target remapping.";
    auto step_logits = base_logits.select(1, step) + bias;
    auto sampled = sampler.forward(step_logits, sampling);
    synchronize_tokens(sampled.next_tokens);
    tokens.select(1, step).copy_(sampled.next_tokens);
    previous = sampled.next_tokens;
    if (sampling.return_logits) {
      draft_logits_steps.emplace_back(std::move(sampled.draft_logits));
    }
  }
  return DraftProposal(std::move(tokens),
                       stack_draft_logits(draft_logits_steps));
}

}  // namespace xllm
