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

#pragma once

#include <functional>

#include "core/framework/sampling/draft_proposal.h"
#include "core/framework/sampling/sampling_params.h"

namespace xllm {

// Consensus must complete before feeding a sampled token to the next step.
DraftProposal sample_dspark_block(
    const torch::Tensor& base_logits,
    const torch::Tensor& anchor_token_ids,
    SamplingParameters sampling,
    const std::function<torch::Tensor(const torch::Tensor&)>& markov_bias,
    const std::function<void(torch::Tensor&)>& synchronize_tokens,
    torch::Tensor tokens = {});

}  // namespace xllm
