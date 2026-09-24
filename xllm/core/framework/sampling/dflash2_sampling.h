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

#include "core/framework/model/causal_lm.h"
#include "core/framework/sampling/sampling_params.h"

namespace xllm {

struct DFlash2SampleOutput {
  torch::Tensor token_ids;
  torch::Tensor dense_probs;
};

// Walk the trained selector path. Dense proposal probabilities preserve exact
// rejection recovery for random rows; greedy rows remain delta proposals.
DFlash2SampleOutput sample_dflash2_path(
    const DFlash2CandidateOutput& candidates,
    const SamplingParameters& sampling_params,
    const torch::Tensor& gumbel_noise,
    int64_t vocab_size,
    bool need_dense_probs);

}  // namespace xllm
