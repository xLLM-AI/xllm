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
#include <vector>

#include "framework/sampling/sampling_params.h"

namespace xllm {

class DraftProposal;
struct SampleOutput;
struct ForwardOutput;

struct SpeculativeOutputStats {
  std::vector<int64_t> accepted_per_position;
  std::vector<SpeculativeTokenStats> sequence_stats;
  int64_t committed_tokens = 0;
};

SpeculativeOutputStats calculate_speculative_output_stats(
    const torch::Tensor& tokens,
    int64_t num_speculative_tokens);

// Count the contiguous accepted prefix, excluding the one target replacement
// or bonus token and capping each row at its actual proposal width.
std::vector<SpeculativeTokenStats> calculate_contiguous_speculative_token_stats(
    const torch::Tensor& tokens,
    const std::vector<int32_t>& proposed_tokens);

// Install the validation sample output after algorithm-specific cache/state
// work.
ForwardOutput finalize_verify_output(ForwardOutput target_output,
                                     SampleOutput val_output);

namespace spec_verify {

struct SamplerPolicy {
  torch::Tensor do_sample;
  bool all_random_sample = false;
  bool all_greedy_sample = false;
};

SampleOutput run_rejection_sampling(const SamplerPolicy& policy,
                                    const DraftProposal& draft_proposal,
                                    const torch::Tensor& target_logits,
                                    const ForwardOutput& target_output,
                                    const torch::Tensor& bonus_token_ids,
                                    bool enable_fused_kernel);

}  // namespace spec_verify

}  // namespace xllm
