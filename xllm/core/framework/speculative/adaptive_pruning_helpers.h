/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "runtime/forward_params.h"
#include "util/tensor_helper.h"

namespace xllm {
namespace adaptive_pruning {

// Extract per-step selected token probabilities from draft outputs into
// a [batch, num_speculative_tokens] matrix for the pruning controller.
torch::Tensor selected_probs_by_step(
    const std::vector<ForwardOutput>& draft_outputs);

// Check whether all draft outputs have defined probs (required for pruning).
bool has_selected_probs_by_step(
    const std::vector<ForwardOutput>& draft_outputs);

// Clamp pruning prefix lengths to valid range [0, num_speculative_tokens].
void clamp_prefix_lengths(std::vector<int32_t>& prefix_lengths,
                          int32_t batch_size,
                          int32_t num_speculative_tokens);

// Get the maximum prefix length in the batch (determines padded validate
// width).
int32_t max_pruned_prefix_length(const std::vector<int32_t>& prefix_lengths,
                                 int32_t num_speculative_tokens);

// Truncate draft outputs to only the first num_speculative_tokens entries.
std::vector<ForwardOutput> truncate_draft_outputs(
    const std::vector<ForwardOutput>& draft_outputs,
    int32_t num_speculative_tokens);

// Apply per-seq pruning to rejection sampling output: mask positions beyond
// each seq's prefix_len to -1, and replace the boundary position with the
// target model's token (acting as bonus token for the truncated sequence).
void apply_pruned_prefix_lengths(
    SampleOutput& sample_output,
    const torch::Tensor& target_next_tokens,
    int32_t num_speculative_tokens,
    const std::vector<int32_t>& pruned_prefix_lengths);

// Correct logprobs/top_logprobs at pruning boundaries: replace the boundary
// position's logprob with target model's logprob (since the token changed from
// a potentially-accepted draft token to a target-resampled token).
void sync_pruned_boundary_outputs(
    SampleOutput& sample_output,
    const ForwardOutput& target_output,
    int32_t batch_size,
    int32_t num_val_tokens,
    int32_t num_speculative_tokens,
    const std::vector<int32_t>& pruned_prefix_lengths);

}  // namespace adaptive_pruning
}  // namespace xllm
