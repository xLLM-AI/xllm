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

// Build the drop mask [B, W] (positions > prefix_len) on `device`.
torch::Tensor build_pruned_prefix_masks(
    const std::vector<int32_t>& pruned_prefix_lengths,
    int32_t num_speculative_tokens,
    const torch::Device& device);

// Apply per-seq pruning to rejection sampling output: mask positions beyond
// each seq's prefix_len to -1.
void apply_pruned_prefix_lengths(SampleOutput& sample_output,
                                 int32_t num_speculative_tokens,
                                 const torch::Tensor& drop_mask);

// Scatter a per-seq variable-length target output (produced by a pruned
// validate forward) back into the padded dense [batch * max_val_tokens, ...]
// layout the rejection sampler and the apply pruning helper expect.
// seq i's cu-packed rows [cu_offset, cu_offset + per_seq_val_tokens[i]) map
// 1:1 onto dense cols [0, per_seq_val_tokens[i]) of row i, so the per-seq
// bonus lands at col (per_seq_val_tokens[i] - 1), not the fixed last column.
// Verify inputs are logits-only, so only logits and embeddings are padded;
// logits pad with -inf (strictly-rejecting), embeddings with 0. Rewrites
// `target_output` in place; no-op when its rows already equal
// batch_size * max_val_tokens (uniform width).
void scatter_varlen_target_output_to_dense(
    ForwardOutput& target_output,
    const std::vector<int32_t>& per_seq_val_tokens,
    int32_t batch_size,
    int32_t max_val_tokens);

}  // namespace adaptive_pruning
}  // namespace xllm
