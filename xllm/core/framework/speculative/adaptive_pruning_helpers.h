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

// Precomputed device-side masks derived from the per-seq pruning prefix
// lengths. Built once per validate() so the three pruning helpers do not
// each re-upload prefix_lengths and re-run arange+eq+logical_and.
//
// Shapes (with batch=B, num_val_tokens=W):
//   keep_mask [B, W] — positions <= prefix_len (token kept).
//   cut_mask  [B, W] — position == prefix_len && prefix_len < num_spec
//                       (boundary position that switches to target token).
struct PrunedPrefixMasks {
  torch::Tensor keep_mask;
  torch::Tensor cut_mask;
};

// Build keep_mask / cut_mask on `device`. Helpers below can then re-target
// them to per-tensor devices with .to(...), which is a no-op if they match.
PrunedPrefixMasks build_pruned_prefix_masks(
    const std::vector<int32_t>& pruned_prefix_lengths,
    int32_t num_speculative_tokens,
    const torch::Device& device);

// Apply per-seq pruning to rejection sampling output: mask positions beyond
// each seq's prefix_len to -1, and replace the boundary position with the
// target model's token (acting as bonus token for the truncated sequence).
void apply_pruned_prefix_lengths(SampleOutput& sample_output,
                                 const torch::Tensor& target_next_tokens,
                                 int32_t num_speculative_tokens,
                                 const PrunedPrefixMasks& masks);

// Correct logprobs/top_logprobs at pruning boundaries: replace the boundary
// position's logprob with target model's logprob (since the token changed from
// a potentially-accepted draft token to a target-resampled token).
void sync_pruned_boundary_outputs(SampleOutput& sample_output,
                                  const ForwardOutput& target_output,
                                  int32_t batch_size,
                                  int32_t num_val_tokens,
                                  const PrunedPrefixMasks& masks);

// Scatter a per-seq variable-length target output (produced by a pruned
// validate forward) back into the padded dense [batch * max_val_tokens, ...]
// layout the rejection sampler and the sync/apply pruning helpers expect.
//
// seq i's cu-packed rows [cu_offset, cu_offset + per_seq_val_tokens[i]) map
// 1:1 onto dense cols [0, per_seq_val_tokens[i]) of row i, so the per-seq
// bonus lands at col (per_seq_val_tokens[i] - 1), not the fixed last column.
// Every populated field of `target_output` (logits, next_tokens, embeddings,
// logprobs, top_tokens, top_logprobs) is padded in lockstep; logits pad with
// -inf (strictly-rejecting), the rest with 0 except next_tokens which pads
// with `next_token_pad_value`. Rewrites `target_output` in place; no-op when
// its rows already equal batch_size * max_val_tokens (uniform width).
//
// next_token_pad_value differs by caller: DFlash uses -1 (a padded slot read
// at a per-seq cut position must not surface a real token id — Qwen id 0 is
// "!"); MTP passes 0 to preserve its established output. Both are masked to
// -1 by apply_pruned_prefix_lengths at trailing positions downstream.
void scatter_varlen_target_output_to_dense(
    ForwardOutput& target_output,
    const std::vector<int32_t>& per_seq_val_tokens,
    int32_t batch_size,
    int32_t max_val_tokens,
    int64_t next_token_pad_value);

}  // namespace adaptive_pruning
}  // namespace xllm
