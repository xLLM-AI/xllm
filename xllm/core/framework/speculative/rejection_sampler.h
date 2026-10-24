/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <utility>

#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/draft_proposal.h"

namespace xllm {

// Params for a logits-only validate forward: transforms cleared (the
// RejectionSampler applies them itself from the per-sequence source), token
// idxes span the validate positions, do_sample repeats per speculative
// token.
inline void expand_sampling_params_for_verify(
    SamplingParameters& sampling_params,
    int64_t repeats,
    int64_t total_num_val_tokens,
    const torch::Device& device) {
  make_logits_only(sampling_params);
  sampling_params.selected_token_idxes =
      arange_indices(total_num_val_tokens, device);
  repeat_interleave_if_defined(sampling_params.do_sample, repeats);
}

class RejectionSampler final {
 public:
  // The request's sampling constraints, applied once inside forward().
  // Sink: moves the caller's params in.
  explicit RejectionSampler(SamplingParameters sampling_params)
      : sampling_params_(std::move(sampling_params)) {}

  // The single sampling owner for speculative verification. Consumes RAW
  // target logits flat [batch * (n_spec + 1), vocab] and applies the
  // sampling transforms itself on those rows. The general-purpose Sampler
  // never runs for verify steps.
  SampleOutput forward(const DraftProposal& draft_proposal,
                       torch::Tensor target_logits) const;


 private:
  // Greedy / random / mixed verification for a nonzero draft width.
  // `target_lse` matches the narrowed draft-token view of `target_logits`;
  // `target_greedy_token_ids` spans all verify tokens including the bonus.
  // `greedy_mask` is the [num_seqs, 1] greedy mask for the mixed legs (see
  // greedy_mask in sampling_params.h), built by forward().
  torch::Tensor verify(const DraftProposal& draft_proposal,
                       const torch::Tensor& target_logits,
                       const torch::Tensor& target_lse,
                       const torch::Tensor& bonus_token_ids,
                       const torch::Tensor& target_greedy_token_ids,
                       const torch::Tensor& greedy_mask) const;

  // Fills logprobs/top-k outputs from the sampled next_tokens.
  // `grouped_logits` is the [batch, val_tokens, vocab] grouped view (not
  // forward()'s flat raw input).
  SampleOutput fill_logprob_outputs(SampleOutput output,
                                    const torch::Tensor& grouped_logits) const;

  SamplingParameters sampling_params_;
};

}  // namespace xllm
