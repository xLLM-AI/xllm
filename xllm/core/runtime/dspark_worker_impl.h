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

#include <functional>

#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/draft_proposal.h"
#include "runtime/dflash_worker_impl.h"

namespace xllm {

class ProcessGroup;

// The DSpark Markov head: [rows] tokens in, [rows, vocab] bias out. Both
// callers hold the draft model privately, so the head enters as a callable.
using DSparkMarkovBias =
    std::function<torch::Tensor(const torch::Tensor& previous_token_ids)>;

// The sampled tokens must agree across TP ranks before the next step;
// greedy argmax is rank-local, so the walker never invokes this on greedy
// batches. The caller owns which process groups to broadcast over.
using SyncSampledTokens = std::function<void(torch::Tensor& sampled_tokens)>;

// Walks the block logits [rows, steps, vocab] step by step through
// `markov_bias` and the shared sample(); `anchors` seed the first step.
// `token_ids_out` borrows a preallocated [rows, steps] buffer (undefined
// allocates). Lives next to its only model-coupled consumer, not in
// framework/.
DraftProposal sample_dspark_markov_block(
    const DSparkMarkovBias& markov_bias,
    const torch::Tensor& base_logits,
    const torch::Tensor& anchors,
    const SamplingParameters& sampling,
    const SyncSampledTokens& sync_tokens,
    bool is_probabilistic_draft,
    torch::Tensor token_ids_out = torch::Tensor());

// DSpark = DFlash block-diffusion draft + a low-rank Markov head that adds a
// prefix-dependent bias to each block position's draft logits. It reuses the
// entire DFlash pipeline (prefill, context-K/V injection, validate, cache
// writes) and overrides only:
//   1. sample_from_anchor() — returns true, so the shared query builder makes
//      an N-wide block (every position predicts, anchor predicts the first
//      draft token) instead of DFlash's (1 + N) anchor + masks.
//   2. run_decode_draft — performs the logits-only block forward and sequential
//      Markov sampling before returning proposals to the shared validator.
class DSparkWorkerImpl final : public DFlashWorkerImpl {
 public:
  DSparkWorkerImpl(const ParallelArgs& parallel_args,
                   const torch::Device& device,
                   const runtime::Options& options);

  ~DSparkWorkerImpl() override = default;

 protected:
  // N-wide query block: anchor (slot 0) carries the last real token and
  // predicts the first draft token; every position is a prediction. DFlash's
  // prepare_query_inputs reads this hook to pick the block layout.
  bool sample_from_anchor() const override { return true; }

  // Build the DSpark query and produce the complete proposal block.
  DraftProposal run_decode_draft(const LlmForwardInput& input,
                                 LlmForwardInput& validate_input) override;
};

}  // namespace xllm
