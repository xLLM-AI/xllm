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

#include "runtime/dflash_worker_impl.h"

namespace xllm {

class ProcessGroup;

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
  DraftBlock run_decode_draft(const ForwardInput& input,
                              ForwardInput& validate_input) override;

 private:
  struct BlockSampleOutput {
    torch::Tensor token_ids;
    torch::Tensor probs;
  };

  BlockSampleOutput sample_block(
      const torch::Tensor& base_logits,
      const torch::Tensor& anchor_token_ids,
      const SamplingParameters& sampling_params) const;

  void synchronize_sampled_token_ids(
      torch::Tensor& sampled_token_ids,
      const SamplingParameters& sampling_params) const;

  ProcessGroup* sampling_process_group_ = nullptr;
};

}  // namespace xllm
