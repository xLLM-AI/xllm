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

#include "core/framework/speculative/draft_proposal.h"
#include "runtime/dflash_worker_impl.h"

namespace xllm {

class ProcessGroup;

// Gumbel-noise walk of the DFlash2 candidate tree; undefined noise (greedy
// draft) degrades to the raw argmax. Shared with the task pipeline; TP
// noise consensus is the caller's.
DraftProposal sample_dflash2_path(
    const DFlash2CandidateOutput& candidates,
    const SamplingParameters& sampling_params,
    torch::Tensor gumbel_noise,
    int64_t vocab_size,
    torch::Tensor token_ids_out = torch::Tensor());
// Selector noise for the DFlash2 candidate walk; only a deterministic draft
// (SamplingMode::GREEDY) returns an undefined tensor — callers key
// is_probabilistic_draft off that. A MIXED batch still draws noise (the
// 0.0 greedy-row sentinel keeps its greedy rows). With a non-null
// sampling_group the noise is broadcast from rank 0 so every rank walks the
// same candidates. batch_size is the row count the noise must match — the
// caller's own geometry, not derived from the sampling params.
torch::Tensor broadcast_gumbel_noise(int64_t batch_size,
                                     const SamplingParameters& sampling,
                                     int64_t num_steps,
                                     int64_t num_classes,
                                     const torch::Device& device,
                                     ProcessGroup* sampling_group);

class DFlash2WorkerImpl final : public DFlashWorkerImpl {
 public:
  DFlash2WorkerImpl(const ParallelArgs& parallel_args,
                    const torch::Device& device,
                    const runtime::Options& options);

  ~DFlash2WorkerImpl() override = default;

 protected:
  DraftProposal run_decode_draft(const LlmForwardInput& input,
                                 LlmForwardInput& validate_input) override;
};

}  // namespace xllm
