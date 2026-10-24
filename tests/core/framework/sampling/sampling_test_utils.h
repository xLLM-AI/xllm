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

#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/rejection_sampler.h"
#include "core/util/tensor_helper.h"
#include "platform/platform.h"

namespace xllm {
namespace testutil {

// Tensor form of the SamplingMode derivation; an undefined do_sample maps
// to GREEDY (the greedy-batch contract).
inline SamplingMode sampling_mode_from_do_sample(
    const torch::Tensor& do_sample) {
  if (!do_sample.defined()) {
    return SamplingMode::GREEDY;
  }
  const bool all_random = do_sample.all().item<bool>();
  const bool any_random = do_sample.any().item<bool>();
  return all_random ? SamplingMode::RANDOM
                    : (any_random ? SamplingMode::MIXED : SamplingMode::GREEDY);
}

inline SamplingParameters make_sampling_params(const torch::Tensor& do_sample) {
  SamplingParameters params;
  params.do_sample = do_sample;
  if (do_sample.defined()) {
    params.selected_token_idxes =
        arange_indices(do_sample.size(0), torch::Device(torch::kCPU));
  }
  params.sampling_mode = sampling_mode_from_do_sample(do_sample);
  return params;
}

// All-greedy batch; do_sample stays undefined as in production.
inline SamplingParameters make_sampling_params(int64_t batch_size) {
  SamplingParameters params;
  params.selected_token_idxes =
      arange_indices(batch_size, torch::Device(torch::kCPU));
  return params;
}

// Uniform-mode batch (greedy or random); MIXED needs do_sample, so use the
// tensor overload above.
inline SamplingParameters make_sampling_params(int64_t batch_size,
                                               SamplingMode mode) {
  SamplingParameters params = make_sampling_params(batch_size);
  params.sampling_mode = mode;
  return params;
}

inline SamplingParameters make_rejection_params(const torch::Tensor& do_sample,
                                                bool logprobs = false,
                                                int64_t max_top_logprobs = 0) {
  SamplingParameters params = make_sampling_params(do_sample);
  params.logprobs = logprobs;
  params.max_top_logprobs = max_top_logprobs;
  return params;
}

inline SamplingParameters make_rejection_params(bool logprobs = false,
                                                int64_t max_top_logprobs = 0) {
  return make_rejection_params(torch::Tensor(), logprobs, max_top_logprobs);
}

// Flattens [..., vocab] logits to the flat [rows, vocab] layout
// rejection_sample consumes.
inline torch::Tensor flatten_rows(const torch::Tensor& logits) {
  return logits.view({-1, logits.size(-1)});
}

// MLU/CUDA when available, otherwise CPU.
inline torch::Device get_test_device() {
  if ((Platform::is_mlu() || Platform::is_cuda()) &&
      Platform::device_count() > 0) {
    return torch::Device(Platform::type_torch(), 0);
  }
  return torch::Device(torch::kCPU);
}

inline torch::TensorOptions get_test_options(
    torch::ScalarType dtype = torch::kFloat32) {
  return torch::dtype(dtype).device(get_test_device());
}

// Drives random rejection_sample() over normalized
// per-position probs. Without an explicit bonus_row, the bonus row repeats
// the last target row's distribution. The seed is for reproducibility only —
// deterministic tests construct p/q in {0, 1} so the outcome is
// draw-independent.
inline torch::Tensor random_sample_from_probs(
    const DraftProposal& proposal,
    const torch::Tensor& target_probs,
    int64_t uniform_seed,
    const torch::Tensor& bonus_row = torch::Tensor()) {
  const int64_t batch_size = target_probs.size(0);
  const torch::Tensor bonus = bonus_row.defined()
                                  ? bonus_row
                                  : target_probs.select(/*dim=*/1, /*index=*/-1)
                                        .log()
                                        .unsqueeze(/*dim=*/1);
  const torch::Tensor target_logits =
      torch::cat({target_probs.log(), bonus}, /*dim=*/1);
  const SamplingParameters params =
      make_rejection_params(torch::ones({batch_size}, torch::kBool));
  torch::manual_seed(uniform_seed);
  SampleOutput output =
      rejection_sample(params, proposal, flatten_rows(target_logits));
  return output.next_tokens;
}

}  // namespace testutil
}  // namespace xllm
