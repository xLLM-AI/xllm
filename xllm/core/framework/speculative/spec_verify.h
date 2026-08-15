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

#include <torch/torch.h>

#include <cstdint>

namespace xllm {

struct SampleOutput;
struct ForwardOutput;

namespace spec_verify {

struct SamplerPolicy {
  torch::Tensor do_sample;
  bool all_random_sample = false;
  bool all_greedy_sample = false;
};

// Shared accept core for all speculative workers: build the RejectionSampler
// from the given policy, run it, reshape embeddings. Callers own
// bonus_token_ids and pass the policy explicitly so suffix's forced-greedy
// verify can reuse it.
SampleOutput run_rejection_sampling(const SamplerPolicy& policy,
                                    const torch::Tensor& draft_token_ids,
                                    const torch::Tensor& draft_probs,
                                    const ForwardOutput& target_output,
                                    const torch::Tensor& bonus_token_ids,
                                    int32_t batch_size,
                                    int32_t num_val_tokens,
                                    bool enable_fused_kernel);

}  // namespace spec_verify

}  // namespace xllm
