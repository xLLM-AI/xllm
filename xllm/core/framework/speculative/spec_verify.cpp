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

#include "core/framework/speculative/spec_verify.h"

#include <memory>

#include "framework/sampling/rejection_sampler.h"
#include "framework/sampling/sampling_params.h"
#include "runtime/forward_params.h"

namespace xllm::spec_verify {

SampleOutput run_rejection_sampling(const SamplerPolicy& policy,
                                    const torch::Tensor& draft_token_ids,
                                    const torch::Tensor& draft_probs,
                                    const ForwardOutput& target_output,
                                    const torch::Tensor& bonus_token_ids,
                                    int32_t batch_size,
                                    int32_t num_val_tokens,
                                    bool enable_fused_kernel) {
  const int32_t vocab_size =
      static_cast<int32_t>(target_output.logits.size(/*dim=*/-1));
  torch::Tensor target_logits =
      target_output.logits.view({batch_size, num_val_tokens, vocab_size});

  auto rejection_sampler =
      std::make_unique<RejectionSampler>(policy.do_sample,
                                         policy.all_random_sample,
                                         policy.all_greedy_sample,
                                         target_output.logprobs,
                                         target_output.max_top_logprobs,
                                         enable_fused_kernel);

  SampleOutput sample_output =
      rejection_sampler->forward(draft_token_ids.to(bonus_token_ids),
                                 draft_probs.to(target_logits.device()),
                                 target_logits,
                                 bonus_token_ids,
                                 /*mask_out_rejected_tokens=*/true);

  const torch::Tensor& embeddings = target_output.sample_output.embeddings;
  sample_output.embeddings =
      embeddings.view({batch_size, num_val_tokens, embeddings.size(-1)});
  return sample_output;
}

}  // namespace xllm::spec_verify
