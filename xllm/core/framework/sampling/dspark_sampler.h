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

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <utility>

#include "framework/sampling/block_draft_sampler.h"
#include "framework/sampling/sampler.h"

namespace xllm::dspark {

template <typename MarkovBiasFn, typename SampledTokenSyncFn>
BlockDraftSampleOutput sample_block(
    const torch::Tensor& base_logits,
    const torch::Tensor& anchor_token_ids,
    const SamplingParameters& sampling_params,
    MarkovBiasFn&& markov_bias,
    SampledTokenSyncFn&& sync_sampled_token_ids) {
  CHECK_EQ(base_logits.dim(), 3)
      << "DSpark base_logits must be [num_reqs, n_spec, draft_vocab].";
  CHECK_EQ(anchor_token_ids.dim(), 1)
      << "DSpark anchor_token_ids must be [num_reqs].";
  CHECK_EQ(anchor_token_ids.size(0), base_logits.size(0))
      << "DSpark anchor token batch must match base_logits.";
  CHECK_EQ(anchor_token_ids.scalar_type(), torch::kLong)
      << "DSpark anchor_token_ids must use int64.";
  CHECK_EQ(anchor_token_ids.device(), base_logits.device())
      << "DSpark anchor_token_ids and base_logits must share a device.";

  const int64_t num_reqs = base_logits.size(0);
  const int64_t num_speculative_tokens = base_logits.size(1);
  SamplingParameters step_sampling_params = sampling_params;
  const torch::TensorOptions index_options =
      torch::TensorOptions().dtype(torch::kInt).device(base_logits.device());
  step_sampling_params.selected_token_idxes = torch::empty({0}, index_options);
  step_sampling_params.sample_idxes = torch::empty({0}, index_options);
  step_sampling_params.return_probs = !step_sampling_params.all_greedy_sample;
  step_sampling_params.logprobs = false;
  step_sampling_params.max_top_logprobs = 0;
  step_sampling_params.use_beam_search = false;

  torch::Tensor token_ids = torch::empty(
      {num_reqs, num_speculative_tokens},
      torch::TensorOptions().dtype(torch::kLong).device(base_logits.device()));
  torch::Tensor proposal_probs =
      torch::empty({num_reqs, num_speculative_tokens},
                   torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(base_logits.device()));

  using ISlice = torch::indexing::Slice;
  Sampler sampler;
  torch::Tensor previous_token_ids = anchor_token_ids;
  for (int64_t token_idx = 0; token_idx < num_speculative_tokens; ++token_idx) {
    torch::Tensor step_logits =
        base_logits.select(/*dim=*/1, /*index=*/token_idx) +
        markov_bias(previous_token_ids);
    SampleOutput sample_output =
        sampler.forward(step_logits, step_sampling_params);
    torch::Tensor sampled_token_ids = sample_output.next_tokens;
    sync_sampled_token_ids(sampled_token_ids);

    torch::Tensor selected_proposal_probs;
    if (step_sampling_params.all_greedy_sample) {
      selected_proposal_probs = torch::ones({num_reqs},
                                            torch::TensorOptions()
                                                .dtype(torch::kFloat32)
                                                .device(base_logits.device()));
    } else {
      CHECK_EQ(sample_output.probs.dim(), 2)
          << "DSpark random/mixed sampling requires dense proposal probs.";
      selected_proposal_probs =
          sample_output.probs.gather(/*dim=*/1, sampled_token_ids.view({-1, 1}))
              .view({-1})
              .to(torch::kFloat32);
      if (!step_sampling_params.all_random_sample) {
        selected_proposal_probs =
            torch::where(step_sampling_params.do_sample,
                         selected_proposal_probs,
                         torch::ones_like(selected_proposal_probs));
      }
    }

    token_ids.index_put_({ISlice(), token_idx}, sampled_token_ids);
    proposal_probs.index_put_({ISlice(), token_idx}, selected_proposal_probs);
    previous_token_ids = sampled_token_ids;
  }
  return {.token_ids = std::move(token_ids),
          .probs = std::move(proposal_probs)};
}

template <typename MarkovBiasFn>
BlockDraftSampleOutput sample_block(const torch::Tensor& base_logits,
                                    const torch::Tensor& anchor_token_ids,
                                    const SamplingParameters& sampling_params,
                                    MarkovBiasFn&& markov_bias) {
  return sample_block(base_logits,
                      anchor_token_ids,
                      sampling_params,
                      std::forward<MarkovBiasFn>(markov_bias),
                      [](torch::Tensor& /*sampled_token_ids*/) {});
}

}  // namespace xllm::dspark
