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

#include "core/framework/sampling/draft_proposal.h"
#include "sampling_params.h"

namespace xllm {

class RejectionSampler final {
 public:
  RejectionSampler(const torch::Tensor& do_sample,
                   bool all_random_sample,
                   bool all_greedy_sample,
                   bool logprobs,
                   int64_t max_top_logprobs);

  // operator() allows us to use the module as a function.
  template <typename... Args>
  auto operator()(Args&&... args) const {
    return this->forward(::std::forward<Args>(args)...);
  }

  // The single sampling owner for speculative verification. The
  // general-purpose Sampler never runs for verify steps.
  SampleOutput forward(const DraftProposal& draft_proposal,
                       const torch::Tensor& target_logits,
                       bool mask_out_rejected_tokens = false) const;

  // build mask from accepted matrix
  // for example: [[1, 1, 0, 1],   ->   [[1, 1, 1, 0, 0],
  //               [1, 0, 0, 0]]         [1, 1, 0, 0, 0]]
  static torch::Tensor build_accepted_mask(const torch::Tensor& accepted);

  static std::tuple<torch::Tensor, torch::Tensor> random_sample(
      const DraftProposal& draft_proposal,
      const torch::Tensor& target_probs,
      const torch::Tensor& uniform_rand,
      const torch::Tensor& bonus_token_ids,
      bool mask_out_rejected_tokens);

 private:
  // whether to return logprobs
  bool logprobs_ = false;

  // max number of top logprobs in the batch
  int64_t max_top_logprobs_ = 0;

  // [batch_size]
  torch::Tensor do_sample_;
  bool all_random_sample_ = true;
  bool all_greedy_sample_ = true;
};

}  // namespace xllm
