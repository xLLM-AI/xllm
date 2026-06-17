/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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
#include <torch/types.h>

#include "sampling_params.h"

namespace xllm {

class RejectionSampler final {
 public:
  RejectionSampler(const torch::Tensor& do_sample,
                   bool all_random_sample,
                   bool all_greedy_sample,
                   bool logprobs,
                   int64_t max_top_logprobs,
                   bool enable_fused_kernel = false);

  // operator() allows us to use the module as a function.
  template <typename... Args>
  auto operator()(Args&&... args) const {
    return this->forward(::std::forward<Args>(args)...);
  }

  // Sample tokens ids using rejection sampling.
  // draft_token_ids: [batch_size, n_speculative_tokens]
  // draft_probs:
  //   1) dense format: [batch_size, n_speculative_tokens, vocab_size]
  //   2) selected-only format: [batch_size, n_speculative_tokens]
  // target_logits: [batch_size, n_speculative_tokens + 1, vocab_size]
  // bonus_token_ids: [batch_size, 1]
  SampleOutput forward(const torch::Tensor& draft_token_ids,
                       const torch::Tensor& draft_probs,
                       const torch::Tensor& target_logits,
                       const torch::Tensor& bonus_token_ids,
                       bool mask_out_rejected_tokens = false) const;

  // build mask from accepted matrix
  // for example: [[1, 1, 0, 1],   ->   [[1, 1, 1, 0, 0],
  //               [1, 0, 0, 0]]         [1, 1, 0, 0, 0]]
  static torch::Tensor build_accepted_mask(const torch::Tensor& accepted);

  static std::tuple<torch::Tensor, torch::Tensor> random_sample(
      const torch::Tensor& draft_token_ids,
      const torch::Tensor& draft_probs,
      const torch::Tensor& target_probs,
      const torch::Tensor& uniform_rand,
      const torch::Tensor& bonus_token_ids,
      bool mask_out_rejected_tokens);

  static std::tuple<torch::Tensor, torch::Tensor> random_sample_fused(
      const torch::Tensor& draft_token_ids,
      const torch::Tensor& draft_probs,
      const torch::Tensor& target_probs,
      const torch::Tensor& uniform_rand,
      const torch::Tensor& bonus_token_ids,
      bool mask_out_rejected_tokens);

  static std::tuple<torch::Tensor, torch::Tensor> greedy_sample(
      const torch::Tensor& draft_token_ids,
      const torch::Tensor& target_probs,
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

  // whether to use fused kernel
  bool enable_fused_kernel_ = false;
};

}  // namespace xllm
