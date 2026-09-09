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
#include <c10/core/TensorOptions.h>
#include <torch/torch.h>
#include <torch/types.h>

#include "sampling_params.h"

namespace xllm {

class Sampler final {
 public:
  Sampler() = default;

  // operator() allows us to use the module as a function.
  template <typename... Args>
  auto operator()(Args&&... args) const {
    return this->forward(::std::forward<Args>(args)...);
  }

  // logits: [batch_size, vocab_size]
  SampleOutput forward(
      torch::Tensor& logits,
      const SamplingParameters& params,
      const torch::Tensor& filter_mask = torch::Tensor()) const;

  // helper functions
  // probs: [..., vocab_size]
  static torch::Tensor greedy_sample(const torch::Tensor& probs);

  // probs: [..., vocab_size]
  static torch::Tensor random_sample(const torch::Tensor& probs);
};

}  // namespace xllm
