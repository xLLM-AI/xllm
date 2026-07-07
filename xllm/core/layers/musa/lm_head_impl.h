/* Copyright 2025-2026 The xLLM Authors.

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

#include <cstdint>
#include <vector>

#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace xllm {
namespace layer {
class MUSALmHeadImpl : public torch::nn::Module {
 public:
  explicit MUSALmHeadImpl(const ModelContext& context);

  ~MUSALmHeadImpl() {};

  void load_state_dict(StateDict const& state_dict);

  torch::Tensor forward(torch::Tensor const& input);

 private:
  int64_t hidden_size_;
  int64_t vocab_size_;
  torch::TensorOptions options_;
  std::vector<torch::Tensor> weights_;
};
}  // namespace layer
}  // namespace xllm