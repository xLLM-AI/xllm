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

#include "layers/musa/lm_head_impl.h"

#include "MTTOplib/Ops.h"
#include "MTTOplib/WeightReorder.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace xllm {
namespace layer {
MUSALmHeadImpl::MUSALmHeadImpl(const ModelContext& context)
    : options_(context.get_tensor_options()),
      hidden_size_(context.get_model_args().hidden_size()),
      vocab_size_(context.get_model_args().vocab_size()) {}

void MUSALmHeadImpl::load_state_dict(StateDict const& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  DEFINE_WEIGHT(weight);
  weight_ = torch::empty({vocab_size_, hidden_size_}, options_);
  LOAD_WEIGHT(weight);
  if (weight_is_loaded_) {
    weights_.emplace_back(weight_.clone());
    weights_ = xllm_musa::ReorderLMHead(weights_);
  }
}

torch::Tensor MUSALmHeadImpl::forward(torch::Tensor const& input) {
  torch::Tensor out =
      torch::empty({input.size(0), weights_[0].size(0)}, input.options());
  xllm_musa::Matmul(input, out, weights_[0]);
  return out;
}
}  // namespace layer
}  // namespace xllm
