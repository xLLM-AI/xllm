/* Copyright 2025-2026 The xLLM Authors.

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

#include "framework/model/model_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace xllm {
namespace layer {

class MoEFusedTopkImpl : public torch::nn::Module {
 public:
  MoEFusedTopkImpl(const ModelArgs& model_args,
                   const QuantArgs& quant_args,
                   const torch::TensorOptions& options);

  std::tuple<torch::Tensor, torch::Tensor> forward(
      torch::Tensor& router_logits);

  void load_state_dict(const StateDict& state_dict);

 private:
  int64_t topk_;
  int64_t num_expert_group_;
  int64_t topk_group_;
  double route_scale_;
  int64_t hidden_size_;
  bool renormalize_;
  std::string scoring_func_;

  DEFINE_WEIGHT(e_score_correction_bias);
};

TORCH_MODULE(MoEFusedTopk);
}  // namespace layer
}  // namespace xllm
