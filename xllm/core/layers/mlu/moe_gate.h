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

#include <optional>

#include "framework/model/model_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "layers/common/linear.h"
#include "layers/mlu/moe_softplus_topk.h"

namespace xllm {
namespace layer {

// MoE Gate: gate projection (ReplicatedLinear) + moe_active_topk.
class MoEGateImpl : public torch::nn::Module {
 public:
  MoEGateImpl() = default;

  // Reads MoE gate config from model_args (n_routed_experts,
  // num_experts_per_tok, n_group, topk_group, routed_scaling_factor,
  // hidden_size, norm_topk_prob, scoring_func, topk_method).
  MoEGateImpl(const ModelArgs& model_args,
              const QuantArgs& quant_args,
              const torch::TensorOptions& options,
              bool use_hash = false);

  // Forward: hidden_states -> (reduce_weight, expert_id).
  // Reshapes hidden_states to 2D, runs gate projection, then moe_active_topk.
  std::tuple<torch::Tensor, torch::Tensor> forward(
      torch::Tensor& hidden_states,
      const std::optional<torch::Tensor>& input_ids = std::nullopt);

  // Load state dict. Gate and e_score_correction_bias
  void load_state_dict(const StateDict& state_dict);

 private:
  int64_t num_experts_;
  int64_t topk_;
  int64_t num_expert_group_;
  int64_t topk_group_;
  double route_scale_;
  int64_t hidden_size_;
  int64_t renormalize_;
  std::string scoring_func_;

  ReplicatedLinear gate_{nullptr};
  MOESoftPlusTopK moe_softplus_topk_{nullptr};
  DEFINE_WEIGHT(e_score_correction_bias);
};

TORCH_MODULE(MoEGate);

}  // namespace layer
}  // namespace xllm
