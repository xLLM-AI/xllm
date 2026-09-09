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

#include "moe_gate.h"

#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

MoEGateImpl::MoEGateImpl(const ModelArgs& model_args,
                         const QuantArgs& quant_args,
                         const torch::TensorOptions& options,
                         bool use_hash)
    : num_experts_(model_args.n_routed_experts()),
      topk_(model_args.num_experts_per_tok()),
      num_expert_group_(model_args.n_group()),
      topk_group_(model_args.topk_group()),
      route_scale_(static_cast<double>(model_args.routed_scaling_factor())),
      hidden_size_(model_args.hidden_size()),
      renormalize_(model_args.norm_topk_prob() ? 1 : 0),
      scoring_func_(model_args.scoring_func()) {
  const std::string& topk_method = model_args.topk_method();
  if (topk_method == "noaux_tc") {
    e_score_correction_bias_ = register_parameter(
        "e_score_correction_bias",
        torch::empty({model_args.n_routed_experts()}, options),
        false);
  }

  if (scoring_func_ == "sqrtsoftplus") {
    moe_softplus_topk_ =
        register_module("moe_softplus_topk",
                        MOESoftPlusTopK(model_args.n_routed_experts(),
                                        model_args.num_experts_per_tok(),
                                        model_args.routed_scaling_factor(),
                                        model_args.vocab_size(),
                                        use_hash,
                                        options));
  }

  gate_ = register_module("gate_proj",
                          ReplicatedLinear(model_args.hidden_size(),
                                           model_args.n_routed_experts(),
                                           false,
                                           quant_args,
                                           options));
}

std::tuple<torch::Tensor, torch::Tensor> MoEGateImpl::forward(
    torch::Tensor& hidden_states,
    const std::optional<torch::Tensor>& input_ids) {
  torch::Tensor router_logits = gate_->forward(hidden_states);
  torch::Tensor router_logits_2d =
      router_logits.reshape({-1, router_logits.size(-1)});

  if (moe_softplus_topk_) {
    return moe_softplus_topk_->forward(router_logits_2d, input_ids);
  }

  std::optional<torch::Tensor> e_score_correction_bias = std::nullopt;
  if (e_score_correction_bias_.defined()) {
    e_score_correction_bias = e_score_correction_bias_;
  }

  xllm::kernel::MoeFusedTopkParams moe_active_topk_params;
  moe_active_topk_params.input = router_logits_2d;
  moe_active_topk_params.topk = topk_;
  moe_active_topk_params.num_expert_group = num_expert_group_;
  moe_active_topk_params.topk_group = topk_group_;
  moe_active_topk_params.normalize = static_cast<bool>(renormalize_);
  moe_active_topk_params.normed_by = "topk_logit";
  moe_active_topk_params.scoring_func = scoring_func_;
  moe_active_topk_params.route_scale = route_scale_;
  moe_active_topk_params.e_score_correction_bias = e_score_correction_bias;

  return xllm::kernel::moe_active_topk(moe_active_topk_params);
}

void MoEGateImpl::load_state_dict(const StateDict& state_dict) {
  gate_->load_state_dict(state_dict);
  if (e_score_correction_bias_.defined() &&
      !e_score_correction_bias_is_loaded_) {
    LOAD_WEIGHT(e_score_correction_bias);
  }
  if (moe_softplus_topk_) {
    moe_softplus_topk_->load_state_dict(state_dict);
  }
}

}  // namespace layer
}  // namespace xllm
