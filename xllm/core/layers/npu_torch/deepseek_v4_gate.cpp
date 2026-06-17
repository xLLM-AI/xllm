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

#include "deepseek_v4_gate.h"

#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#endif

#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

namespace {

bool check_npu_moe_gating_top_k(const torch::Tensor& hidden_states,
                                int64_t top_k,
                                bool renormalize,
                                int64_t norm_type) {
  if (norm_type == 1 && !renormalize) {
    return false;
  }
  if (norm_type != 0 && norm_type != 1 && norm_type != 2) {
    return false;
  }

  constexpr int64_t topk_group = 1;
  constexpr int64_t num_expert_group = 1;

  const int64_t hidden_dim = hidden_states.size(-1);
  if (!(num_expert_group > 0 && hidden_dim % num_expert_group == 0 &&
        hidden_dim / num_expert_group > 2)) {
    return false;
  }
  if (topk_group < 1 || topk_group > num_expert_group) {
    return false;
  }
  if (top_k < 1 || top_k > (hidden_dim / (num_expert_group * topk_group))) {
    return false;
  }
  if (topk_group * hidden_dim / num_expert_group < top_k) {
    return false;
  }
  return true;
}

torch::Tensor renormalize_topk_weights(const torch::Tensor& topk_weights) {
  auto denom = topk_weights.sum(-1, true);
  denom = torch::clamp_min(denom, 1e-20);
  return topk_weights / denom;
}

}  // namespace

DeepseekV4GateImpl::DeepseekV4GateImpl(const ModelContext& context,
                                       int32_t layer_id)
    : DeepseekV4GateImpl(context.get_model_args(),
                         layer_id,
                         context.get_tensor_options()) {}

DeepseekV4GateImpl::DeepseekV4GateImpl(const ModelArgs& args,
                                       int32_t layer_id,
                                       const torch::TensorOptions& options) {
  hidden_size_ = args.hidden_size();
  n_routed_experts_ = args.n_routed_experts();
  topk_ = args.n_activated_experts();
  n_hash_layers_ = args.n_hash_layers();
  route_scale_ =
      1.0;  // args.routed_scaling_factor(); # TODO: add param for dsv4
  score_func_ = args.scoring_func();
  hash_layer_ = layer_id >= 0 && layer_id < n_hash_layers_;

  CHECK_GT(hidden_size_, 0)
      << "DeepseekV4Gate requires hidden_size > 0, got " << hidden_size_;
  CHECK_GT(n_routed_experts_, 0)
      << "DeepseekV4Gate requires n_routed_experts > 0, got "
      << n_routed_experts_;
  CHECK_GT(topk_, 0)
      << "DeepseekV4Gate requires n_activated_experts(topk) > 0, got " << topk_;
  CHECK_LE(topk_, n_routed_experts_)
      << "DeepseekV4Gate n_activated_experts(topk) must be <= "
      << "n_routed_experts, got topk=" << topk_
      << ", n_routed_experts=" << n_routed_experts_;

  weight_ = register_parameter(
      "weight",
      torch::empty({n_routed_experts_, hidden_size_}, options),
      /*requires_grad=*/false);

  weight_.set_data(
      at_npu::native::npu_format_cast(weight_, ACL_FORMAT_FRACTAL_NZ));

  if (hash_layer_) {
    const int64_t vocab_size = args.vocab_size();
    CHECK_GT(vocab_size, 0)
        << "DeepseekV4Gate hash layer requires vocab_size > 0, got "
        << vocab_size;
    tid2eid_ = register_parameter(
        "tid2eid",
        torch::empty({vocab_size, topk_},
                     options.dtype(torch::kInt32).device(options.device())),
        /*requires_grad=*/false);
  } else {
    bias_ = register_parameter(
        "bias",
        torch::empty({n_routed_experts_},
                     options.dtype(torch::kFloat32).device(options.device())),
        /*requires_grad=*/false);
  }
}

std::tuple<torch::Tensor, torch::Tensor> DeepseekV4GateImpl::forward(
    const torch::Tensor& hidden_states,
    const std::optional<torch::Tensor>& input_ids) {
  CHECK(hidden_states.defined())
      << "DeepseekV4Gate::forward hidden_states is undefined";
  CHECK_EQ(hidden_states.size(-1), hidden_size_)
      << "DeepseekV4Gate::forward hidden_states last dim mismatch, expected "
      << hidden_size_ << " got " << hidden_states.size(-1);

  auto logits = torch::matmul(hidden_states, weight_.transpose(0, 1));

  constexpr bool renormalize = true;
  const int64_t norm_type = score_func_to_norm_type(score_func_);
  const bool is_support_npu_moe_gating_top_k =
      check_npu_moe_gating_top_k(hidden_states, topk_, renormalize, norm_type);

  if (!is_support_npu_moe_gating_top_k) {
    return select_experts_native(logits, input_ids);
  }

  kernel::MoeGatingTopKHashParams gate_params;
  gate_params.x = logits;
  gate_params.k = topk_;
  gate_params.k_group = 1;
  gate_params.group_count = 1;
  gate_params.group_select_mode = 1;
  gate_params.norm_type = norm_type;
  gate_params.renorm = (gate_params.norm_type == 2) ? 0 : 1;
  gate_params.out_flag = false;
  gate_params.routed_scaling_factor = route_scale_;
  gate_params.eps = 1e-20;

  const bool has_hash_table = hash_layer_ && tid2eid_.defined();
  if (has_hash_table) {
    if (input_ids.has_value() && input_ids.value().defined()) {
      gate_params.input_ids = input_ids.value();
    } else {
      gate_params.input_ids = c10::nullopt;
    }
    gate_params.tid2eid = tid2eid_;
    gate_params.bias = c10::nullopt;
  } else {
    gate_params.input_ids = c10::nullopt;
    gate_params.tid2eid = c10::nullopt;
    if (!hash_layer_ && bias_.defined()) {
      gate_params.bias = bias_.to(logits.dtype());
    } else {
      gate_params.bias = c10::nullopt;
    }
  }
  auto [topk_weights, topk_idx, score_out] =
      kernel::moe_gating_top_k_hash(gate_params);
  (void)score_out;

  if (gate_params.norm_type == 0 && renormalize) {
    topk_weights = renormalize_topk_weights(topk_weights);
  }

  if (gate_params.norm_type == 2) {
    topk_weights = renormalize_topk_weights(topk_weights);
  }

  return std::make_tuple(topk_weights, topk_idx.to(torch::kInt32));
}

std::tuple<torch::Tensor, torch::Tensor>
DeepseekV4GateImpl::select_experts_native(
    const torch::Tensor& router_logits,
    const std::optional<torch::Tensor>& input_ids) const {
  torch::Tensor scores;
  const int64_t norm_type = score_func_to_norm_type(score_func_);
  if (norm_type == 0) {
    scores = torch::softmax(router_logits, -1);
  } else if (norm_type == 1) {
    scores = torch::sigmoid(router_logits);
  } else {
    scores = torch::softplus(router_logits).sqrt();
  }

  auto original_scores = scores;
  if (!hash_layer_ && bias_.defined()) {
    scores = scores + bias_.to(scores.dtype());
  }

  torch::Tensor topk_idx;
  if (hash_layer_ && input_ids.has_value() && input_ids.value().defined() &&
      tid2eid_.defined()) {
    auto lookup_ids = input_ids.value().to(torch::kLong);
    topk_idx = tid2eid_.index({lookup_ids});
  } else {
    topk_idx = std::get<1>(torch::topk(scores,
                                       topk_,
                                       /*dim=*/-1,
                                       /*largest=*/true,
                                       /*sorted=*/false));
  }

  auto gather_idx = topk_idx.to(torch::kLong);
  auto topk_weights = original_scores.gather(-1, gather_idx);
  auto denom = topk_weights.sum(-1, true);
  denom = torch::clamp_min(denom, 1e-20);
  topk_weights = topk_weights / denom;
  topk_weights = topk_weights * route_scale_;

  return std::make_tuple(topk_weights, gather_idx.to(torch::kInt32));
}

void DeepseekV4GateImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }

  auto gate_weight = state_dict.get_tensor("weight");
  CHECK(gate_weight.defined())
      << "DeepseekV4Gate missing weight in state_dict with prefix "
      << state_dict.prefix();
  {
    torch::NoGradGuard no_grad;
    weight_.copy_(gate_weight.to(weight_.device()).to(weight_.dtype()));
  }

  if (hash_layer_) {
    auto tid2eid = state_dict.get_tensor("tid2eid");
    if (!tid2eid.defined()) {
      tid2eid = state_dict.get_tensor("tid2eid.weight");
    }
    CHECK(tid2eid.defined())
        << "DeepseekV4Gate hash layer missing tid2eid in state_dict with "
        << "prefix " << state_dict.prefix();
    torch::NoGradGuard no_grad;
    tid2eid_.copy_(tid2eid.to(tid2eid_.device()).to(tid2eid_.dtype()));
    return;
  }

  auto bias = state_dict.get_tensor("bias");
  if (!bias.defined()) {
    bias = state_dict.get_tensor("e_score_correction_bias");
  }
  CHECK(bias.defined()) << "DeepseekV4Gate non-hash layer missing bias (or "
                        << "e_score_correction_bias) in state_dict with prefix "
                        << state_dict.prefix();
  {
    torch::NoGradGuard no_grad;
    bias_.copy_(bias.to(bias_.device()).to(bias_.dtype()));
  }
}

int64_t DeepseekV4GateImpl::score_func_to_norm_type(
    const std::string& score_func) const {
  std::string lowered = score_func;
  std::transform(
      lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
      });

  if (lowered == "softmax") {
    return 0;
  }
  if (lowered == "sigmoid") {
    return 1;
  }
  if (lowered == "sqrtsoftplus") {
    return 2;
  }

  CHECK(false) << "DeepseekV4Gate unsupported score_func: " << score_func;
  return 0;
}

}  // namespace layer
}  // namespace xllm
