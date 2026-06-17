/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <optional>
#include <string>
#include <utility>

#include "common/flash_comm1_context.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "layers/common/dense_mlp.h"
#include "layers/common/fused_moe_base.h"
#include "layers/common/linear.h"

namespace xllm {
namespace layer {

class FusedMoEImpl : public torch::nn::Module {
 public:
  FusedMoEImpl() = default;
  FusedMoEImpl(const ModelArgs& model_args,
               const FusedMoEArgs& moe_args,
               const QuantArgs& quant_args,
               const ParallelArgs& parallel_args,
               const torch::TensorOptions& options);

  torch::Tensor forward_expert(
      const torch::Tensor& hidden_states,
      const torch::Tensor& router_logits,
      const std::optional<torch::Tensor>& shared_output,
      const FlashComm1Context* fc1_ctx = nullptr);
  torch::Tensor forward_with_selected_experts(
      const torch::Tensor& hidden_states,
      const torch::Tensor& topk_weights,
      const torch::Tensor& topk_ids,
      const ModelInputParams& input_params);
  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const ModelInputParams& input_params);
  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const ModelInputParams& input_params,
                        const FlashComm1Context* fc1_ctx);
  void load_state_dict(const StateDict& state_dict);

 private:
  // struct to store the selected expert info
  struct SelectedExpertInfo {
    torch::Tensor reduce_weight;
    torch::Tensor combine_idx;
    torch::Tensor token_count_slice;
    torch::Tensor cusum_token_count;
    std::optional<torch::Tensor> input_scale;
  };

  // initial steps for MoE computation, select the experts for each token
  torch::Tensor select_experts(const torch::Tensor& hidden_states_2d,
                               const torch::Tensor& router_logits_2d,
                               SelectedExpertInfo& selected_expert_info);

 private:
  int64_t num_total_experts_;
  int64_t topk_;
  int64_t num_expert_group_;
  int64_t topk_group_;
  double route_scale_;
  int64_t hidden_size_;
  int64_t n_shared_experts_;
  bool is_gated_;
  bool has_score_bias_;
  bool has_bias_;
  bool skip_bias_add_;
  bool skip_gate_load_;
  bool is_deepseek_v4_ = false;
  bool shared_expert_gate_is_loaded_ = false;
  int64_t renormalize_;
  std::string hidden_act_;
  std::string scoring_func_;
  bool is_smoothquant_;
  std::optional<std::string> resolved_moe_quant_method_;
  std::optional<std::pair<torch::Tensor, torch::Tensor>> preselected_experts_;

  int64_t num_experts_per_rank_;
  int64_t start_expert_id_;
  int64_t local_intermediate_size_;
  bool w4a8_dynamic_preprocessed_ = false;
  bool w13_group_gemm_layout_prepared_ = false;
  bool w2_group_gemm_layout_prepared_ = false;

  ReplicatedLinear gate_{nullptr};
  DenseMLP shared_experts_{nullptr};
  torch::nn::Linear shared_expert_gate_{nullptr};
  QuantArgs quant_args_;
  ParallelArgs parallel_args_;
  torch::TensorOptions options_;
  ProcessGroup* tp_pg_;

  DEFINE_WEIGHT(w13);
  DEFINE_FUSED_WEIGHT(w1);
  DEFINE_FUSED_WEIGHT(w3);
  DEFINE_FUSED_WEIGHT(w2);
  DEFINE_WEIGHT(e_score_correction_bias);
  DEFINE_WEIGHT(w13_scale);
  DEFINE_FUSED_WEIGHT(w1_scale);
  DEFINE_FUSED_WEIGHT(w3_scale);
  DEFINE_FUSED_WEIGHT(w2_scale);
  DEFINE_FUSED_WEIGHT(w1_scale_second);
  DEFINE_FUSED_WEIGHT(w3_scale_second);
  DEFINE_FUSED_WEIGHT(w13_scale_second);
  DEFINE_FUSED_WEIGHT(w2_scale_second);
  DEFINE_FUSED_WEIGHT(w1_scale_bias);
  DEFINE_FUSED_WEIGHT(w3_scale_bias);
  DEFINE_FUSED_WEIGHT(w13_scale_bias);
  DEFINE_FUSED_WEIGHT(w2_scale_bias);
  DEFINE_FUSED_WEIGHT(input_smooth);
  DEFINE_FUSED_WEIGHT(act_smooth);

  void load_e_score_correction_bias(const StateDict& state_dict);
  void load_experts(const StateDict& state_dict);
  void resolve_quant_method_from_state_dict(const StateDict& state_dict);
  void validate_resolved_quant_method() const;
  void ensure_quant_weight_layout();
  void ensure_group_gemm_weight_layout(torch::Tensor& weight,
                                       bool& prepared,
                                       int64_t input_dim,
                                       int64_t output_dim,
                                       const char* name);
  void preprocess_w4a8_dynamic_weights();
  void clear_w4a8_dynamic_source_weight_cache();
  bool should_gather_dp_inputs_for_moe() const;
  bool can_use_ep2_dispatch_combine(const ModelInputParams& input_params,
                                    const torch::Tensor& hidden_states) const;
  int32_t fused_mc2_mode() const;
  bool prepare_dispatch_ffn_combine_inputs();
  bool prepare_dispatch_gmm_combine_decode_inputs();
  torch::Tensor forward_with_dispatch_ffn_combine(
      const torch::Tensor& input_2d,
      const torch::Tensor& weights_2d,
      const torch::Tensor& ids_2d,
      at::IntArrayRef hidden_states_shape);
  torch::Tensor forward_with_dispatch_gmm_combine_decode(
      const torch::Tensor& input_2d,
      const torch::Tensor& weights_2d,
      const torch::Tensor& ids_2d,
      at::IntArrayRef hidden_states_shape,
      int64_t global_bs);
  torch::Tensor forward_with_selected_experts_ep2(
      const torch::Tensor& hidden_states,
      const torch::Tensor& topk_weights,
      const torch::Tensor& topk_ids,
      const ModelInputParams& input_params);
  const std::string& get_moe_ep_group_name();

  bool enable_ep2_dispatch_combine_ = false;
  bool dispatch_ffn_combine_prepared_ = false;
  bool dispatch_gmm_combine_decode_prepared_ = false;
  torch::Tensor dispatch_ffn_w13_scale_;
  torch::Tensor dispatch_ffn_w2_scale_;
  std::string moe_ep_group_name_;
};
TORCH_MODULE(FusedMoE);

}  // namespace layer
}  // namespace xllm
