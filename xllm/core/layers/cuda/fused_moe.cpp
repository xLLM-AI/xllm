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

#include "fused_moe.h"

#include <glog/logging.h>

#include <string>
#include <vector>

#include "framework/parallel_state/parallel_state.h"
#include "kernels/cuda/cuda_ops_api.h"

namespace xllm {
namespace layer {

FusedMoEImpl::FusedMoEImpl(const ModelArgs& model_args,
                           const FusedMoEArgs& moe_args,
                           const QuantArgs& quant_args,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options)
    : num_total_experts_(model_args.n_routed_experts()),
      hidden_size_(model_args.hidden_size()),
      options_(options),
      tp_pg_(parallel_args.tp_group_) {
  CHECK(moe_args.is_gated) << "CUDA FusedMoE only supports gated experts.";
  CHECK(moe_args.enable_result_reduction)
      << "CUDA FusedMoE requires result reduction.";
  CHECK(quant_args.quant_method().empty())
      << "CUDA FusedMoE currently supports only unquantized expert weights.";
  CHECK_EQ(model_args.n_shared_experts(), 0)
      << "CUDA FusedMoE does not support shared experts yet.";
  CHECK(tp_pg_ != nullptr) << "CUDA FusedMoE requires a TP process group.";

  const int64_t num_experts = num_total_experts_;
  const int64_t intermediate_size =
      static_cast<int64_t>(model_args.moe_intermediate_size());
  const int32_t ep_size = parallel_args.ep_size();
  int32_t ep_rank = 0;
  CHECK_GT(ep_size, 0) << "ep_size must be positive.";
  if (ep_size > 1) {
    CHECK(parallel_args.moe_ep_group_ != nullptr)
        << "CUDA FusedMoE requires a MoE EP process group when ep_size > 1.";
    CHECK(parallel_args.moe_tp_group_ != nullptr)
        << "CUDA FusedMoE requires a MoE TP process group when ep_size > 1.";
    ep_rank = parallel_args.moe_ep_group_->rank();
    tp_pg_ = parallel_args.moe_tp_group_;
    ep_pg_ = parallel_args.moe_ep_group_;
  }
  CHECK_EQ(num_experts % ep_size, 0)
      << "n_routed_experts must be divisible by ep_size.";

  ep_size_ = ep_size;
  ep_rank_ = ep_rank;
  num_experts_per_rank_ = num_experts / ep_size;
  start_expert_id_ = ep_rank * num_experts_per_rank_;

  gate_ = register_module(
      "gate",
      ReplicatedLinear(hidden_size_, num_experts, false, quant_args, options));
  fused_topk_ = register_module("fused_topk",
                                MoEFusedTopk(model_args, quant_args, options));

  // create weight buffer
  const int64_t world_size = tp_pg_->world_size();
  CHECK_EQ(intermediate_size % world_size, 0)
      << "moe_intermediate_size must be divisible by TP world size.";
  const int64_t local_intermediate_size = intermediate_size / world_size;

  w13_ = register_parameter(
      "w13",
      torch::empty(
          {num_experts_per_rank_, local_intermediate_size * 2, hidden_size_},
          options_),
      false);
  w2_ = register_parameter(
      "w2",
      torch::empty(
          {num_experts_per_rank_, hidden_size_, local_intermediate_size},
          options_),
      false);
}

torch::Tensor FusedMoEImpl::forward_experts(const torch::Tensor& hidden_states,
                                            torch::Tensor router_logits) {
  auto [token_final_scales, token_selected_experts] =
      fused_topk_->forward(router_logits);

  std::vector<torch::Tensor> quant_scales;
  torch::Tensor output =
      xllm::kernel::cuda::cutlass_fused_moe(hidden_states,
                                            token_selected_experts,
                                            token_final_scales,
                                            w13_,
                                            w2_,
                                            options_.dtype().toScalarType(),
                                            quant_scales,
                                            tp_pg_->world_size(),
                                            tp_pg_->rank(),
                                            ep_size_,
                                            ep_rank_,
                                            /*cluster_size=*/1,
                                            /*cluster_rank=*/0);
  output = parallel_state::reduce(output, tp_pg_);
  return parallel_state::reduce(output, ep_pg_);
}

torch::Tensor FusedMoEImpl::forward(const torch::Tensor& hidden_states,
                                    const ModelInputParams& /*input_params*/) {
  torch::Tensor router_logits = gate_->forward(hidden_states);
  return forward_experts(hidden_states, router_logits);
}

void FusedMoEImpl::load_experts(const StateDict& state_dict) {
  const int64_t rank = tp_pg_->rank();
  const int64_t world_size = tp_pg_->world_size();
  const int64_t start_expert_id = start_expert_id_;
  const int64_t num_experts_per_rank = num_experts_per_rank_;
  // CUTLASS SwiGLU consumes fc1 as [linear, gate], i.e. [up_proj, gate_proj].
  std::vector<std::string> prefixes = {"up_proj.", "gate_proj."};

  LOAD_MOE_FUSED_WEIGHT("weight", w1, w3, w13);
  LOAD_MOE_WEIGHT("down_proj.", "weight", w2, 1);
}

void FusedMoEImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }

  gate_->load_state_dict(state_dict.get_dict_with_prefix("gate."));
  fused_topk_->load_state_dict(state_dict.get_dict_with_prefix("gate."));
  load_experts(state_dict.get_dict_with_prefix("experts."));
}

}  // namespace layer
}  // namespace xllm
