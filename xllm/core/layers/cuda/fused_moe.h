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

#include <string>
#include <vector>

#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "layers/common/fused_moe_base.h"
#include "layers/common/linear.h"
#include "layers/common/moe_fused_topk.h"

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

  torch::Tensor forward_experts(const torch::Tensor& hidden_states,
                                torch::Tensor router_logits);
  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const ModelInputParams& input_params);
  void load_state_dict(const StateDict& state_dict);

 private:
  int64_t num_total_experts_;
  int64_t hidden_size_;

  int64_t num_experts_per_rank_;
  int64_t start_expert_id_;
  int32_t ep_size_;
  int32_t ep_rank_;

  ReplicatedLinear gate_{nullptr};
  MoEFusedTopk fused_topk_{nullptr};

  torch::TensorOptions options_;
  ProcessGroup* tp_pg_ = nullptr;
  ProcessGroup* ep_pg_ = nullptr;

  DEFINE_WEIGHT(w13);
  DEFINE_FUSED_WEIGHT(w1);
  DEFINE_FUSED_WEIGHT(w3);
  DEFINE_FUSED_WEIGHT(w2);

  void load_experts(const StateDict& state_dict);
};
TORCH_MODULE(FusedMoE);

}  // namespace layer
}  // namespace xllm
