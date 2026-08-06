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
#include <string>
#include <vector>

#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "layers/common/activation.h"
#include "layers/common/dense_mlp.h"
#include "layers/common/fused_moe_base.h"
#include "layers/common/linear.h"

namespace xllm {
namespace layer {

// Qwen3.5 routed MoE with masked grouped-GEMM. Currently TP1/EP1 only:
// partial expert replication would be incorrect, so larger TP/EP fails fast.
class Qwen3_5FusedMoEImpl : public torch::nn::Module {
 public:
  Qwen3_5FusedMoEImpl() = default;
  Qwen3_5FusedMoEImpl(const ModelArgs& model_args,
                      const FusedMoEArgs& moe_args,
                      const QuantArgs& quant_args,
                      const ParallelArgs& parallel_args,
                      const torch::TensorOptions& options);

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const ModelInputParams& input_params);

  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights() const;

 private:
  torch::Tensor forward_chunk(const torch::Tensor& hidden_states,
                              bool is_decode);
  void load_routed_weights(const StateDict& state_dict);

  int64_t num_experts_ = 0;
  int64_t topk_ = 0;
  int64_t hidden_size_ = 0;
  int64_t intermediate_size_ = 0;
  bool use_fp8_ = false;
  bool use_contiguous_bf16_moe_ = false;
  bool use_contiguous_fp8_moe_ = false;

  int64_t rank_ = 0;
  int64_t world_size_ = 1;
  int64_t start_expert_id_ = 0;
  int64_t num_experts_per_rank_ = 0;

  torch::TensorOptions options_;
  ProcessGroup* tp_pg_ = nullptr;

  ReplicatedLinear gate_{nullptr};
  DenseMLP shared_experts_{nullptr};
  torch::nn::Linear shared_expert_gate_{nullptr};
  Activation activation_{nullptr};

  DEFINE_WEIGHT(w13);
  DEFINE_FUSED_WEIGHT(w2);
  DEFINE_WEIGHT(w13_scale_inv);
  DEFINE_FUSED_WEIGHT(w2_scale_inv);
  DEFINE_FUSED_WEIGHT(w1);
  DEFINE_FUSED_WEIGHT(w3);
  DEFINE_FUSED_WEIGHT(w1_scale_inv);
  DEFINE_FUSED_WEIGHT(w3_scale_inv);

  bool shared_expert_gate_is_loaded_ = false;
};

TORCH_MODULE(Qwen3_5FusedMoE);

}  // namespace layer
}  // namespace xllm
