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

#include <utility>

#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "layers/common/dense_mlp.h"
#include "layers/common/fused_moe_base.h"
#include "layers/common/linear.h"
#include "platform/device.h"

namespace xllm {
namespace layer {

class FusedMoEImpl final : public torch::nn::Module {
 public:
  FusedMoEImpl() = default;
  FusedMoEImpl(const ModelArgs& model_args,
               const FusedMoEArgs& moe_args,
               const QuantArgs& quant_args,
               const ParallelArgs& parallel_args,
               const torch::TensorOptions& options);

  torch::Tensor forward_experts(const torch::Tensor& hidden_states,
                                const torch::Tensor& router_logits);
  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const ModelInputParams& input_params);
  void load_state_dict(const StateDict& state_dict);

 private:
  struct SelectedExpertInfo {
    torch::Tensor reduce_weight;  // [num_tokens, topk] float32
    torch::Tensor expert_id;      // [num_tokens, topk] int32
    // src_dst[i] = expert-grouped position for flat index i (all global
    // experts)
    torch::Tensor src_dst;  // [num_tokens * topk] int32
    // dst_src[j] = flat index for global expert-grouped position j
    torch::Tensor dst_src;  // [num_tokens * topk] int32
    // token counts for LOCAL experts only
    torch::Tensor token_count;  // [num_experts_per_rank] int32
    // global offset into the expanded buffer where local experts' tokens begin
    int64_t global_offset;
    // total number of tokens for local experts
    int64_t local_total;
  };

  torch::Tensor select_experts(const torch::Tensor& hidden_states_2d,
                               const torch::Tensor& router_logits_2d,
                               SelectedExpertInfo& selected_expert_info);

  std::pair<torch::Tensor, torch::Tensor> route_experts(
      const torch::Tensor& router_logits_2d);

  torch::Tensor forward_fp8_channelwise_experts(
      const torch::Tensor& hidden_states,
      const torch::Tensor& hidden_states_2d,
      const torch::Tensor& router_logits_2d);

  // per-expert matmul: groups input by token_count, multiplies by weight[e]
  torch::Tensor expert_gemm(const torch::Tensor& input,
                            const torch::Tensor& weight,
                            const torch::Tensor& token_count);

 private:
  int64_t num_total_experts_;
  int64_t topk_;
  int64_t num_expert_group_;
  int64_t topk_group_;
  double route_scale_;
  int64_t hidden_size_;
  int64_t n_shared_experts_;
  bool is_gated_;
  int64_t renormalize_;
  std::string hidden_act_;
  std::string scoring_func_;

  int64_t num_experts_per_rank_;
  int64_t start_expert_id_;

  // streams for parallel shared experts + allreduce
  std::unique_ptr<Stream> shared_stream_;
  std::unique_ptr<Stream> routed_stream_;
  xllm::Device device_;
  bool stream_initialized_ = false;

  ReplicatedLinear gate_{nullptr};
  DenseMLP shared_experts_{nullptr};

  QuantArgs quant_args_;
  ParallelArgs parallel_args_;
  torch::TensorOptions options_;
  ProcessGroup* tp_pg_;

  DEFINE_WEIGHT(w13);
  DEFINE_FUSED_WEIGHT(w1);
  DEFINE_FUSED_WEIGHT(w3);
  DEFINE_FUSED_WEIGHT(w2);
  DEFINE_FUSED_WEIGHT(w1_scale);
  DEFINE_FUSED_WEIGHT(w3_scale);
  DEFINE_WEIGHT(w13_scale);
  DEFINE_FUSED_WEIGHT(w2_scale);
  DEFINE_WEIGHT(e_score_correction_bias);

  void load_e_score_correction_bias(const StateDict& state_dict);
  void load_experts(const StateDict& state_dict);
};
TORCH_MODULE(FusedMoE);

}  // namespace layer
}  // namespace xllm
