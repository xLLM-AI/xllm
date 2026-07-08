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

#include <optional>
#include <utility>

#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "layers/common/deep_ep.h"
#include "layers/common/dense_mlp.h"
#include "layers/common/fused_moe_base.h"
#include "layers/mlu/moe_gate.h"
#include "platform/device.h"
#include "util/tensor_helper.h"

namespace xllm {
namespace layer {

class FusedMoEImpl : public torch::nn::Module {
 public:
  struct RouteInfo {
    torch::Tensor reduce_weight;
    torch::Tensor expert_id;
  };

  FusedMoEImpl() = default;
  FusedMoEImpl(const ModelArgs& model_args,
               const FusedMoEArgs& moe_args,
               const QuantArgs& quant_args,
               const ParallelArgs& parallel_args,
               const torch::TensorOptions& options);

  RouteInfo prep_route(
      torch::Tensor& hidden_states,
      const std::optional<torch::Tensor>& input_ids = std::nullopt);
  torch::Tensor forward_experts(
      const torch::Tensor& hidden_states,
      bool enable_all2all_communication,
      const std::optional<RouteInfo>& route_info = std::nullopt,
      const std::optional<torch::Tensor>& input_ids = std::nullopt);
  torch::Tensor forward_shared(const torch::Tensor& hidden_states);
  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const ModelInputParams& input_params);
  virtual void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights() const;
  bool has_shared() const { return static_cast<bool>(shared_experts_); }
  void init_async(const torch::Tensor& hidden_states) {
    init_streams(hidden_states);
  }
  ProcessGroup* shared_pg() const { return shared_pg_; }
  Stream* shared_stream() const { return shared_stream_.get(); }
  Stream* routed_stream() const { return routed_stream_.get(); }

 protected:
  // struct to store the selected expert info
  struct SelectedExpertInfo {
    torch::Tensor reduce_weight;
    torch::Tensor combine_idx;
    torch::Tensor token_count_slice;
    std::optional<torch::Tensor> cusum_token_count;
    std::optional<torch::Tensor> input_scale;
  };

  // ===== Base path methods (AllReduce communication mode) =====
  torch::Tensor select_experts_base(const torch::Tensor& hidden_states_2d,
                                    const torch::Tensor& reduce_weight,
                                    const torch::Tensor& expert_id,
                                    SelectedExpertInfo& selected_expert_info);

  torch::Tensor forward_experts_base(
      const torch::Tensor& hidden_states,
      const std::optional<RouteInfo>& route_info,
      const std::optional<torch::Tensor>& input_ids);

  virtual void final_comm_allreduce(torch::Tensor& final_hidden_states,
                                    const torch::Tensor& hidden_states,
                                    torch::Tensor& shared_expert_output);

  // ===== All2All path methods (DeepEP communication mode) =====
  void select_experts_all2all(const torch::Tensor& hidden_states_2d,
                              const torch::Tensor& reduce_weight,
                              const torch::Tensor& expert_id,
                              SelectedExpertInfo& selected_expert_info);

  torch::Tensor forward_experts_all2all(
      const torch::Tensor& hidden_states,
      const std::optional<RouteInfo>& route_info,
      const std::optional<torch::Tensor>& input_ids);

  // Result structure for combine_step
  struct CombineResult {
    torch::Tensor output;
    torch::Tensor shared_expert_output;
  };

  CombineResult combine_step(const torch::Tensor& gemm2_out,
                             const torch::Tensor& gather_by_rank_index,
                             const torch::Tensor& token_sum,
                             const torch::Tensor& hidden_states,
                             int64_t num_token_expand);

  RouteInfo get_route(torch::Tensor& hidden_states_2d,
                      bool enable_all2all_communication,
                      const std::optional<RouteInfo>& route_info,
                      const std::optional<torch::Tensor>& input_ids);
  void check_route(const torch::Tensor& hidden_states_2d,
                   const RouteInfo& route_info) const;
  void init_streams(const torch::Tensor& hidden_states);
  torch::Tensor compute_routed_experts(
      torch::Tensor expand_hidden_states,
      torch::ScalarType hidden_states_dtype,
      int64_t group_gemm_max_dim,
      int64_t expert_size,
      SelectedExpertInfo& selected_expert_info);

 protected:
  int64_t num_total_experts_;
  int64_t topk_;
  int64_t hidden_size_;
  int64_t n_shared_experts_;
  bool is_gated_;
  bool enable_result_reduction_;
  std::string hidden_act_;
  bool is_smoothquant_;
  int64_t moe_weight_bits_ = 8;
  int64_t weight_pack_factor_ = 1;

  int64_t num_experts_per_rank_;
  int64_t start_expert_id_;

  // Deep EP related parameters
  bool enable_deep_ep_;
  DeepEPBuffer deep_ep_buffer_;
  DeepEPParams deep_ep_params_;
  torch::Tensor dispatch_recv_token_tensor_head_;
  torch::Tensor dispatch_recv_token_tensor_tail_;

  // streams for parallel shared experts
  std::unique_ptr<Stream> shared_stream_;
  std::unique_ptr<Stream> routed_stream_;
  xllm::Device device_;
  bool stream_initialized_ = false;
  bool use_hash_ = false;

  MoEGate gate_{nullptr};
  std::optional<float> swiglu_limit_;
  DenseMLP shared_experts_{nullptr};
  DeepEP deep_ep_{nullptr};

  QuantArgs quant_args_;
  ParallelArgs parallel_args_;
  torch::TensorOptions options_;
  ProcessGroup* tp_pg_;
  ProcessGroup* shared_pg_ = nullptr;

  DEFINE_WEIGHT(w13);
  DEFINE_FUSED_WEIGHT(w1);
  DEFINE_FUSED_WEIGHT(w3);
  DEFINE_FUSED_WEIGHT(w2);
  DEFINE_WEIGHT(w13_scale);
  DEFINE_FUSED_WEIGHT(w1_scale);
  DEFINE_FUSED_WEIGHT(w3_scale);
  DEFINE_FUSED_WEIGHT(w2_scale);
  DEFINE_FUSED_WEIGHT(input_smooth);
  DEFINE_FUSED_WEIGHT(act_smooth);

  torch::List<int64_t> w13_scale_quant_flag_;
  torch::List<int64_t> w2_scale_quant_flag_;
  bool scale_layout_prepared_ = false;

  void load_experts(const StateDict& state_dict);

  void prepare_scale_layout();

  // create the group gemm output tensor with the workspace
  torch::Tensor create_group_gemm_output(const torch::Tensor& a,
                                         const torch::Tensor& b,
                                         const torch::Tensor& group_list,
                                         torch::ScalarType dtype,
                                         torch::Tensor& workspace);
};
TORCH_MODULE(FusedMoE);

}  // namespace layer
}  // namespace xllm
