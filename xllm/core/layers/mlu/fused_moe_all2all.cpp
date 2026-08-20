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

// FusedMoE All2All path implementation (DeepEP communication mode)
// This file contains the All2All path methods extracted from fused_moe.cpp

#include <glog/logging.h>

#include "fused_moe.h"
#include "kernels/ops_api.h"
#include "util/tensor_helper.h"

namespace xllm {
namespace layer {

void FusedMoEImpl::select_experts_all2all(
    const torch::Tensor& hidden_states_2d,
    const torch::Tensor& reduce_weight,
    const torch::Tensor& expert_id,
    SelectedExpertInfo& selected_expert_info) {
  // Step 1: generate expert ids (gate / routing already done by MoEGate)
  torch::Tensor gather_idx;
  torch::Tensor combine_idx;
  torch::Tensor token_count;
  {
    xllm::kernel::MoeGenIdxParams moe_gen_idx_params;
    moe_gen_idx_params.expert_id = expert_id;
    moe_gen_idx_params.expert_num = num_total_experts_;
    std::vector<torch::Tensor> output_vec =
        xllm::kernel::moe_gen_idx(moe_gen_idx_params);
    gather_idx = output_vec[0];
    combine_idx = output_vec[1];
    token_count = output_vec[2];
    // All2All path does not need cusum_token_count
    selected_expert_info.cusum_token_count = std::nullopt;
  }

  // Step 2: expand and quantize input if needed
  torch::Tensor expand_hidden_states;
  torch::Tensor hidden_states_scale;

  // All2All path: use full token count (no slicing)
  selected_expert_info.token_count_slice = token_count;

  int64_t num_token_expand = hidden_states_2d.size(0) * topk_;
  int64_t dispatch_bytes =
      num_token_expand * deep_ep_params_.dispatch_token_size;
  torch::Tensor dispatch_send_token_tensor =
      deep_ep_buffer_.dispatch_send_token_tensor.slice(0, 0, dispatch_bytes)
          .view({num_token_expand, deep_ep_params_.dispatch_token_size});

  if (is_smoothquant_) {
    xllm::kernel::ScaledQuantizeParams scaled_quantize_params;
    scaled_quantize_params.x = hidden_states_2d;
    // use dispatch_send_token_tensor buffer for input to reduce memory
    // footprint
    scaled_quantize_params.smooth = input_smooth_;
    scaled_quantize_params.output =
        dispatch_send_token_tensor.slice(1, 0, hidden_size_);
    scaled_quantize_params.token_count = selected_expert_info.token_count_slice;
    scaled_quantize_params.gather_index = gather_idx;
    scaled_quantize_params.act_mode = "none";
    scaled_quantize_params.active_coef = 1.0;
    scaled_quantize_params.is_gated = false;
    scaled_quantize_params.quant_type = torch::kChar;
    std::tie(expand_hidden_states, hidden_states_scale) =
        xllm::kernel::scaled_quantize(scaled_quantize_params);
    // since view_as_dtype has not supported stride yet,
    // we need to copy the scale output to the dispatch buffer
    torch::Tensor dispatch_scale_slice =
        dispatch_send_token_tensor.slice(1, hidden_size_);
    torch::Tensor hidden_states_scale_bytes =
        view_as_dtype(hidden_states_scale, torch::kInt8)
            .view_as(dispatch_scale_slice);
    dispatch_scale_slice.copy_(hidden_states_scale_bytes);
  } else {
    xllm::kernel::MoeExpandInputParams moe_expand_input_params;
    moe_expand_input_params.input = hidden_states_2d;
    moe_expand_input_params.gather_index = gather_idx;
    moe_expand_input_params.cusum_token_count = std::nullopt;
    moe_expand_input_params.start_expert_id = 0;
    moe_expand_input_params.expert_size = num_total_experts_;
    expand_hidden_states =
        xllm::kernel::moe_expand_input(moe_expand_input_params);
    // use copy to place the output inside the dispatch buffer
    torch::Tensor dispatch_tensor =
        view_as_dtype(expand_hidden_states, torch::kChar);
    dispatch_send_token_tensor.copy_(dispatch_tensor);
  }

  // collect the selected tensor
  selected_expert_info.reduce_weight = reduce_weight;
  selected_expert_info.combine_idx = combine_idx;
  if (is_smoothquant_) {
    selected_expert_info.input_scale = hidden_states_scale;
  }
}

FusedMoEImpl::CombineResult FusedMoEImpl::combine_step(
    const torch::Tensor& gemm2_out,
    const torch::Tensor& gather_by_rank_index,
    const torch::Tensor& token_sum,
    const torch::Tensor& hidden_states,
    int64_t num_token_expand) {
  torch::ScalarType hidden_states_dtype = hidden_states.dtype().toScalarType();

  // Delegate pack, layout generation and combine to DeepEP
  torch::Tensor combine_send_layout =
      deep_ep_->combine_step_pack(gemm2_out,
                                  gather_by_rank_index,
                                  token_sum,
                                  hidden_size_,
                                  hidden_states_dtype);

  // create a wait event for the current stream to finish computation
  auto current_stream = device_.current_stream();
  routed_stream_->wait_stream(*current_stream);

  // pure communication kernel: combine
  torch::Tensor output;
  {
    torch::StreamGuard stream_guard = routed_stream_->set_stream_guard();
    output = deep_ep_->combine_step_comm(combine_send_layout,
                                         num_token_expand,
                                         hidden_size_,
                                         hidden_states_dtype);
  }

  // pure computation kernel: shared experts
  torch::Tensor shared_expert_output;
  if (n_shared_experts_ > 0) {
    shared_stream_->wait_stream(*current_stream);
    torch::StreamGuard stream_guard = shared_stream_->set_stream_guard();
    shared_expert_output = shared_experts_(hidden_states);
  }

  // join for parallelization
  current_stream->wait_stream(*routed_stream_);
  if (n_shared_experts_ > 0) {
    current_stream->wait_stream(*shared_stream_);
  }

  return CombineResult{output, shared_expert_output};
}

torch::Tensor FusedMoEImpl::forward_experts_all2all(
    const torch::Tensor& hidden_states,
    const std::optional<RouteInfo>& route_info,
    const std::optional<torch::Tensor>& input_ids) {
  init_streams(hidden_states);

  torch::Tensor shared_expert_output;
  torch::IntArrayRef hidden_states_shape = hidden_states.sizes();
  torch::ScalarType hidden_states_dtype = hidden_states.dtype().toScalarType();
  torch::Tensor hidden_states_2d =
      hidden_states.reshape({-1, hidden_states.size(-1)});

  RouteInfo route = get_route(hidden_states_2d,
                              /*enable_all2all_communication=*/true,
                              route_info,
                              input_ids);

  int64_t group_gemm_max_dim = deep_ep_params_.max_num_tokens_recv / topk_;
  int64_t expert_size = w13_.size(0);

  SelectedExpertInfo selected_expert_info;
  select_experts_all2all(hidden_states_2d,
                         route.reduce_weight,
                         route.expert_id,
                         selected_expert_info);

  // Communication Step 1: Dispatch
  int64_t dispatch_token_num = hidden_states_2d.size(0) * topk_;

  // 1. Dispatch Step: Generate layout and send data
  deep_ep_->dispatch_step(dispatch_token_num,
                          selected_expert_info.token_count_slice);

  // 2. Process Result: Generate indices and unpack to computation buffer
  // use the buffer during initialization for the output
  torch::Tensor expand_hidden_states = dispatch_recv_token_tensor_head_;
  std::optional<torch::Tensor> output_tail = std::nullopt;
  if (is_smoothquant_) {
    output_tail = dispatch_recv_token_tensor_tail_;
    // update selected_expert_info with the tail (input scale)
    selected_expert_info.input_scale = output_tail;
  }

  DeepEPMetaResult deep_ep_meta = deep_ep_->process_dispatch_result(
      num_experts_per_rank_, expand_hidden_states, output_tail);

  // Extract metadata for subsequent steps
  torch::Tensor gather_by_rank_index = deep_ep_meta.gather_rank_index;
  selected_expert_info.token_count_slice = deep_ep_meta.token_count_slice;
  torch::Tensor token_sum = deep_ep_meta.token_sum;

  torch::Tensor gemm2_out =
      compute_routed_experts(std::move(expand_hidden_states),
                             hidden_states_dtype,
                             group_gemm_max_dim,
                             expert_size,
                             selected_expert_info);

  // Communication Step 2: Combine
  int64_t num_token_expand = hidden_states_2d.size(0) * topk_;
  CombineResult combine_result = combine_step(gemm2_out,
                                              gather_by_rank_index,
                                              token_sum,
                                              hidden_states,
                                              num_token_expand);
  gemm2_out = combine_result.output;
  shared_expert_output = combine_result.shared_expert_output;

  // Step 7: combine the intermediate results and get the final hidden states
  torch::Tensor final_hidden_states;
  // ensure the lifespan of these parameters via brace
  {
    xllm::kernel::MoeCombineResultParams moe_combine_result_params;
    moe_combine_result_params.input = gemm2_out;
    moe_combine_result_params.reduce_weight =
        selected_expert_info.reduce_weight;
    moe_combine_result_params.gather_ids = selected_expert_info.combine_idx;
    moe_combine_result_params.cusum_token_count =
        selected_expert_info.cusum_token_count;
    moe_combine_result_params.start_expert_id = start_expert_id_;
    moe_combine_result_params.expert_size = expert_size;
    moe_combine_result_params.bias = std::nullopt;
    // if all2all communication is enabled and shared output is provided,
    // we will fused the add up to combine result
    if (n_shared_experts_ > 0) {
      moe_combine_result_params.residual =
          shared_expert_output.reshape({-1, shared_expert_output.size(-1)});
    }

    final_hidden_states =
        xllm::kernel::moe_combine_result(moe_combine_result_params);
  }

  // reshape the final hidden states to the original shape
  final_hidden_states = final_hidden_states.reshape(hidden_states_shape);

  return final_hidden_states;
}

}  // namespace layer
}  // namespace xllm
