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

#include "fused_moe.h"

#include <glog/logging.h>

#include "common/global_flags.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/speculative_config.h"
#include "kernels/ops_api.h"
#include "layers/common/dp_utils.h"
#include "util/tensor_helper.h"
#include "util/utils.h"

namespace xllm {
namespace layer {

FusedMoEImpl::FusedMoEImpl(const ModelArgs& model_args,
                           const FusedMoEArgs& moe_args,
                           const QuantArgs& quant_args,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options)
    : num_total_experts_(model_args.n_routed_experts()),
      topk_(model_args.num_experts_per_tok()),
      hidden_size_(model_args.hidden_size()),
      n_shared_experts_(model_args.n_shared_experts()),
      is_gated_(moe_args.is_gated),
      enable_result_reduction_(moe_args.enable_result_reduction),
      hidden_act_(model_args.hidden_act()),
      quant_args_(quant_args),
      parallel_args_(parallel_args),
      options_(options),
      device_(options.device()) {
  const int64_t num_experts = num_total_experts_;
  const int64_t intermediate_size =
      static_cast<int64_t>(model_args.moe_intermediate_size());
  int64_t ep_size = parallel_args.ep_size();
  int64_t ep_rank = 0;
  tp_pg_ = parallel_args.tp_group_;
  if (ep_size > 1) {
    ep_rank = parallel_args.moe_ep_group_->rank();
    tp_pg_ = parallel_args.moe_tp_group_;
  }

  moe_weight_bits_ = quant_args.moe_weight_bits();
  weight_pack_factor_ = (moe_weight_bits_ == 4 ? 2 : 1);

  // smoothquant check: if quant_method is not empty, only smoothquant with
  // A8 and expert W8/W4 is supported.
  if (!quant_args.quant_method().empty()) {
    if (quant_args.quant_method() != kQuantMethodSmoothquant ||
        quant_args.bits() != 8 || !quant_args.activation_dynamic() ||
        (moe_weight_bits_ != 8 && moe_weight_bits_ != 4)) {
      LOG(FATAL)
          << "FusedMoE only supports the current SmoothQuant MoE path with "
             "non-expert bits=8, dynamic activation, and expert weight bits "
             "in {4,8}. "
          << "Got quant_method=" << quant_args.quant_method()
          << ", bits=" << quant_args.bits()
          << ", moe_weight_bits=" << moe_weight_bits_
          << ", activation_dynamic=" << quant_args.activation_dynamic();
    }
    is_smoothquant_ = true;
  } else {
    is_smoothquant_ = false;
  }

  // Deep EP initialization check
  enable_deep_ep_ =
      ::xllm::EPLBConfig::get_instance().expert_parallel_degree() == 2 &&
      ep_size > 1;
  if (enable_deep_ep_) {
    // for now, we only implement the deep ep for decode stage.
    // so we will assume the max_token_num is limited to max_batch_size * (1+K)
    // K is the number of speculative tokens.
    int64_t dispatch_token_size;
    if (quant_args.quant_method() == kQuantMethodSmoothquant) {
      // float32 is for the scale of the quantized input
      dispatch_token_size = hidden_size_ * get_dtype_size(torch::kInt8) +
                            get_dtype_size(torch::kFloat32);
    } else {
      dispatch_token_size =
          hidden_size_ * get_dtype_size(options_.dtype().toScalarType());
    }
    torch::ScalarType combine_dtype = options_.dtype().toScalarType();
    int64_t combine_token_size = hidden_size_ * get_dtype_size(combine_dtype);
    // Ensure calculation base is at least ep_size
    int64_t effective_seqs = std::max(
        (int64_t)::xllm::SchedulerConfig::get_instance().max_seqs_per_batch(),
        (int64_t)ep_size);
    // NOTE: ::xllm::SchedulerConfig::get_instance().max_seqs_per_batch()
    // represents the maximum total batch size, regardless of the dp size. To
    // ensure robust scheduling and account for the worst-case scenario, we must
    // guarantee that each rank is capable of handling the maximum possible
    // number of tokens. Therefore, we define max_num_tokens_per_rank as the
    // full maximum value, without dividing by either the rank count or the dp
    // size.
    int64_t max_num_tokens_per_rank =
        (1 +
         ::xllm::SpeculativeConfig::get_instance().num_speculative_tokens()) *
        effective_seqs * topk_;

    // make sure that all layers share the same deep ep instance
    //  so that the memory footprint is minimized
    deep_ep_ = DeepEPManager::get_instance(dispatch_token_size,
                                           combine_token_size,
                                           max_num_tokens_per_rank,
                                           num_experts,
                                           parallel_args,
                                           options_);

    // obtain the buffer and parameters of deep ep
    deep_ep_buffer_ = deep_ep_->get_buffer();
    deep_ep_params_ = deep_ep_->get_params();

    // intermediate buffer that can be initialized once
    // we place these tensor here in order to speed up forward pass
    int64_t n_tokens_recv = deep_ep_params_.max_num_tokens_recv;
    int64_t token_bytes = is_smoothquant_
                              ? get_dtype_size(torch::kInt8)
                              : get_dtype_size(options_.dtype().toScalarType());
    token_bytes = token_bytes * hidden_size_;
    int64_t head_size = n_tokens_recv * token_bytes;
    dispatch_recv_token_tensor_head_ =
        deep_ep_buffer_.combine_send_token_tensor.narrow(0, 0, head_size)
            .view({n_tokens_recv, token_bytes});
    // input scale in smoothquant
    if (is_smoothquant_) {
      int64_t tail_size = n_tokens_recv * get_dtype_size(torch::kFloat32);
      dispatch_recv_token_tensor_tail_ =
          deep_ep_buffer_.combine_send_token_tensor
              .narrow(0, head_size, tail_size)
              .view({n_tokens_recv, -1});
    }
  }

  // calculate the number of experts per rank
  num_experts_per_rank_ = num_experts / ep_size;
  start_expert_id_ = ep_rank * num_experts_per_rank_;

  gate_ = register_module("gate", MoEGate(model_args, quant_args, options));
  if (n_shared_experts_ > 0) {
    CHECK(parallel_args.single_rank_group_ != nullptr)
        << "shared experts require single_rank_group_";
    // Shared experts always run with full weights on the local rank. This
    // keeps the shared path independent from TP/EP sharding.
    shared_pg_ = parallel_args.single_rank_group_;
    shared_experts_ =
        register_module("shared_experts",
                        DenseMLP(hidden_size_,
                                 intermediate_size * n_shared_experts_,
                                 is_gated_,
                                 false,
                                 hidden_act_,
                                 enable_result_reduction_,
                                 quant_args,
                                 shared_pg_,
                                 options));
  }

  // create weight buffer
  const int64_t world_size = tp_pg_->world_size();
  int64_t local_intermediate_size = intermediate_size / world_size;
  const bool is_groupwise_scale = quant_args_.group_size() > 0;
  int64_t w13_scale_group_cols = 0;
  int64_t w2_scale_group_cols = 0;
  if (is_smoothquant_ && weight_pack_factor_ > 1) {
    CHECK_EQ(hidden_size_ % weight_pack_factor_, 0)
        << "hidden_size must be divisible by weight_pack_factor for W4 MoE. "
        << "hidden_size=" << hidden_size_
        << ", weight_pack_factor=" << weight_pack_factor_;
    CHECK_EQ(local_intermediate_size % weight_pack_factor_, 0)
        << "local_intermediate_size must be divisible by weight_pack_factor "
           "for W4 MoE. local_intermediate_size="
        << local_intermediate_size
        << ", weight_pack_factor=" << weight_pack_factor_;
  }
  if (is_smoothquant_ && is_groupwise_scale) {
    CHECK_GT(quant_args_.group_size(), 0)
        << "group_size must be positive for group-wise smoothquant.";
    CHECK_EQ(hidden_size_ % quant_args_.group_size(), 0)
        << "hidden_size must be divisible by group_size for group-wise "
           "smoothquant. hidden_size="
        << hidden_size_ << ", group_size=" << quant_args_.group_size();
    CHECK_EQ(local_intermediate_size % quant_args_.group_size(), 0)
        << "local_intermediate_size must be divisible by group_size for "
           "group-wise smoothquant. local_intermediate_size="
        << local_intermediate_size
        << ", group_size=" << quant_args_.group_size();
    w13_scale_group_cols = hidden_size_ / quant_args_.group_size();
    w2_scale_group_cols = local_intermediate_size / quant_args_.group_size();
  }
  if (is_smoothquant_) {
    // W4 expert qweight is stored as packed int4 in an int8 tensor container.
    auto quant_option = options_.dtype(torch::kInt8);
    auto fp_option = options_.dtype(torch::kFloat32);
    w13_ = register_parameter("w13",
                              torch::empty({num_experts_per_rank_,
                                            local_intermediate_size * 2,
                                            hidden_size_ / weight_pack_factor_},
                                           quant_option),
                              false);
    w13_scale_ = register_parameter(
        "w13_scale",
        is_groupwise_scale
            ? torch::empty({num_experts_per_rank_,
                            local_intermediate_size * 2,
                            w13_scale_group_cols},
                           fp_option)
            : torch::empty({num_experts_per_rank_, local_intermediate_size * 2},
                           fp_option),
        false);
    // Note: We do not check enable_deep_ep_ here, since smooth quantization
    // information may be needed even when deep EP mode is disabled. This allows
    // retrieving quantization parameters for any subset of experts as required.
    input_smooth_ = register_parameter(
        "input_smooth",
        torch::empty({num_total_experts_, hidden_size_}, fp_option),
        false);
    w2_ = register_parameter(
        "w2",
        torch::empty({num_experts_per_rank_,
                      hidden_size_,
                      local_intermediate_size / weight_pack_factor_},
                     quant_option),
        false);
    w2_scale_ = register_parameter(
        "w2_scale",
        is_groupwise_scale
            ? torch::empty(
                  {num_experts_per_rank_, hidden_size_, w2_scale_group_cols},
                  fp_option)
            : torch::empty({num_experts_per_rank_, hidden_size_}, fp_option),
        false);
    act_smooth_ = register_parameter(
        "act_smooth",
        torch::empty({num_experts_per_rank_, local_intermediate_size},
                     fp_option),
        false);

  } else {
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
}

std::pair<torch::Tensor, std::optional<torch::List<int64_t>>>
FusedMoEImpl::prepare_group_gemm_weight_scale(
    const torch::Tensor& b_scale) const {
  if (moe_weight_bits_ != 4 || b_scale.dim() != 3) {
    return {b_scale, std::nullopt};
  }

  torch::List<int64_t> quant_flag;
  const int64_t num_k_groups = b_scale.size(2);
  const int64_t num_experts = b_scale.size(0);
  for (int64_t expert_id = 0; expert_id < num_experts; ++expert_id) {
    for (int64_t group_id = 0; group_id < num_k_groups; ++group_id) {
      quant_flag.emplace_back(4);
    }
  }

  // torch_mlu_ops quant_flag path expects b_scale as [group_cols, experts, n].
  return {b_scale.permute({2, 0, 1}).contiguous(), quant_flag};
}

torch::Tensor FusedMoEImpl::create_group_gemm_output(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& group_list,
    torch::ScalarType dtype,
    torch::Tensor& workspace) {
  // unify shape logic: define the target shape once.
  bool is_3d_weight = (b.dim() != 2);
  int64_t num_tokens = a.size(0);
  int64_t out_dim = is_3d_weight ? b.size(1) : b.size(0);

  std::vector<int64_t> output_shape;
  int64_t required_elements = num_tokens * out_dim;

  if (is_3d_weight) {
    output_shape = {num_tokens, out_dim};
  } else {
    output_shape = {group_list.size(0), num_tokens, out_dim};
    required_elements *= group_list.size(0);
  }

  auto options = a.options().dtype(dtype);

  // non-smoothquant: direct allocation
  if (!is_smoothquant_) {
    return torch::empty(output_shape, options);
  }

  // smoothquant: managed workspace logic
  if (!workspace.defined()) {
    // Lazy initialization: allocate max buffer for the lifecycle
    // Note: accessing class members w13_ and w2_ directly for context
    int64_t max_width = std::max(w13_.size(1), w2_.size(1));
    workspace = torch::empty({num_tokens * max_width}, options);
  }

  // view construction
  CHECK(workspace.numel() >= required_elements)
      << "FusedMoE Workspace too small! Alloc: " << workspace.numel()
      << ", Req: " << required_elements;

  // utilize the pre-calculated output_shape
  return workspace.slice(0, 0, required_elements).view(output_shape);
}

void FusedMoEImpl::init_streams(const torch::Tensor& hidden_states) {
  if (stream_initialized_) {
    return;
  }

  device_ = xllm::Device(hidden_states.device());
  routed_stream_ = device_.get_stream_from_pool();
  shared_stream_ = device_.get_stream_from_pool();
  stream_initialized_ = true;
}

torch::Tensor FusedMoEImpl::compute_routed_experts(
    torch::Tensor expand_hidden_states,
    torch::ScalarType hidden_states_dtype,
    int64_t group_gemm_max_dim,
    int64_t expert_size,
    SelectedExpertInfo& selected_expert_info) {
  torch::Tensor gemm_workspace;

  torch::Tensor gemm1_out =
      create_group_gemm_output(expand_hidden_states,
                               w13_,
                               selected_expert_info.token_count_slice,
                               hidden_states_dtype,
                               gemm_workspace);
  {
    xllm::kernel::GroupGemmParams group_gemm_params;
    torch::ScalarType a_dtype =
        is_smoothquant_ ? torch::kInt8 : hidden_states_dtype;
    group_gemm_params.a =
        view_as_dtype(expand_hidden_states, a_dtype).view({-1, hidden_size_});
    group_gemm_params.b = w13_;
    group_gemm_params.token_count = selected_expert_info.token_count_slice;
    if (is_smoothquant_) {
      auto [b_scale, quant_flag] = prepare_group_gemm_weight_scale(w13_scale_);
      torch::Tensor a_scale =
          selected_expert_info.input_scale.value().flatten();
      selected_expert_info.input_scale =
          view_as_dtype(a_scale, torch::kFloat32);
      group_gemm_params.a_scale = selected_expert_info.input_scale;
      group_gemm_params.b_scale = b_scale;
      group_gemm_params.quant_flag = quant_flag;
    }
    group_gemm_params.max_dim = group_gemm_max_dim;
    group_gemm_params.trans_a = false;
    group_gemm_params.trans_b = true;
    group_gemm_params.a_quant_bit = is_smoothquant_ ? 8 : -1;
    group_gemm_params.output = gemm1_out;
    gemm1_out = xllm::kernel::group_gemm(group_gemm_params);
  }

  torch::Tensor act_out;
  torch::Tensor act_out_scale;
  if (is_smoothquant_) {
    int64_t slice_dim = gemm1_out.size(1);
    if (is_gated_) {
      slice_dim /= 2;
    }
    act_out = expand_hidden_states.slice(1, 0, slice_dim);
    act_out_scale =
        selected_expert_info.input_scale.value().slice(0, 0, gemm1_out.size(0));
    xllm::kernel::ScaledQuantizeParams scaled_quantize_params;
    scaled_quantize_params.x = gemm1_out;
    scaled_quantize_params.smooth = act_smooth_;
    scaled_quantize_params.token_count = selected_expert_info.token_count_slice;
    scaled_quantize_params.output = act_out;
    scaled_quantize_params.output_scale = act_out_scale;
    scaled_quantize_params.act_mode = hidden_act_;
    scaled_quantize_params.active_coef = 1.0;
    scaled_quantize_params.is_gated = is_gated_;
    scaled_quantize_params.quant_type = torch::kChar;
    std::tie(act_out, act_out_scale) =
        xllm::kernel::scaled_quantize(scaled_quantize_params);
  } else {
    act_out =
        is_gated_ ? gemm1_out.slice(1, 0, gemm1_out.size(1) / 2) : gemm1_out;
    xllm::kernel::ActivationParams activation_params;
    activation_params.input = gemm1_out;
    activation_params.output = act_out;
    activation_params.cusum_token_count =
        selected_expert_info.cusum_token_count;
    activation_params.act_mode = hidden_act_;
    activation_params.is_gated = is_gated_;
    activation_params.start_expert_id = start_expert_id_;
    activation_params.expert_size = expert_size;
    xllm::kernel::active(activation_params);
  }

  torch::Tensor gemm2_out =
      create_group_gemm_output(act_out,
                               w2_,
                               selected_expert_info.token_count_slice,
                               hidden_states_dtype,
                               gemm_workspace);
  {
    xllm::kernel::GroupGemmParams group_gemm_params;
    group_gemm_params.a = act_out;
    group_gemm_params.b = w2_;
    group_gemm_params.token_count = selected_expert_info.token_count_slice;
    if (is_smoothquant_) {
      auto [b_scale, quant_flag] = prepare_group_gemm_weight_scale(w2_scale_);
      group_gemm_params.a_scale = act_out_scale;
      group_gemm_params.b_scale = b_scale;
      group_gemm_params.quant_flag = quant_flag;
    }
    group_gemm_params.max_dim = group_gemm_max_dim;
    group_gemm_params.trans_a = false;
    group_gemm_params.trans_b = true;
    group_gemm_params.a_quant_bit = is_smoothquant_ ? 8 : -1;
    group_gemm_params.output = gemm2_out;
    gemm2_out = xllm::kernel::group_gemm(group_gemm_params);
  }

  expand_hidden_states = torch::Tensor();
  selected_expert_info.input_scale = std::nullopt;
  act_out = torch::Tensor();
  return gemm2_out;
}

FusedMoEImpl::RouteInfo FusedMoEImpl::prep_route(torch::Tensor& hidden_states) {
  torch::Tensor hidden_states_2d =
      hidden_states.reshape({-1, hidden_states.size(-1)});

  RouteInfo route_info;
  std::tie(route_info.reduce_weight, route_info.expert_id) =
      gate_->forward(hidden_states_2d);
  return route_info;
}

torch::Tensor FusedMoEImpl::forward_shared(const torch::Tensor& hidden_states) {
  if (!shared_experts_) {
    return torch::Tensor();
  }
  return shared_experts_(hidden_states);
}

void FusedMoEImpl::check_route(const torch::Tensor& hidden_states_2d,
                               const RouteInfo& route_info) const {
  CHECK(route_info.reduce_weight.defined())
      << "route_info.reduce_weight must be defined";
  CHECK(route_info.expert_id.defined())
      << "route_info.expert_id must be defined";
  CHECK_EQ(route_info.reduce_weight.dim(), 2)
      << "route_info.reduce_weight must be 2D";
  CHECK_EQ(route_info.expert_id.dim(), 2) << "route_info.expert_id must be 2D";
  CHECK_EQ(route_info.reduce_weight.size(0), hidden_states_2d.size(0))
      << "route_info.reduce_weight token count mismatch";
  CHECK_EQ(route_info.expert_id.size(0), hidden_states_2d.size(0))
      << "route_info.expert_id token count mismatch";
  CHECK_EQ(route_info.reduce_weight.size(1), topk_)
      << "route_info.reduce_weight topk mismatch";
  CHECK_EQ(route_info.expert_id.size(1), topk_)
      << "route_info.expert_id topk mismatch";
  CHECK(route_info.reduce_weight.device() == hidden_states_2d.device())
      << "route_info.reduce_weight device mismatch";
  CHECK(route_info.expert_id.device() == hidden_states_2d.device())
      << "route_info.expert_id device mismatch";
  auto expert_dtype = route_info.expert_id.scalar_type();
  CHECK(expert_dtype == torch::kInt || expert_dtype == torch::kLong)
      << "route_info.expert_id must be int32 or int64";
}

FusedMoEImpl::RouteInfo FusedMoEImpl::get_route(
    torch::Tensor& hidden_states_2d,
    bool enable_all2all_communication,
    const std::optional<RouteInfo>& route_info) {
  if (!route_info.has_value()) {
    return prep_route(hidden_states_2d);
  }

  CHECK(!enable_all2all_communication)
      << "all2all path does not support external route_info";
  check_route(hidden_states_2d, route_info.value());
  return route_info.value();
}

torch::Tensor FusedMoEImpl::forward_experts(
    const torch::Tensor& hidden_states,
    bool enable_all2all_communication,
    const std::optional<RouteInfo>& route_info) {
  // Dispatcher: route to the appropriate path based on communication mode
  if (enable_all2all_communication) {
    return forward_experts_all2all(hidden_states, route_info);
  } else {
    return forward_experts_base(hidden_states, route_info);
  }
}

torch::Tensor FusedMoEImpl::forward(const torch::Tensor& hidden_states,
                                    const ModelInputParams& input_params) {
  // we only support all2all communication for decode stage for now
  bool enable_all2all_communication =
      enable_deep_ep_ && all_dp_ranks_are_decode(input_params);
  return forward_experts(
      hidden_states, enable_all2all_communication, std::nullopt);
}

void FusedMoEImpl::load_experts(const StateDict& state_dict) {
  const int64_t rank = tp_pg_->rank();
  const int64_t world_size = tp_pg_->world_size();
  const int64_t start_expert_id = start_expert_id_;
  const int64_t num_experts_per_rank = num_experts_per_rank_;
  const int64_t num_total_experts = num_total_experts_;
  std::vector<std::string> prefixes = {"gate_proj.", "up_proj."};
  if (is_smoothquant_) {
    LOAD_MOE_FUSED_WEIGHT("qweight", w1, w3, w13);
    LOAD_MOE_FUSED_WEIGHT("per_channel_scale", w1_scale, w3_scale, w13_scale);
    // When supporting DeepEP All2All mode,
    // we need to load the complete set of expert weights corresponding to
    // "up_proj.smooth". Note that even if deep EP mode is not enabled, it
    // remains possible to retrieve the smooth quantization information for a
    // subset of experts. Therefore, we intentionally do not check whether
    // deep_ep_ is enabled in this case.
    LOAD_MOE_ALL_EXPERT_WEIGHT("up_proj.", "smooth", input_smooth, -1);
    LOAD_MOE_WEIGHT("down_proj.", "qweight", w2, 1);
    LOAD_MOE_WEIGHT("down_proj.", "per_channel_scale", w2_scale, -1);
    LOAD_MOE_WEIGHT("down_proj.", "smooth", act_smooth, 0);
  } else {
    LOAD_MOE_FUSED_WEIGHT("weight", w1, w3, w13);
    LOAD_MOE_WEIGHT("down_proj.", "weight", w2, 1);
  }
}

void FusedMoEImpl::verify_loaded_weights() const {
  if (!is_smoothquant_) {
    return;
  }
  CHECK(w13_is_loaded_)
      << "SmoothQuant is enabled but expert gate_proj/up_proj qweight was not "
         "fully loaded. This usually means the checkpoint is missing required "
         "expert qweight tensors.";
  CHECK(w2_is_loaded_)
      << "SmoothQuant is enabled but expert down_proj qweight was not fully "
         "loaded. This usually means the checkpoint is missing required "
         "expert qweight tensors.";
}

void FusedMoEImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }

  if (n_shared_experts_ > 0) {
    shared_experts_->load_state_dict(
        state_dict.get_dict_with_prefix("shared_experts."));
  }
  gate_->load_state_dict(state_dict.get_dict_with_prefix("gate."));
  load_experts(state_dict.get_dict_with_prefix("experts."));
}

}  // namespace layer
}  // namespace xllm
