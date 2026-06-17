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

#include <iomanip>

#include "common/global_flags.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/speculative_config.h"
#include "framework/parallel_state/parallel_state.h"
#include "kernels/ops_api.h"
#include "layers/common/dp_utils.h"
#include "util/utils.h"

namespace {

int32_t get_dtype_size(torch::ScalarType dtype) {
  return static_cast<int32_t>(torch::elementSize(dtype));
}

}  // namespace

namespace xllm {
namespace layer {

FusedMoEImpl::FusedMoEImpl(const ModelArgs& model_args,
                           const FusedMoEArgs& moe_args,
                           const QuantArgs& quant_args,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options)
    : num_total_experts_(static_cast<int64_t>(model_args.n_routed_experts())),
      topk_(model_args.num_experts_per_tok()),
      num_expert_group_(model_args.n_group()),
      topk_group_(model_args.topk_group()),
      route_scale_(model_args.routed_scaling_factor()),
      hidden_size_(model_args.hidden_size()),
      n_shared_experts_(model_args.n_shared_experts()),
      is_gated_(moe_args.is_gated),
      renormalize_(model_args.norm_topk_prob() ? 1 : 0),
      hidden_act_(model_args.hidden_act()),
      scoring_func_(model_args.scoring_func()),
      quant_args_(quant_args),
      parallel_args_(parallel_args),
      options_(options),
      device_(options.device()) {
  const int64_t num_experts = num_total_experts_;
  const int64_t intermediate_size =
      static_cast<int64_t>(model_args.moe_intermediate_size());
  const std::string& topk_method = model_args.topk_method();
  int64_t ep_size = parallel_args.ep_size();
  int64_t ep_rank = 0;
  tp_pg_ = parallel_args.tp_group_;
  if (ep_size > 1) {
    ep_rank = parallel_args.moe_ep_group_->rank();
    tp_pg_ = parallel_args.moe_tp_group_;
  }

  // smoothquant check: If quant_method is not empty, only w8a8 smoothquant is
  // supported
  if (!quant_args.quant_method().empty()) {
    if (quant_args.quant_method() != "smoothquant" || quant_args.bits() != 8 ||
        !quant_args.activation_dynamic()) {
      LOG(FATAL) << "FusedMoE only supports w8a8 smoothquant quantization when "
                    "quant_method is set. "
                 << "Got quant_method=" << quant_args.quant_method()
                 << ", bits=" << quant_args.bits()
                 << ", activation_dynamic=" << quant_args.activation_dynamic();
    }
    // If confirmed as smoothquant w8a8, set is_smoothquant_ to true
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
    if (quant_args.quant_method() == "smoothquant") {
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

  if (topk_method == "noaux_tc") {
    e_score_correction_bias_ = register_parameter(
        "e_score_correction_bias", torch::empty({num_experts}, options), false);
  }

  gate_ = register_module(
      "gate_proj",
      ReplicatedLinear(hidden_size_, num_experts, false, quant_args, options));
  if (n_shared_experts_ > 0) {
    ProcessGroup* shared_expert_pg;
    if (parallel_args_.ep_size() > 1) {
      // we use tp=1 for shared experts computation in deep ep mode
      CHECK(parallel_args_.ep_size() == parallel_args_.world_size())
          << "Models with shared experts only support ep_size equal to "
             "world size for now.";
      shared_expert_pg = parallel_args.moe_tp_group_;
    } else {
      shared_expert_pg = parallel_args.process_group_;
    }
    // The shared experts computation can proceed in parallel with the
    // final communication step during the MoE computation, as long as it
    // remains independent of any communication operations. For optimal
    // performance, ensure that the shared experts layer on each rank always
    // maintains its own unique weights.
    shared_experts_ =
        register_module("shared_experts",
                        DenseMLP(hidden_size_,
                                 intermediate_size * n_shared_experts_,
                                 is_gated_,
                                 false,
                                 hidden_act_,
                                 /*enable_result_reduction=*/true,
                                 quant_args,
                                 shared_expert_pg,
                                 options));
  }

  // create weight buffer
  const int64_t world_size = tp_pg_->world_size();
  int64_t local_intermediate_size = intermediate_size / world_size;
  if (is_smoothquant_) {
    auto quant_option = options_.dtype(torch::kInt8);
    auto fp_option = options_.dtype(torch::kFloat32);
    w13_ = register_parameter(
        "w13",
        torch::empty(
            {num_experts_per_rank_, local_intermediate_size * 2, hidden_size_},
            quant_option),
        false);
    w13_scale_ = register_parameter(
        "w13_scale",
        torch::empty({num_experts_per_rank_, local_intermediate_size * 2},
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
        torch::empty(
            {num_experts_per_rank_, hidden_size_, local_intermediate_size},
            quant_option),
        false);
    w2_scale_ = register_parameter(
        "w2_scale",
        torch::empty({num_experts_per_rank_, hidden_size_}, fp_option),
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

torch::Tensor FusedMoEImpl::select_experts(
    const torch::Tensor& hidden_states_2d,
    const torch::Tensor& router_logits_2d,
    SelectedExpertInfo& selected_expert_info,
    bool enable_all2all_communication) {
  // prepare the parameters for select_experts
  std::optional<torch::Tensor> e_score_correction_bias = std::nullopt;
  if (e_score_correction_bias_.defined()) {
    e_score_correction_bias = e_score_correction_bias_;
  }
  int64_t expert_size = w13_.size(0);

  // Step 1: apply softmax topk or sigmoid topk / routing logic
  torch::Tensor reduce_weight;
  torch::Tensor expert_id;
  {
    xllm::kernel::MoeFusedTopkParams moe_active_topk_params;
    moe_active_topk_params.input = router_logits_2d;
    moe_active_topk_params.topk = topk_;
    moe_active_topk_params.num_expert_group = num_expert_group_;
    moe_active_topk_params.topk_group = topk_group_;
    moe_active_topk_params.normalize = renormalize_;
    moe_active_topk_params.normed_by = "topk_logit";
    moe_active_topk_params.scoring_func = scoring_func_;
    moe_active_topk_params.route_scale = route_scale_;
    moe_active_topk_params.e_score_correction_bias = e_score_correction_bias;
    std::tie(reduce_weight, expert_id) =
        xllm::kernel::moe_active_topk(moe_active_topk_params);
  }

  // Step 2: generate expert ids
  torch::Tensor gather_idx;
  torch::Tensor combine_idx;
  torch::Tensor token_count;
  std::optional<torch::Tensor> cusum_token_count;
  {
    xllm::kernel::MoeGenIdxParams moe_gen_idx_params;
    moe_gen_idx_params.expert_id = expert_id;
    moe_gen_idx_params.expert_num = num_total_experts_;
    std::vector<torch::Tensor> output_vec =
        xllm::kernel::moe_gen_idx(moe_gen_idx_params);
    gather_idx = output_vec[0];
    combine_idx = output_vec[1];
    token_count = output_vec[2];
    // during all2all communication, we do not need cusum_token_count in the
    // following computation
    if (enable_all2all_communication) {
      cusum_token_count = std::nullopt;
    } else {
      cusum_token_count = output_vec[3];
    }
  }

  // Step 3: expand and quantize input if needed
  torch::Tensor expand_hidden_states;
  torch::Tensor hidden_states_scale;
  torch::Tensor token_count_slice;
  // all2all related variables
  torch::Tensor dispatch_send_token_tensor;
  // in all2all, the input is scattered, so there is no need to slice the token
  // count, and we can use the dispatch buffer directly
  if (enable_all2all_communication) {
    token_count_slice = token_count;
    int64_t num_token_expand = hidden_states_2d.size(0) * topk_;
    int64_t dispatch_bytes =
        num_token_expand * deep_ep_params_.dispatch_token_size;
    dispatch_send_token_tensor =
        deep_ep_buffer_.dispatch_send_token_tensor.slice(0, 0, dispatch_bytes)
            .view({num_token_expand, deep_ep_params_.dispatch_token_size});
  } else {
    token_count_slice =
        token_count.slice(0, start_expert_id_, start_expert_id_ + expert_size);
  }

  if (is_smoothquant_) {
    xllm::kernel::ScaledQuantizeParams scaled_quantize_params;
    scaled_quantize_params.x = hidden_states_2d;
    // use dispatch_send_token_tensor buffer for input
    //  to reduce memory footprint
    if (enable_all2all_communication) {
      scaled_quantize_params.smooth = input_smooth_;
      scaled_quantize_params.output =
          dispatch_send_token_tensor.slice(1, 0, hidden_size_);
    } else {
      scaled_quantize_params.smooth = input_smooth_.slice(
          0, start_expert_id_, start_expert_id_ + expert_size);
      scaled_quantize_params.gather_index_start_position =
          cusum_token_count.value().index({start_expert_id_}).unsqueeze(0);
    }
    scaled_quantize_params.token_count = token_count_slice;
    scaled_quantize_params.gather_index = gather_idx;
    scaled_quantize_params.act_mode = "none";
    scaled_quantize_params.active_coef = 1.0;
    scaled_quantize_params.is_gated = false;
    scaled_quantize_params.quant_type = torch::kChar;
    std::tie(expand_hidden_states, hidden_states_scale) =
        xllm::kernel::scaled_quantize(scaled_quantize_params);
    if (enable_all2all_communication) {
      // since view_as_dtype has not supported stride yet,
      //  we need to copy the scale output to the dispatch buffer
      torch::Tensor dispatch_scale_slice =
          dispatch_send_token_tensor.slice(1, hidden_size_);
      torch::Tensor hidden_states_scale_bytes =
          view_as_dtype(hidden_states_scale, torch::kInt8)
              .view_as(dispatch_scale_slice);
      dispatch_scale_slice.copy_(hidden_states_scale_bytes);
    }
  } else {
    xllm::kernel::MoeExpandInputParams moe_expand_input_params;
    moe_expand_input_params.input = hidden_states_2d;
    moe_expand_input_params.gather_index = gather_idx;
    moe_expand_input_params.combine_idx = combine_idx;
    moe_expand_input_params.topk = topk_;
    expand_hidden_states =
        xllm::kernel::moe_expand_input(moe_expand_input_params);
    if (enable_all2all_communication) {
      // use copy to place the output inside the dispatch buffer
      torch::Tensor dispatch_tensor =
          view_as_dtype(expand_hidden_states, torch::kChar);
      dispatch_send_token_tensor.copy_(dispatch_tensor);
    }
  }

  // collect the selected tensor
  selected_expert_info.reduce_weight = reduce_weight;
  selected_expert_info.combine_idx = combine_idx;
  selected_expert_info.token_count_slice = token_count_slice;
  selected_expert_info.cusum_token_count = cusum_token_count;
  if (is_smoothquant_) {
    selected_expert_info.input_scale = hidden_states_scale;
  }

  return expand_hidden_states;
}

torch::Tensor FusedMoEImpl::forward_experts(const torch::Tensor& hidden_states,
                                            const torch::Tensor& router_logits,
                                            bool enable_all2all_communication) {
  if (!stream_initialized_) {
    // update device record
    device_ = xllm::Device(hidden_states.device());

    // acquire streams from the pool again
    routed_stream_ = device_.get_stream_from_pool();
    shared_stream_ = device_.get_stream_from_pool();
    stream_initialized_ = true;
  }

  std::optional<torch::Tensor> e_score_correction_bias = std::nullopt;
  if (e_score_correction_bias_.defined()) {
    e_score_correction_bias = e_score_correction_bias_;
  }

  // prepare the parameters for MoE computation
  torch::Tensor shared_expert_output;
  torch::IntArrayRef hidden_states_shape = hidden_states.sizes();
  torch::ScalarType hidden_states_dtype = hidden_states.dtype().toScalarType();
  torch::Tensor hidden_states_2d =
      hidden_states.reshape({-1, hidden_states.size(-1)});
  torch::Tensor router_logits_2d =
      router_logits.reshape({-1, router_logits.size(-1)});
  int64_t group_gemm_max_dim = enable_all2all_communication
                                   ? deep_ep_params_.max_num_tokens_recv / topk_
                                   : hidden_states_2d.size(0);
  int64_t expert_size = w13_.size(0);

  // Step 1-3: select experts
  SelectedExpertInfo selected_expert_info;
  torch::Tensor expand_hidden_states =
      select_experts(hidden_states_2d,
                     router_logits_2d,
                     selected_expert_info,
                     enable_all2all_communication);

  // Communciation Step 1: Dipatch
  // intermediate outputs that are used both in dispatch and combine
  torch::Tensor gather_by_rank_index;
  torch::Tensor token_sum;
  if (enable_all2all_communication) {
    int64_t dispatch_token_num = hidden_states_2d.size(0) * topk_;

    // 1. Dispatch Step: Generate layout and send data
    deep_ep_->dispatch_step(dispatch_token_num,
                            selected_expert_info.token_count_slice);

    // 2. Process Result: Generate indices and unpack to computation buffer
    // use the buffer during initialization for the output
    expand_hidden_states = dispatch_recv_token_tensor_head_;
    std::optional<torch::Tensor> output_tail = std::nullopt;
    if (is_smoothquant_) {
      output_tail = dispatch_recv_token_tensor_tail_;
      // update selected_expert_info with the tail (input scale)
      selected_expert_info.input_scale = output_tail;
    }

    DeepEPMetaResult deep_ep_meta = deep_ep_->process_dispatch_result(
        num_experts_per_rank_, expand_hidden_states, output_tail);

    // Extract metadata for subsequent steps
    gather_by_rank_index = deep_ep_meta.gather_rank_index;
    selected_expert_info.token_count_slice = deep_ep_meta.token_count_slice;
    token_sum = deep_ep_meta.token_sum;
  }

  // common gemm workspace for reduce memory footprint
  torch::Tensor gemm_workspace;

  // Step 4: group gemm 1
  torch::Tensor gemm1_out =
      create_group_gemm_output(expand_hidden_states,
                               w13_,
                               selected_expert_info.token_count_slice,
                               hidden_states_dtype,
                               gemm_workspace);
  // ensure the lifespan of these parameters via brace
  {
    xllm::kernel::GroupGemmParams group_gemm_params;
    torch::ScalarType a_dtype =
        is_smoothquant_ ? torch::kInt8 : hidden_states_dtype;
    group_gemm_params.a =
        view_as_dtype(expand_hidden_states, a_dtype).view({-1, hidden_size_});
    group_gemm_params.b = w13_;
    group_gemm_params.token_count =
        selected_expert_info.token_count_slice.to("cpu");
    if (is_smoothquant_) {
      torch::Tensor a_scale =
          selected_expert_info.input_scale.value().flatten();
      selected_expert_info.input_scale =
          view_as_dtype(a_scale, torch::kFloat32);
      group_gemm_params.a_scale = selected_expert_info.input_scale;
      group_gemm_params.b_scale = w13_scale_;
    }
    group_gemm_params.max_dim = group_gemm_max_dim;
    group_gemm_params.trans_a = false;
    group_gemm_params.trans_b = true;
    group_gemm_params.a_quant_bit = is_smoothquant_ ? 8 : -1;
    group_gemm_params.output = gemm1_out;
    group_gemm_params.combine_idx = std::nullopt;
    gemm1_out = xllm::kernel::group_gemm(group_gemm_params);
  }

  // Step 5: activation or scaled quantization(fused with activation)
  torch::Tensor act_out;
  torch::Tensor act_out_scale;
  if (is_smoothquant_) {
    int64_t slice_dim = gemm1_out.size(1);
    if (is_gated_) slice_dim /= 2;
    // slice operation is a view, does not take up extra memory, but points to
    // the same memory
    act_out = expand_hidden_states.slice(1, 0, slice_dim);
    act_out_scale =
        selected_expert_info.input_scale.value().slice(0, 0, gemm1_out.size(0));
    // call scaled quantization kernel (also fused with activation)
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
    act_out = is_gated_
                  ? gemm1_out.slice(1, 0, gemm1_out.size(1) / 2).contiguous()
                  : gemm1_out;
    // call activation kernel
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

  // Step 6: group gemm 2
  torch::Tensor gemm2_out =
      create_group_gemm_output(act_out,
                               w2_,
                               selected_expert_info.token_count_slice,
                               hidden_states_dtype,
                               gemm_workspace);
  // ensure the lifespan of these parameters via brace
  {
    xllm::kernel::GroupGemmParams group_gemm_params;
    group_gemm_params.a = act_out;
    group_gemm_params.b = w2_;
    group_gemm_params.token_count =
        selected_expert_info.token_count_slice.to("cpu");
    if (is_smoothquant_) {
      group_gemm_params.a_scale = act_out_scale;
      group_gemm_params.b_scale = w2_scale_;
    }
    group_gemm_params.max_dim = group_gemm_max_dim;
    group_gemm_params.trans_a = false;
    group_gemm_params.trans_b = true;
    group_gemm_params.a_quant_bit = is_smoothquant_ ? 8 : -1;
    group_gemm_params.output = gemm2_out;
    group_gemm_params.combine_idx = selected_expert_info.combine_idx;
    gemm2_out = xllm::kernel::group_gemm(group_gemm_params);
  }

  // Communciation Step 2: Combine
  if (enable_all2all_communication) {
    int64_t num_token_expand = hidden_states_2d.size(0) * topk_;
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
    // pure communciation kernel: dispatch
    {
      torch::StreamGuard stream_guard = routed_stream_->set_stream_guard();
      gemm2_out = deep_ep_->combine_step_comm(combine_send_layout,
                                              num_token_expand,
                                              hidden_size_,
                                              hidden_states_dtype);
    }

    // pure computation kernel: shared experts
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
  }

  // After group gemm is finished, some tensors are no
  // longer needed. We must explicitly release the memory.
  expand_hidden_states = torch::Tensor();
  selected_expert_info.input_scale = std::nullopt;
  act_out = torch::Tensor();

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
    //  we will fused the add up to combine result
    if (enable_all2all_communication && n_shared_experts_ > 0) {
      moe_combine_result_params.residual =
          shared_expert_output.reshape({-1, shared_expert_output.size(-1)});
    }

    final_hidden_states =
        xllm::kernel::moe_combine_result(moe_combine_result_params);
  }

  // reshape the final hidden states to the original shape
  final_hidden_states = final_hidden_states.reshape(hidden_states_shape);

  if (enable_all2all_communication) {
    return final_hidden_states;
  }

  // Communciation Step 3: AllReduce for non-all2all communication
  // shared experts can be parallelized with the final communication step
  // during moe computation.
  auto current_stream = device_.current_stream();
  routed_stream_->wait_stream(*current_stream);
  {
    torch::StreamGuard stream_guard = routed_stream_->set_stream_guard();
    if (tp_pg_->world_size() > 1) {
      final_hidden_states = parallel_state::reduce(final_hidden_states, tp_pg_);
    }
    if (parallel_args_.ep_size() > 1) {
      final_hidden_states = parallel_state::reduce(
          final_hidden_states, parallel_args_.moe_ep_group_);
    }
  }

  if (n_shared_experts_ > 0) {
    shared_stream_->wait_stream(*current_stream);
    torch::StreamGuard stream_guard = shared_stream_->set_stream_guard();
    // for non all2all, we compute the shared experts parallelized with the
    // final communication step
    shared_expert_output = shared_experts_(hidden_states);
    shared_expert_output =
        shared_expert_output.reshape({-1, shared_expert_output.size(-1)});
  }

  // join for parallelization
  current_stream->wait_stream(*routed_stream_);
  if (n_shared_experts_ > 0) {
    current_stream->wait_stream(*shared_stream_);
    final_hidden_states += shared_expert_output;
  }

  return final_hidden_states;
}

torch::Tensor FusedMoEImpl::forward(const torch::Tensor& hidden_states,
                                    const ModelInputParams& input_params) {
  // we only support all2all communication for decode stage for now
  bool enable_all2all_communication =
      enable_deep_ep_ && std::all_of(input_params.parallel.dp_is_decode.begin(),
                                     input_params.parallel.dp_is_decode.end(),
                                     [](int32_t val) { return val == 1; });

  bool is_dp_ep_parallel =
      parallel_args_.dp_size() > 1 && parallel_args_.ep_size() > 1;
  // during all2all communication, the output has been
  //  gathered and sliced by dispatch and combine steps,
  //  so we do not need to gather input and slice output again
  bool need_gather_and_slice =
      is_dp_ep_parallel && !enable_all2all_communication;

  auto input = hidden_states;
  if (need_gather_and_slice) {
    input = parallel_state::gather(input,
                                   parallel_args_.dp_local_process_group_,
                                   input_params.parallel.dp_global_token_nums);
  }
  // MoE Gate
  auto router_logits = gate_(input);

  // MoE Experts
  auto output =
      forward_experts(input, router_logits, enable_all2all_communication);

  if (need_gather_and_slice) {
    output = get_dp_local_slice(output, input_params, parallel_args_);
  }

  return output;
}

void FusedMoEImpl::load_e_score_correction_bias(const StateDict& state_dict) {
  if (e_score_correction_bias_.defined() &&
      !e_score_correction_bias_is_loaded_) {
    LOAD_WEIGHT(e_score_correction_bias);
  }
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

void FusedMoEImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }

  if (n_shared_experts_ > 0) {
    shared_experts_->load_state_dict(
        state_dict.get_dict_with_prefix("shared_experts."));
  }
  gate_->load_state_dict(state_dict.get_dict_with_prefix("gate."));
  load_e_score_correction_bias(state_dict.get_dict_with_prefix("gate."));
  load_experts(state_dict.get_dict_with_prefix("experts."));
}

}  // namespace layer
}  // namespace xllm
