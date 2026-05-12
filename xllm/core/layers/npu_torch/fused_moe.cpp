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

#include "fused_moe.h"

#include <glog/logging.h>

#include <numeric>
#include <vector>

#include "common/flash_comm1_context.h"
#include "framework/parallel_state/parallel_state.h"
#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

namespace {
// Generic local tensor helpers.
torch::Tensor create_group_gemm_output(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& group_list,
    torch::ScalarType dtype = torch::ScalarType::BFloat16) {
  torch::TensorOptions target_options = a.options().dtype(dtype);
  if (b.dim() != 2) {
    return torch::empty({a.size(0), b.size(1)}, target_options);
  }
  return torch::empty({group_list.size(0), a.size(0), b.size(0)},
                      target_options);
}

torch::Tensor get_tensor_with_weight_suffix(const StateDict& state_dict,
                                            const std::string& tensor_name) {
  auto tensor = state_dict.get_tensor(tensor_name);
  if (!tensor.defined()) {
    tensor = state_dict.get_tensor(tensor_name + ".weight");
  }
  return tensor;
}

torch::Tensor slice_expert_weights(const torch::Tensor& weight,
                                   int64_t start_expert_id,
                                   int64_t num_experts_per_rank) {
  return weight
      .slice(0, start_expert_id, start_expert_id + num_experts_per_rank)
      .contiguous();
}

// Qwen3.5-MoE fused checkpoint fallback helpers.
bool load_fused_gate_up_fallback(const StateDict& state_dict,
                                 int64_t rank,
                                 int64_t world_size,
                                 int64_t start_expert_id,
                                 int64_t num_experts_per_rank,
                                 torch::Tensor& w13) {
  auto fused_gate_up =
      get_tensor_with_weight_suffix(state_dict, "gate_up_proj");
  if (!fused_gate_up.defined()) {
    return false;
  }

  if (world_size > 1) {
    CHECK_EQ(fused_gate_up.size(1) % 2, 0)
        << "gate_up_proj dim1 must be even, got " << fused_gate_up.size(1);
    const int64_t full_intermediate = fused_gate_up.size(1) / 2;
    CHECK_EQ(full_intermediate % world_size, 0)
        << "gate_up_proj intermediate dim is not divisible by world_size";
    const int64_t inter_shard = full_intermediate / world_size;

    auto gate_full = fused_gate_up.slice(1, 0, full_intermediate);
    auto up_full =
        fused_gate_up.slice(1, full_intermediate, full_intermediate * 2);
    auto gate_shard =
        gate_full.slice(1, rank * inter_shard, (rank + 1) * inter_shard);
    auto up_shard =
        up_full.slice(1, rank * inter_shard, (rank + 1) * inter_shard);
    fused_gate_up = torch::cat({gate_shard, up_shard}, 1);
  }

  auto gate_up_slice = slice_expert_weights(
      fused_gate_up, start_expert_id, num_experts_per_rank);
  CHECK_EQ(w13.sizes(), gate_up_slice.sizes())
      << "weight size mismatch for " << state_dict.prefix()
      << "experts.gate_up_proj";
  w13.copy_(gate_up_slice);
  return true;
}

bool load_fused_down_fallback(const StateDict& state_dict,
                              int64_t rank,
                              int64_t world_size,
                              int64_t start_expert_id,
                              int64_t num_experts_per_rank,
                              torch::Tensor& w2) {
  auto fused_down = get_tensor_with_weight_suffix(state_dict, "down_proj");
  if (!fused_down.defined()) {
    return false;
  }

  if (world_size > 1) {
    CHECK_EQ(fused_down.size(2) % world_size, 0)
        << "down_proj dim2 is not divisible by world_size";
    const int64_t down_shard = fused_down.size(2) / world_size;
    fused_down =
        fused_down.slice(2, rank * down_shard, (rank + 1) * down_shard);
  }

  auto down_slice =
      slice_expert_weights(fused_down, start_expert_id, num_experts_per_rank);
  CHECK_EQ(w2.sizes(), down_slice.sizes())
      << "weight size mismatch for " << state_dict.prefix()
      << "experts.down_proj";
  w2.copy_(down_slice);
  return true;
}

}  // namespace

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
      renormalize_(model_args.norm_topk_prob() ? 1 : 0),
      hidden_act_(model_args.hidden_act()),
      is_smoothquant_(false),
      quant_args_(quant_args),
      parallel_args_(parallel_args),
      options_(options),
      tp_pg_(parallel_args.tp_group_) {
  const int64_t num_experts = num_total_experts_;
  const int64_t intermediate_size =
      static_cast<int64_t>(model_args.moe_intermediate_size());
  const std::string& topk_method = model_args.topk_method();
  int64_t ep_size = parallel_args.ep_size();
  int64_t ep_rank = 0;
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
    /*
    The shared_experts are usually implemented using the RowParallelLinear
    layer. Typically, this output serves as the enable_result_reduction results
    for the module. If only tensor parallelism is applied, immediate
    reduction of the shared_experts output isn't necessary; instead, we perform
    the reduction once at the end of the MoE operation.
    */
    shared_experts_ =
        register_module("shared_experts",
                        DenseMLP(hidden_size_,
                                 intermediate_size * n_shared_experts_,
                                 is_gated_,
                                 false,
                                 hidden_act_,
                                 /*enable_result_reduction=*/false,
                                 quant_args,
                                 tp_pg_,
                                 options));
    shared_expert_gate_ = register_module(
        "shared_expert_gate",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size_, 1).bias(false)));
    shared_expert_gate_->weight.set_data(
        shared_expert_gate_->weight.to(options));
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
    input_smooth_ = register_parameter(
        "input_smooth",
        torch::empty({num_experts_per_rank_, hidden_size_}, fp_option),
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

torch::Tensor FusedMoEImpl::select_experts(
    const torch::Tensor& hidden_states_2d,
    const torch::Tensor& router_logits_2d,
    SelectedExpertInfo& selected_expert_info) {
  // prepare the parameters for select_experts
  xllm::kernel::MoeFusedTopkParams moe_active_topk_params;
  moe_active_topk_params.input = router_logits_2d;
  moe_active_topk_params.finished = torch::Tensor();
  moe_active_topk_params.topk = topk_;
  moe_active_topk_params.scoring_func = "softmax";
  auto [topk_weights, topk_ids] =
      xllm::kernel::moe_active_topk(moe_active_topk_params);
  topk_ids = topk_ids.to(torch::kInt32);
  if (renormalize_) {
    topk_weights = topk_weights / (topk_weights.sum(-1, true) + 1e-6);
  }

  xllm::kernel::MoeInitRoutingV2Params moe_init_routing_params;
  moe_init_routing_params.x = hidden_states_2d;
  moe_init_routing_params.expert_idx = topk_ids;
  moe_init_routing_params.scale = std::nullopt;
  moe_init_routing_params.offset = std::nullopt;
  moe_init_routing_params.active_num = hidden_states_2d.size(0) * topk_;
  moe_init_routing_params.expert_capacity = 0;
  moe_init_routing_params.expert_num = num_experts_per_rank_;
  moe_init_routing_params.drop_pad_mode = 0;
  moe_init_routing_params.expert_tokens_num_type = 1;
  moe_init_routing_params.expert_tokens_num_flag = true;
  moe_init_routing_params.row_idx_type = 0;
  std::vector<int64_t> expert_range = {
      start_expert_id_, start_expert_id_ + num_experts_per_rank_};
  moe_init_routing_params.active_expert_range = expert_range;
  moe_init_routing_params.quant_mode = -1;
  // TODO: NPU moe_init_routing_v2 is equivalent to moe_gen_idx +
  // moe_expand_input (and the token_count/cusum outputs) on other backends.
  auto [expand_hidden_states, expand_row_ids, group_list, dynamic_scale] =
      xllm::kernel::moe_init_routing_v2(moe_init_routing_params);
  (void)dynamic_scale;

  // collect the selected tensor
  selected_expert_info.reduce_weight = topk_weights;
  selected_expert_info.combine_idx = expand_row_ids;
  selected_expert_info.token_count_slice = group_list;
  selected_expert_info.cusum_token_count = group_list;
  return expand_hidden_states;
}

torch::Tensor FusedMoEImpl::forward_expert(
    const torch::Tensor& hidden_states,
    const torch::Tensor& router_logits,
    const std::optional<torch::Tensor>& shared_output) {
  // prepare the parameters for MoE computation
  torch::IntArrayRef hidden_states_shape = hidden_states.sizes();
  torch::ScalarType hidden_states_dtype = hidden_states.dtype().toScalarType();
  torch::Tensor hidden_states_2d =
      hidden_states.reshape({-1, hidden_states.size(-1)});
  torch::Tensor router_logits_2d =
      router_logits.reshape({-1, router_logits.size(-1)});

  // Step 1-3: select experts
  SelectedExpertInfo selected_expert_info;
  torch::Tensor expand_hidden_states =
      select_experts(hidden_states_2d, router_logits_2d, selected_expert_info);

  // Step 4: group gemm 1
  torch::Tensor gemm1_out =
      create_group_gemm_output(expand_hidden_states,
                               w13_,
                               selected_expert_info.token_count_slice,
                               hidden_states_dtype);

  {
    xllm::kernel::GroupGemmParams group_gemm_params;
    group_gemm_params.a = expand_hidden_states;
    if (w13_.size(1) != expand_hidden_states.size(1)) {
      w13_ = w13_.transpose(1, 2);
    }
    group_gemm_params.b = w13_;
    group_gemm_params.group_list = selected_expert_info.token_count_slice;
    group_gemm_params.split_item = 2;
    group_gemm_params.group_type = 0;
    group_gemm_params.group_list_type = 1;
    gemm1_out = xllm::kernel::group_gemm(group_gemm_params);
  }

  // Step 5: activation
  torch::Tensor act_out;

  xllm::kernel::ActivationParams activation_params;
  activation_params.input = gemm1_out;
  activation_params.output = act_out;
  activation_params.act_mode = hidden_act_;
  activation_params.is_gated = is_gated_;
  xllm::kernel::active(activation_params);
  act_out = activation_params.output;
  // Step 6: group gemm 2
  torch::Tensor gemm2_out =
      create_group_gemm_output(act_out,
                               w2_,
                               selected_expert_info.token_count_slice,
                               hidden_states_dtype);

  {
    xllm::kernel::GroupGemmParams group_gemm_params;
    group_gemm_params.a = act_out;
    if (w2_.size(1) != act_out.size(1)) {
      w2_ = w2_.transpose(1, 2);
    }
    group_gemm_params.b = w2_;
    group_gemm_params.group_list = selected_expert_info.token_count_slice;
    group_gemm_params.split_item = 2;
    group_gemm_params.group_type = 0;
    group_gemm_params.group_list_type = 1;
    gemm2_out = xllm::kernel::group_gemm(group_gemm_params);
  }

  // Step 7: combine the intermediate results and get the final hidden states
  torch::Tensor final_hidden_states;
  xllm::kernel::MoeCombineResultParams moe_combine_params;
  moe_combine_params.input = gemm2_out;
  moe_combine_params.reduce_weight = selected_expert_info.reduce_weight;
  moe_combine_params.gather_ids = selected_expert_info.combine_idx;
  final_hidden_states = xllm::kernel::moe_combine_result(moe_combine_params);
  if (shared_output.has_value()) {
    final_hidden_states = final_hidden_states + shared_output.value();
  }
  // reshape the final hidden states to the original shape
  final_hidden_states = final_hidden_states.reshape(hidden_states_shape);

  if (tp_pg_->world_size() > 1) {
    final_hidden_states = parallel_state::reduce(final_hidden_states, tp_pg_);
  }
  if (parallel_args_.ep_size() > 1) {
    final_hidden_states = parallel_state::reduce(final_hidden_states,
                                                 parallel_args_.moe_ep_group_);
  }
  return final_hidden_states;
}

torch::Tensor FusedMoEImpl::forward(const torch::Tensor& hidden_states,
                                     const ModelInputParams& input_params) {
  return forward(hidden_states, input_params, nullptr);
}

torch::Tensor FusedMoEImpl::forward(const torch::Tensor& hidden_states,
                                     const ModelInputParams& input_params,
                                     const FlashComm1Context* fc1_ctx) {
  torch::Tensor input = hidden_states;

  if (fc1_ctx && fc1_ctx->is_sequence_sharded()) {
    input = gather_sequence(hidden_states, *fc1_ctx);
  }

  bool need_slice = false;
  if (parallel_args_.dp_size() > 1 && parallel_args_.ep_size() > 1) {
    input = parallel_state::gather(input,
                                    parallel_args_.dp_local_process_group_,
                                    input_params.dp_global_token_nums);
    need_slice = true;
  }

  std::optional<torch::Tensor> shared_output = std::nullopt;
  if (n_shared_experts_ > 0) {
    shared_output = shared_experts_->forward(input, fc1_ctx);
    if (shared_expert_gate_) {
      auto gate = torch::sigmoid(shared_expert_gate_->forward(input));
      if (shared_output.has_value()) {
        torch::Tensor res = gate * shared_output.value();
        shared_output = res;
      }
    }
  }
  auto router_logits = gate_(input);
  auto output = forward_expert(input, router_logits, shared_output);

  if (fc1_ctx && fc1_ctx->is_sequence_sharded()) {
    FlashComm1Context ctx_copy = *fc1_ctx;
    ctx_copy.tp_group = tp_pg_;
    output = maybe_pad_and_reduce(output, ctx_copy, RowParallelReduceMode::REDUCE_SCATTER);
  } else if (tp_pg_->world_size() > 1) {
    output = parallel_state::reduce(output, tp_pg_);
  }

  if (parallel_args_.ep_size() > 1) {
    output = parallel_state::reduce(output, parallel_args_.moe_ep_group_);
  }

  if (need_slice) {
    const auto& dp_tokens = input_params.dp_global_token_nums;
    const int64_t dp_rank = parallel_args_.dp_local_process_group_->rank();
    auto start =
        std::accumulate(dp_tokens.begin(), dp_tokens.begin() + dp_rank, 0);
    auto end = start + dp_tokens[dp_rank];
    output = output.slice(0, start, end);
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
  std::vector<std::string> prefixes = {"gate_proj.", "up_proj."};
  if (is_smoothquant_) {
    LOAD_MOE_FUSED_WEIGHT("qweight", w1, w3, w13);
    LOAD_MOE_FUSED_WEIGHT("per_channel_scale", w1_scale, w3_scale, w13_scale);
    LOAD_MOE_WEIGHT("up_proj.", "smooth", input_smooth, -1);
    LOAD_MOE_WEIGHT("down_proj.", "qweight", w2, 1);
    LOAD_MOE_WEIGHT("down_proj.", "per_channel_scale", w2_scale, -1);
    LOAD_MOE_WEIGHT("down_proj.", "smooth", act_smooth, 0);
  } else {
    LOAD_MOE_FUSED_WEIGHT("weight", w1, w3, w13);
    LOAD_MOE_WEIGHT("down_proj.", "weight", w2, 1);

    // Some Qwen3.5-MoE checkpoints store expert weights in fused tensors
    // (gate_up_proj / down_proj). Fall back to this format when split
    // gate_proj/up_proj tensors are absent.
    if (!w13_is_loaded_) {
      w13_is_loaded_ = load_fused_gate_up_fallback(state_dict,
                                                   rank,
                                                   world_size,
                                                   start_expert_id,
                                                   num_experts_per_rank,
                                                   w13_);
    }

    if (!w2_is_loaded_) {
      w2_is_loaded_ = load_fused_down_fallback(state_dict,
                                               rank,
                                               world_size,
                                               start_expert_id,
                                               num_experts_per_rank,
                                               w2_);
    }
  }
}

void FusedMoEImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }

  if (n_shared_experts_ > 0) {
    shared_experts_->load_state_dict(
        state_dict.get_dict_with_prefix("shared_expert."));
    auto weight = state_dict.get_tensor("shared_expert_gate.weight");
    if (weight.defined()) {
      weight = weight.reshape({weight.size(0), -1});
      DCHECK_EQ(shared_expert_gate_->weight.sizes(), weight.sizes())
          << "proj weight size mismatch for " << name();
      shared_expert_gate_->weight.data().copy_(weight);
    }
  }

  gate_->load_state_dict(state_dict.get_dict_with_prefix("gate."));
  load_e_score_correction_bias(state_dict.get_dict_with_prefix("gate."));
  load_experts(state_dict.get_dict_with_prefix("experts."));
}

}  // namespace layer
}  // namespace xllm
