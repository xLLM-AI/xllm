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

#include "fused_moe.h"

#include <glog/logging.h>

#include <numeric>

#include "framework/parallel_state/parallel_state.h"
#include "kernels/dcu/aiter_moe_adapter.h"
#include "kernels/dcu/aiter_quant_adapter.h"
#include "kernels/ops_api.h"
#include "layers/common/dp_utils.h"

namespace xllm {
namespace layer {
namespace {

constexpr int64_t kAiterMoeBlockSize = 16;

struct LocalFp8ExpertRouting {
  torch::Tensor token_ids;
  torch::Tensor expert_ids;
  torch::Tensor weights;
  int64_t num_assignments = 0;
};

bool is_fp8_channelwise_moe(const QuantArgs& quant_args) {
  return quant_args.quant_method() == kQuantMethodFp8 &&
         quant_args.activation_dynamic();
}

torch::Tensor shuffle_moe_gemm1_weight(const torch::Tensor& weight) {
  return xllm::kernel::dcu::aiter::moe_layout_shuffle_gemm1(weight);
}

torch::Tensor shuffle_moe_gemm2_weight(const torch::Tensor& weight) {
  return xllm::kernel::dcu::aiter::moe_layout_shuffle_gemm2(weight);
}

torch::Tensor materialize_fp8_moe_weight(torch::Tensor shuffled_weight,
                                         const torch::Tensor& parameter) {
  CHECK(shuffled_weight.defined()) << "shuffled FP8 MoE weight is undefined.";
  CHECK(parameter.defined()) << "target FP8 MoE parameter is undefined.";
  CHECK_EQ(parameter.sizes(), shuffled_weight.sizes())
      << "shuffled FP8 MoE weight size mismatch.";
  if (shuffled_weight.device() == parameter.device() &&
      shuffled_weight.scalar_type() == parameter.scalar_type() &&
      shuffled_weight.is_contiguous()) {
    return shuffled_weight;
  }
  return shuffled_weight.to(parameter.options()).contiguous();
}

LocalFp8ExpertRouting select_local_fp8_experts(
    const torch::Tensor& expert_id,
    const torch::Tensor& reduce_weight,
    int64_t topk,
    int64_t start_expert_id,
    int64_t num_experts_per_rank) {
  CHECK_EQ(expert_id.sizes(), reduce_weight.sizes())
      << "FP8 MoE expert id/reduce weight shape mismatch.";
  const int64_t end_expert_id = start_expert_id + num_experts_per_rank;
  torch::Tensor local_mask = torch::logical_and(expert_id.ge(start_expert_id),
                                                expert_id.lt(end_expert_id));
  torch::Tensor flat_indices =
      torch::nonzero(local_mask.reshape({-1})).reshape({-1}).to(torch::kInt64);
  LocalFp8ExpertRouting routing;
  routing.num_assignments = flat_indices.numel();
  if (routing.num_assignments == 0) {
    return routing;
  }

  torch::Tensor flat_expert_id = expert_id.reshape({-1});
  torch::Tensor flat_reduce_weight = reduce_weight.reshape({-1});
  routing.token_ids = flat_indices.div(topk, "floor").to(torch::kInt64);
  routing.expert_ids =
      (flat_expert_id.index_select(0, flat_indices).to(torch::kInt64) -
       start_expert_id)
          .to(torch::kInt32)
          .view({routing.num_assignments, 1})
          .contiguous();
  routing.weights = flat_reduce_weight.index_select(0, flat_indices)
                        .view({routing.num_assignments, 1})
                        .contiguous();
  return routing;
}

}  // namespace

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
  CHECK_GT(ep_size, 0) << "ep_size must be positive.";
  tp_pg_ = parallel_args.tp_group_;
  if (ep_size > 1) {
    CHECK(parallel_args.moe_ep_group_ != nullptr)
        << "DCU FusedMoE requires a MoE EP process group when ep_size > 1.";
    CHECK(parallel_args.moe_tp_group_ != nullptr)
        << "DCU FusedMoE requires a MoE TP process group when ep_size > 1.";
    ep_rank = parallel_args.moe_ep_group_->rank();
    tp_pg_ = parallel_args.moe_tp_group_;
  }
  CHECK_EQ(num_experts % ep_size, 0)
      << "n_routed_experts must be divisible by ep_size.";

  num_experts_per_rank_ = num_experts / ep_size;
  start_expert_id_ = ep_rank * num_experts_per_rank_;

  if (topk_method == "noaux_tc") {
    e_score_correction_bias_ = register_parameter(
        "e_score_correction_bias",
        torch::empty({num_experts}, options.dtype(torch::kFloat32)),
        /*requires_grad*/ false);
  }

  gate_ = register_module(
      "gate_proj",
      ReplicatedLinear(
          hidden_size_, num_experts, /*bias*/ false, quant_args, options));

  if (n_shared_experts_ > 0) {
    ProcessGroup* shared_expert_pg;
    if (parallel_args_.ep_size() > 1) {
      CHECK(parallel_args_.ep_size() == parallel_args_.world_size())
          << "Models with shared experts only support ep_size equal to "
             "world size for now.";
      shared_expert_pg = parallel_args.tp_group_;
    } else {
      shared_expert_pg = parallel_args.process_group_;
    }
    shared_experts_ =
        register_module("shared_experts",
                        DenseMLP(hidden_size_,
                                 intermediate_size * n_shared_experts_,
                                 is_gated_,
                                 /*bias*/ false,
                                 hidden_act_,
                                 /*enable_result_reduction=*/true,
                                 quant_args,
                                 shared_expert_pg,
                                 options));
  }

  const int64_t world_size = tp_pg_->world_size();
  CHECK_EQ(intermediate_size % world_size, 0)
      << "moe_intermediate_size must be divisible by TP world size.";
  const int64_t local_intermediate_size = intermediate_size / world_size;
  torch::TensorOptions expert_weight_options = options_;
  if (is_fp8_channelwise_moe(quant_args_)) {
    expert_weight_options = options_.dtype(torch::kFloat8_e4m3fn);
  }

  w13_ = register_parameter(
      "w13",
      torch::empty(
          {num_experts_per_rank_, local_intermediate_size * 2, hidden_size_},
          expert_weight_options),
      false);

  w2_ = register_parameter(
      "w2",
      torch::empty(
          {num_experts_per_rank_, hidden_size_, local_intermediate_size},
          expert_weight_options),
      false);

  if (is_fp8_channelwise_moe(quant_args_)) {
    w13_scale_ = register_parameter(
        "w13_scale",
        torch::empty({num_experts_per_rank_, local_intermediate_size * 2, 1},
                     options_.dtype(torch::kFloat32)),
        false);
    w2_scale_ = register_parameter(
        "w2_scale",
        torch::empty({num_experts_per_rank_, hidden_size_, 1},
                     options_.dtype(torch::kFloat32)),
        false);
  }
}

// ---------------------------------------------------------------------------
// expert_gemm: dispatched through ops_api::group_gemm.
// On DCU this routes to the fused cuda::moe_fused_group_gemm kernel.
// ---------------------------------------------------------------------------
torch::Tensor FusedMoEImpl::expert_gemm(const torch::Tensor& input,
                                        const torch::Tensor& weight,
                                        const torch::Tensor& token_count) {
  auto output = torch::empty({input.size(0), weight.size(1)}, input.options());
  xllm::kernel::GroupGemmParams params;
  params.a = input;
  params.b = weight;
  params.token_count = token_count;
  params.output = output;
  params.max_dim = input.size(0);
  params.trans_a = false;
  params.trans_b = true;
  params.a_quant_bit = -1;
  return xllm::kernel::group_gemm(params);
}

// ---------------------------------------------------------------------------
// select_experts: steps 1–3 of the MoE pipeline
//   step 1 — moe_active_topk → cuda::moe_fused_topk
//   step 2 — moe_gen_idx     → cuda::moe_fused_compute_index (fused)
//   step 3 — moe_expand_input → torch index_select
// ---------------------------------------------------------------------------
std::pair<torch::Tensor, torch::Tensor> FusedMoEImpl::route_experts(
    const torch::Tensor& router_logits_2d) {
  std::optional<torch::Tensor> e_score_correction_bias = std::nullopt;
  if (e_score_correction_bias_.defined()) {
    e_score_correction_bias =
        e_score_correction_bias_.scalar_type() == torch::kFloat32
            ? e_score_correction_bias_
            : e_score_correction_bias_.to(torch::kFloat32);
  }

  torch::Tensor reduce_weight;
  torch::Tensor expert_id;
  xllm::kernel::MoeFusedTopkParams params;
  params.input = router_logits_2d;
  params.topk = topk_;
  params.num_expert_group = num_expert_group_;
  params.topk_group = topk_group_;
  params.normalize = renormalize_;
  params.normed_by = "topk_logit";
  params.scoring_func = scoring_func_;
  params.route_scale = route_scale_;
  params.e_score_correction_bias = e_score_correction_bias;
  std::tie(reduce_weight, expert_id) = xllm::kernel::moe_active_topk(params);
  return {reduce_weight, expert_id};
}

torch::Tensor FusedMoEImpl::select_experts(
    const torch::Tensor& hidden_states_2d,
    const torch::Tensor& router_logits_2d,
    SelectedExpertInfo& selected_expert_info) {
  // ---- Step 1: moe_active_topk ----
  auto [reduce_weight, expert_id] = route_experts(router_logits_2d);

  // ---- Step 2: moe_gen_idx (fused kernel) ----
  torch::Tensor src_dst, dst_src, expert_sizes;
  {
    xllm::kernel::MoeGenIdxParams gen_params;
    gen_params.expert_id = expert_id;
    gen_params.expert_num = num_total_experts_;
    auto output_vec = xllm::kernel::moe_gen_idx(gen_params);
    src_dst = output_vec[0];
    dst_src = output_vec[1];
    expert_sizes = output_vec[2];
  }

  // Compute global offset for local expert subset
  int64_t global_offset = 0;
  if (start_expert_id_ > 0) {
    global_offset =
        expert_sizes.slice(0, 0, start_expert_id_).sum().item<int64_t>();
  }
  int64_t local_total =
      expert_sizes
          .slice(0, start_expert_id_, start_expert_id_ + num_experts_per_rank_)
          .sum()
          .item<int64_t>();

  // ---- Step 3: moe_expand_input (torch index_select) ----
  auto token_ids = dst_src.div(topk_, "floor").to(torch::kInt64);
  auto expand_all = hidden_states_2d.index_select(0, token_ids);

  // Local expert token counts (for the local slice of the expanded buffer)
  auto token_count =
      expert_sizes
          .slice(0, start_expert_id_, start_expert_id_ + num_experts_per_rank_)
          .to(torch::kInt32);

  selected_expert_info.reduce_weight = reduce_weight;
  selected_expert_info.expert_id = expert_id;
  selected_expert_info.src_dst = src_dst;
  selected_expert_info.dst_src = dst_src;
  selected_expert_info.token_count = token_count;
  selected_expert_info.global_offset = global_offset;
  selected_expert_info.local_total = local_total;

  return expand_all;
}

torch::Tensor FusedMoEImpl::forward_fp8_channelwise_experts(
    const torch::Tensor& hidden_states,
    const torch::Tensor& hidden_states_2d,
    const torch::Tensor& router_logits_2d) {
  CHECK(w13_.defined() && w2_.defined())
      << "FP8 MoE weights must be loaded before forward.";
  CHECK(w13_scale_.defined() && w2_scale_.defined())
      << "FP8 MoE weight scales must be loaded before forward.";

  const int64_t num_tokens = hidden_states_2d.size(0);
  const torch::ScalarType output_dtype = hidden_states.dtype().toScalarType();
  auto [reduce_weight, expert_id] = route_experts(router_logits_2d);
  if (parallel_args_.ep_size() > 1) {
    LocalFp8ExpertRouting routing =
        select_local_fp8_experts(expert_id,
                                 reduce_weight,
                                 topk_,
                                 start_expert_id_,
                                 num_experts_per_rank_);
    torch::Tensor final_hidden_states_2d =
        torch::zeros({num_tokens, hidden_size_},
                     hidden_states_2d.options().dtype(output_dtype));
    if (routing.num_assignments == 0) {
      return final_hidden_states_2d.reshape(hidden_states.sizes());
    }

    xllm::kernel::dcu::aiter::MoeSortedTokens sorted_tokens =
        xllm::kernel::dcu::aiter::moe_align_block_size(routing.expert_ids,
                                                       routing.weights,
                                                       num_experts_per_rank_,
                                                       kAiterMoeBlockSize);
    torch::Tensor local_hidden_states =
        hidden_states_2d.index_select(0, routing.token_ids).contiguous();
    torch::Tensor quantized_input;
    torch::Tensor input_scale;
    std::tie(quantized_input, input_scale) =
        xllm::kernel::dcu::aiter::per_token_quant_fp8(local_hidden_states,
                                                      std::nullopt);

    torch::Tensor gemm1_out =
        torch::empty({routing.num_assignments, 1, w13_.size(1)},
                     hidden_states_2d.options().dtype(output_dtype));
    xllm::kernel::dcu::aiter::moe_gemm_fp8_channelwise(
        quantized_input,
        w13_,
        gemm1_out,
        input_scale,
        w13_scale_,
        std::nullopt,
        sorted_tokens,
        /*topk=*/1,
        xllm::kernel::dcu::aiter::select_moe_gemm1_mode(
            routing.num_assignments),
        /*delta=*/1,
        xllm::kernel::dcu::aiter::select_moe_m_key(routing.num_assignments));

    torch::Tensor act_out =
        torch::empty({routing.num_assignments, w13_.size(1) / 2},
                     hidden_states_2d.options().dtype(output_dtype));
    {
      xllm::kernel::ActivationParams activation_params;
      activation_params.input =
          gemm1_out.view({routing.num_assignments, w13_.size(1)});
      activation_params.output = act_out;
      activation_params.act_mode = hidden_act_;
      activation_params.is_gated = is_gated_;
      xllm::kernel::active(activation_params);
    }

    torch::Tensor quantized_act;
    torch::Tensor act_scale;
    std::tie(quantized_act, act_scale) =
        xllm::kernel::dcu::aiter::per_token_quant_fp8(act_out.contiguous(),
                                                      std::nullopt);

    torch::Tensor gemm2_out =
        torch::empty({routing.num_assignments, 1, hidden_size_},
                     hidden_states_2d.options().dtype(output_dtype));
    xllm::kernel::dcu::aiter::moe_gemm_fp8_channelwise(
        quantized_act,
        w2_,
        gemm2_out,
        act_scale,
        w2_scale_,
        routing.weights,
        sorted_tokens,
        /*topk=*/1,
        xllm::kernel::dcu::aiter::select_moe_gemm2_mode(
            routing.num_assignments),
        /*delta=*/1,
        xllm::kernel::dcu::aiter::select_moe_m_key(routing.num_assignments));

    torch::Tensor local_output =
        gemm2_out.view({routing.num_assignments, hidden_size_});
    final_hidden_states_2d.index_add_(0, routing.token_ids, local_output);
    return final_hidden_states_2d.reshape(hidden_states.sizes());
  }

  xllm::kernel::dcu::aiter::MoeSortedTokens sorted_tokens =
      xllm::kernel::dcu::aiter::moe_align_block_size(
          expert_id, reduce_weight, num_total_experts_, kAiterMoeBlockSize);

  torch::Tensor quantized_input;
  torch::Tensor input_scale;
  std::tie(quantized_input, input_scale) =
      xllm::kernel::dcu::aiter::per_token_quant_fp8(
          hidden_states_2d.contiguous(), std::nullopt);

  torch::Tensor gemm1_out =
      torch::empty({num_tokens, topk_, w13_.size(1)},
                   hidden_states_2d.options().dtype(output_dtype));
  xllm::kernel::dcu::aiter::moe_gemm_fp8_channelwise(
      quantized_input,
      w13_,
      gemm1_out,
      input_scale,
      w13_scale_,
      std::nullopt,
      sorted_tokens,
      topk_,
      xllm::kernel::dcu::aiter::select_moe_gemm1_mode(num_tokens),
      topk_,
      xllm::kernel::dcu::aiter::select_moe_m_key(num_tokens));

  torch::Tensor act_out =
      torch::empty({num_tokens * topk_, w13_.size(1) / 2},
                   hidden_states_2d.options().dtype(output_dtype));
  {
    xllm::kernel::ActivationParams activation_params;
    activation_params.input =
        gemm1_out.view({num_tokens * topk_, w13_.size(1)});
    activation_params.output = act_out;
    activation_params.act_mode = hidden_act_;
    activation_params.is_gated = is_gated_;
    xllm::kernel::active(activation_params);
  }

  torch::Tensor quantized_act;
  torch::Tensor act_scale;
  std::tie(quantized_act, act_scale) =
      xllm::kernel::dcu::aiter::per_token_quant_fp8(act_out.contiguous(),
                                                    std::nullopt);

  torch::Tensor gemm2_out =
      torch::empty({num_tokens, topk_, hidden_size_},
                   hidden_states_2d.options().dtype(output_dtype));
  xllm::kernel::dcu::aiter::moe_gemm_fp8_channelwise(
      quantized_act,
      w2_,
      gemm2_out,
      act_scale,
      w2_scale_,
      reduce_weight.contiguous(),
      sorted_tokens,
      /*topk=*/1,
      xllm::kernel::dcu::aiter::select_moe_gemm2_mode(num_tokens),
      topk_,
      xllm::kernel::dcu::aiter::select_moe_m_key(num_tokens));

  torch::Tensor final_hidden_states = gemm2_out.sum(/*dim=*/1);
  return final_hidden_states.reshape(hidden_states.sizes());
}

// ---------------------------------------------------------------------------
// forward_experts
// ---------------------------------------------------------------------------
torch::Tensor FusedMoEImpl::forward_experts(
    const torch::Tensor& hidden_states,
    const torch::Tensor& router_logits) {
  if (!stream_initialized_) {
    device_ = xllm::Device(hidden_states.device());
    routed_stream_ = device_.get_stream_from_pool();
    shared_stream_ = device_.get_stream_from_pool();
    stream_initialized_ = true;
  }

  auto hidden_states_shape = hidden_states.sizes();
  auto hidden_states_dtype = hidden_states.dtype().toScalarType();
  auto hidden_states_2d = hidden_states.reshape({-1, hidden_states.size(-1)});
  auto router_logits_2d = router_logits.reshape({-1, router_logits.size(-1)});

  if (is_fp8_channelwise_moe(quant_args_)) {
    auto final_hidden_states = forward_fp8_channelwise_experts(
        hidden_states, hidden_states_2d, router_logits_2d);

    if (parallel_args_.ep_size() > 1) {
      final_hidden_states = parallel_state::reduce(
          final_hidden_states, parallel_args_.moe_ep_group_);
    }
    if (tp_pg_->world_size() > 1) {
      final_hidden_states = parallel_state::reduce(final_hidden_states, tp_pg_);
    }

    torch::Tensor shared_expert_output;
    if (n_shared_experts_ > 0) {
      auto current_stream = device_.current_stream();
      shared_stream_->wait_stream(*current_stream);
      {
        torch::StreamGuard stream_guard = shared_stream_->set_stream_guard();
        shared_expert_output = shared_experts_(hidden_states);
        shared_expert_output =
            shared_expert_output.reshape({-1, shared_expert_output.size(-1)});
      }

      current_stream->wait_stream(*shared_stream_);
      final_hidden_states += shared_expert_output.reshape(hidden_states_shape);
    }
    return final_hidden_states;
  }

  // ---- Steps 1-3: select experts ----
  SelectedExpertInfo selected_expert_info;
  auto expand_all =
      select_experts(hidden_states_2d, router_logits_2d, selected_expert_info);

  int64_t N = hidden_states_2d.size(0);
  int64_t local_total = selected_expert_info.local_total;
  int64_t global_offset = selected_expert_info.global_offset;
  auto token_count = selected_expert_info.token_count;
  auto dst_src = selected_expert_info.dst_src;
  auto src_dst = selected_expert_info.src_dst;

  // Slice the expanded buffer to only the local experts' tokens
  auto expand_local =
      expand_all.slice(0, global_offset, global_offset + local_total)
          .contiguous();

  // ---- Step 4: group gemm 1 (w13: gate_proj + up_proj) ----
  auto gemm1_out =
      expert_gemm(expand_local.view({-1, hidden_size_}).to(hidden_states_dtype),
                  w13_,
                  token_count);

  // ---- Step 5: activation ----
  torch::Tensor act_out;
  if (is_gated_) {
    act_out = gemm1_out.slice(1, 0, gemm1_out.size(1) / 2).contiguous();
  } else {
    act_out = gemm1_out;
  }
  {
    xllm::kernel::ActivationParams activation_params;
    activation_params.input = gemm1_out;
    activation_params.output = act_out;
    activation_params.act_mode = hidden_act_;
    activation_params.is_gated = is_gated_;
    xllm::kernel::active(activation_params);
  }

  // ---- Step 6: group gemm 2 (w2: down_proj) ----
  auto gemm2_local = expert_gemm(act_out, w2_, token_count);

  // Release intermediates
  expand_all = torch::Tensor();
  expand_local = torch::Tensor();
  act_out = torch::Tensor();

  // ---- Step 7: combine (fused kernel via ops_api::moe_combine_result) ----
  // Scatter local expert outputs into full flat buffer via index_copy_,
  // then weighted reduction via cuda::moe_fused_combine kernel.
  int64_t NT = N * topk_;
  auto gemm2_full = torch::zeros({NT, hidden_size_}, gemm2_local.options());
  if (local_total > 0) {
    auto local_indices =
        dst_src.slice(0, global_offset, global_offset + local_total)
            .to(torch::kInt64);
    gemm2_full.index_copy_(0, local_indices, gemm2_local);
  }
  torch::Tensor final_hidden_states;
  {
    xllm::kernel::MoeCombineResultParams combine_params;
    combine_params.input = gemm2_full;
    combine_params.reduce_weight = selected_expert_info.reduce_weight;
    combine_params.gather_ids = src_dst;
    combine_params.cusum_token_count = std::nullopt;
    combine_params.start_expert_id = 0;
    combine_params.expert_size = 0;
    combine_params.bias = std::nullopt;
    final_hidden_states = xllm::kernel::moe_combine_result(combine_params);
  }
  final_hidden_states = final_hidden_states.reshape(hidden_states_shape);

  // Communication: EP + TP AllReduce, overlapped with shared experts
  auto current_stream = device_.current_stream();
  routed_stream_->wait_stream(*current_stream);
  {
    torch::StreamGuard stream_guard = routed_stream_->set_stream_guard();
    if (parallel_args_.ep_size() > 1) {
      final_hidden_states = parallel_state::reduce(
          final_hidden_states, parallel_args_.moe_ep_group_);
    }
    if (tp_pg_->world_size() > 1) {
      final_hidden_states = parallel_state::reduce(final_hidden_states, tp_pg_);
    }
  }

  torch::Tensor shared_expert_output;
  if (n_shared_experts_ > 0) {
    shared_stream_->wait_stream(*current_stream);
    {
      torch::StreamGuard stream_guard = shared_stream_->set_stream_guard();
      shared_expert_output = shared_experts_(hidden_states);
      shared_expert_output =
          shared_expert_output.reshape({-1, shared_expert_output.size(-1)});
    }
  }

  // Join streams and add shared expert residual
  current_stream->wait_stream(*routed_stream_);
  if (n_shared_experts_ > 0) {
    current_stream->wait_stream(*shared_stream_);
    final_hidden_states += shared_expert_output;
  }

  return final_hidden_states;
}

// ---------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------
torch::Tensor FusedMoEImpl::forward(const torch::Tensor& hidden_states,
                                    const ModelInputParams& input_params) {
  bool is_dp_ep_parallel =
      parallel_args_.dp_size() > 1 && parallel_args_.ep_size() > 1;

  auto input = hidden_states;
  if (is_dp_ep_parallel) {
    input = parallel_state::gather(input,
                                   parallel_args_.dp_local_process_group_,
                                   input_params.parallel.dp_global_token_nums);
  }

  auto router_logits = gate_(input);
  auto output = forward_experts(input, router_logits);

  if (is_dp_ep_parallel) {
    output = get_dp_local_slice(output, input_params, parallel_args_);
  }

  return output;
}

// ---------------------------------------------------------------------------
// weight loading
// ---------------------------------------------------------------------------
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
  if (is_fp8_channelwise_moe(quant_args_)) {
    auto shuffle_w13 = [this](const torch::Tensor& weight) {
      return materialize_fp8_moe_weight(shuffle_moe_gemm1_weight(weight), w13_);
    };
    auto shuffle_w2 = [this](const torch::Tensor& weight) {
      return materialize_fp8_moe_weight(shuffle_moe_gemm2_weight(weight), w2_);
    };
    LOAD_MOE_FUSED_WEIGHT_TRANSFORM("weight", w1, w3, w13, shuffle_w13);
    LOAD_MOE_WEIGHT_TRANSFORM("down_proj.", "weight", w2, shuffle_w2, 1);
    LOAD_MOE_FUSED_WEIGHT("weight_scale", w1_scale, w3_scale, w13_scale);
    LOAD_MOE_WEIGHT("down_proj.", "weight_scale", w2_scale, -1);
    return;
  }
  LOAD_MOE_FUSED_WEIGHT("weight", w1, w3, w13);
  LOAD_MOE_WEIGHT("down_proj.", "weight", w2, 1);
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
