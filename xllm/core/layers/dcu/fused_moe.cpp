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

#include <numeric>

#include "framework/parallel_state/parallel_state.h"
#include "kernels/ops_api.h"
#include "layers/common/dp_utils.h"

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

  num_experts_per_rank_ = num_experts / ep_size;
  start_expert_id_ = ep_rank * num_experts_per_rank_;

  if (topk_method == "noaux_tc") {
    e_score_correction_bias_ = register_parameter(
        "e_score_correction_bias", torch::empty({num_experts}, options),
	/*requires_grad*/false);
  }

  gate_ = register_module(
      "gate_proj",
      ReplicatedLinear(hidden_size_, num_experts, /*bias*/false, quant_args, options));

  if (n_shared_experts_ > 0) {
    ProcessGroup* shared_expert_pg;
    if (parallel_args_.ep_size() > 1) {
      CHECK(parallel_args_.ep_size() == parallel_args_.world_size())
          << "Models with shared experts only support ep_size equal to "
             "world size for now.";
      shared_expert_pg = parallel_args.moe_tp_group_;
    } else {
      shared_expert_pg = parallel_args.process_group_;
    }
    shared_experts_ =
        register_module("shared_experts",
                        DenseMLP(hidden_size_,
                                 intermediate_size * n_shared_experts_,
                                 is_gated_,
                                 /*bias*/false,
                                 hidden_act_,
                                 /*enable_result_reduction=*/true,
                                 quant_args,
                                 shared_expert_pg,
                                 options));
  }

  const int64_t world_size = tp_pg_->world_size();
  int64_t local_intermediate_size = intermediate_size / world_size;

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

// ---------------------------------------------------------------------------
// expert_gemm: batched per-expert matmul via torch::mm.
// input is ordered by expert; token_count[e] tokens for each expert.
// ---------------------------------------------------------------------------
torch::Tensor FusedMoEImpl::expert_gemm(const torch::Tensor& input,
                                        const torch::Tensor& weight,
                                        const torch::Tensor& token_count) {
  int64_t num_experts = token_count.size(0);
  auto token_count_cpu = token_count.to(torch::kCPU);

  std::vector<torch::Tensor> outputs;
  outputs.reserve(num_experts);
  int64_t offset = 0;
  for (int64_t e = 0; e < num_experts; ++e) {
    int64_t cnt = token_count_cpu[e].item<int64_t>();
    if (cnt > 0) {
      auto tokens = input.slice(0, offset, offset + cnt);
      outputs.push_back(torch::mm(tokens, weight[e].t()));
      offset += cnt;
    }
  }
  if (outputs.empty()) {
    return torch::empty({0, weight.size(1)}, input.options());
  }
  return torch::cat(outputs, 0);
}

// ---------------------------------------------------------------------------
// select_experts: steps 1–3 of the MoE pipeline
//   step 1 — moe_active_topk (softmax + topk) → cuda::moe_fused_topk
//   step 2 — moe_gen_idx → torch::argsort / bincount / cumsum
//   step 3 — moe_expand_input → torch index_select
// ---------------------------------------------------------------------------
torch::Tensor FusedMoEImpl::select_experts(
    const torch::Tensor& hidden_states_2d,
    const torch::Tensor& router_logits_2d,
    SelectedExpertInfo& selected_expert_info) {
  std::optional<torch::Tensor> e_score_correction_bias = std::nullopt;
  if (e_score_correction_bias_.defined()) {
    e_score_correction_bias = e_score_correction_bias_;
  }

  // ---- Step 1: moe_active_topk ----
  torch::Tensor reduce_weight;
  torch::Tensor expert_id;
  {
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
  }

  // ---- Step 2: moe_gen_idx (torch ops) ----
  auto expert_id_flat = expert_id.view({-1});  // [N*topk], int32
  int64_t N = expert_id.size(0);

  // Count tokens per global expert
  auto expert_sizes =
      torch::bincount(expert_id_flat.to(torch::kInt64),
                      torch::nullopt,
                      num_total_experts_);  // [num_experts], int64

  // Sort by expert id → expert-grouped ordering
  auto dst_src = torch::argsort(expert_id_flat);  // [N*topk], pos→flat_idx
  auto src_dst = torch::argsort(dst_src);         // [N*topk], flat_idx→pos

  // Compute global offset for local expert subset
  int64_t global_offset = 0;
  if (start_expert_id_ > 0) {
    global_offset = expert_sizes.slice(0, 0, start_expert_id_).sum().item<int64_t>();
  }
  int64_t local_total =
      expert_sizes.slice(0, start_expert_id_, start_expert_id_ + num_experts_per_rank_)
          .sum()
          .item<int64_t>();

  // ---- Step 3: moe_expand_input (torch ops) ----
  // For the expand we still need ALL tokens (the token ordering groups all
  // global experts).  The per-rank GEMMs will slice into the local portion.
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

  // ---- Step 7: combine ----
  // Scatter local expert outputs into a full flat buffer
  // dst_src[global_offset + j] = flat index for local position j
  int64_t NT = N * topk_;
  auto gemm2_full = torch::zeros({NT, hidden_size_}, gemm2_local.options());
  if (local_total > 0) {
    auto local_indices =
        dst_src.slice(0, global_offset, global_offset + local_total)
            .to(torch::kInt64);
    gemm2_full.index_copy_(0, local_indices, gemm2_local);
  }

  // Weighted sum: reshape to [N, topk, hidden], multiply by router weights
  auto reduce_weight =
      selected_expert_info.reduce_weight.to(hidden_states_dtype);
  gemm2_full = gemm2_full.view({N, topk_, -1});  // [N, topk, hidden]
  auto final_hidden_states =
      (gemm2_full * reduce_weight.unsqueeze(-1)).sum(1);  // [N, hidden]
  final_hidden_states = final_hidden_states.reshape(hidden_states_shape);

  // Communication: EP + TP AllReduce on final_hidden_states (smaller than
  // gemm2_full), overlapped with shared expert computation
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
