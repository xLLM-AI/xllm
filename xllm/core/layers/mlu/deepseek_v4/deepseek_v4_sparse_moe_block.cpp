/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "layers/mlu/deepseek_v4/deepseek_v4_sparse_moe_block.h"

#include <glog/logging.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "core/framework/config/eplb_config.h"
#include "framework/parallel_state/parallel_state.h"
#include "layers/common/dp_utils.h"

namespace xllm {
namespace layer {
namespace {

torch::Tensor reshape_topk(const torch::Tensor& topk, int64_t hidden_rows) {
  const int64_t topk_size = topk.size(-1);
  return topk.reshape({hidden_rows, topk_size}).contiguous();
}

std::pair<int32_t, int32_t> split_range(int32_t size,
                                        int32_t parts,
                                        int32_t rank) {
  const int32_t base = size / parts;
  const int32_t remainder = size % parts;
  const int32_t count = base + (rank < remainder ? 1 : 0);
  const int32_t offset = rank * base + std::min(rank, remainder);
  return {offset, count};
}

}  // namespace

DeepseekV4SparseMoEBlockImpl::DeepseekV4SparseMoEBlockImpl(
    const ModelArgs& model_args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options,
    bool use_hash,
    const std::shared_ptr<Stream>& routed_comm_stream,
    const std::shared_ptr<Stream>& shared_compute_stream)
    : parallel_args_(parallel_args) {
  enable_deep_ep_ =
      ::xllm::EPLBConfig::get_instance().expert_parallel_degree() == 2 &&
      parallel_args_.ep_size() > 1;
  const FusedMoEArgs moe_args{
      .is_gated = true, .enable_result_reduction = false, .use_hash = use_hash};
  moe_ = register_module("moe",
                         FusedMoE(model_args,
                                  moe_args,
                                  quant_args,
                                  parallel_args,
                                  options,
                                  routed_comm_stream,
                                  shared_compute_stream));
}

void DeepseekV4SparseMoEBlockImpl::load_state_dict(
    const StateDict& state_dict) {
  moe_->load_state_dict(state_dict);
}

void DeepseekV4SparseMoEBlockImpl::verify_loaded_weights() const {
  moe_->verify_loaded_weights();
}

FusedMoEImpl::RouteInfo DeepseekV4SparseMoEBlockImpl::prep_route(
    torch::Tensor& hidden_states,
    const std::optional<torch::Tensor>& input_ids) {
  return moe_->prep_route(hidden_states, input_ids);
}

bool DeepseekV4SparseMoEBlockImpl::need_gather() const {
  return need_selected_moe_dp_gather(parallel_args_);
}

ProcessGroup* DeepseekV4SparseMoEBlockImpl::routed_pg() const {
  return parallel_args_.ep_size() > 1 ? parallel_args_.moe_ep_group_
                                      : parallel_args_.tp_group_;
}

FusedMoEImpl::RouteInfo DeepseekV4SparseMoEBlockImpl::make_route(
    const torch::Tensor& topk_weights,
    const torch::Tensor& topk_ids,
    int64_t hidden_rows) const {
  const int64_t topk = topk_weights.size(-1);
  CHECK_EQ(topk_ids.size(-1), topk)
      << "topk_ids last dim must match topk_weights last dim";

  FusedMoEImpl::RouteInfo route;
  route.reduce_weight = topk_weights.reshape({hidden_rows, topk});
  route.expert_id = topk_ids.reshape({hidden_rows, topk});
  if (route.expert_id.scalar_type() != torch::kInt) {
    route.expert_id = route.expert_id.to(torch::kInt);
  }
  return route;
}

std::vector<int32_t> DeepseekV4SparseMoEBlockImpl::get_row_dp_tokens(
    int64_t hidden_rows,
    const ModelInputParams& input_params) const {
  const std::vector<int32_t>& token_nums =
      input_params.parallel.dp_global_token_nums;
  CHECK(!token_nums.empty()) << "dp_global_token_nums is empty";

  CHECK(parallel_args_.dp_local_process_group_ != nullptr)
      << "dp_local_process_group_ is not initialized";
  const int64_t dp_rank = parallel_args_.dp_local_process_group_->rank();
  CHECK_GE(dp_rank, 0) << "invalid dp rank " << dp_rank;
  CHECK_LT(dp_rank, static_cast<int64_t>(token_nums.size()))
      << "dp rank " << dp_rank << " exceeds dp_global_token_nums size "
      << token_nums.size();
  const int32_t local_token_num = token_nums[dp_rank];
  CHECK_GT(local_token_num, 0)
      << "local dp token num must be positive for row-level conversion";
  CHECK_EQ(hidden_rows % local_token_num, 0)
      << "hidden rows " << hidden_rows
      << " must be divisible by local dp token num " << local_token_num;
  const int64_t row_factor = hidden_rows / local_token_num;

  std::vector<int32_t> row_token_nums;
  row_token_nums.reserve(token_nums.size());
  for (int32_t token_num : token_nums) {
    const int64_t row_token_num = static_cast<int64_t>(token_num) * row_factor;
    row_token_nums.emplace_back(static_cast<int32_t>(row_token_num));
  }
  return row_token_nums;
}

torch::Tensor DeepseekV4SparseMoEBlockImpl::forward_selected(
    const torch::Tensor& hidden_states,
    const torch::Tensor& topk_weights,
    const torch::Tensor& topk_ids,
    const ModelInputParams& input_params) {
  std::vector<int64_t> hidden_shape = hidden_states.sizes().vec();
  torch::Tensor hidden_rows =
      hidden_states.reshape({-1, hidden_states.size(-1)}).contiguous();
  const int64_t row_count = hidden_rows.size(0);
  torch::Tensor topk_weights_2d = reshape_topk(topk_weights, row_count);
  torch::Tensor topk_ids_2d = reshape_topk(topk_ids, row_count);

  if (enable_deep_ep_ && all_dp_ranks_are_decode(input_params)) {
    FusedMoEImpl::RouteInfo route =
        make_route(topk_weights_2d, topk_ids_2d, /*hidden_rows=*/row_count);
    torch::Tensor output = moe_->forward_experts(
        hidden_rows, /*enable_all2all_communication=*/true, route);
    return output.reshape(hidden_shape);
  }

  std::vector<int32_t> row_token_nums;
  if (need_gather()) {
    row_token_nums = get_row_dp_tokens(row_count, input_params);
  }

  SelectedMoeInputs moe_inputs = gather_selected_moe_inputs(hidden_rows,
                                                            topk_weights_2d,
                                                            topk_ids_2d,
                                                            row_token_nums,
                                                            parallel_args_);

  torch::Tensor shared_out = moe_->forward_shared(moe_inputs.hidden_states);
  const int64_t gathered_rows = moe_inputs.hidden_states.size(0);
  FusedMoEImpl::RouteInfo route = make_route(moe_inputs.topk_weights,
                                             moe_inputs.topk_ids,
                                             /*hidden_rows=*/gathered_rows);
  torch::Tensor routed_out =
      moe_->forward_experts(moe_inputs.hidden_states,
                            /*enable_all2all_communication=*/false,
                            route);
  ProcessGroup* reduce_group = routed_pg();
  CHECK(reduce_group != nullptr) << "routed process group is not initialized";
  if (reduce_group->world_size() > 1) {
    routed_out = parallel_state::reduce(routed_out, reduce_group);
  }

  torch::Tensor output = std::move(routed_out);
  if (shared_out.defined()) {
    output.add_(shared_out);
  }
  if (moe_inputs.need_slice) {
    output = slice_selected_moe_output(
        std::move(output), row_token_nums, parallel_args_);
  }
  return output.reshape(hidden_shape);
}

torch::Tensor DeepseekV4SparseMoEBlockImpl::forward_cp(
    const torch::Tensor& local_hidden_states,
    const std::optional<torch::Tensor>& local_input_ids,
    const mlu_v4_cp::DeepseekV4CpContext& cp_context) {
  ProcessGroup* ep_group = routed_pg();
  ProcessGroup* tp_group = parallel_args_.tp_group_;
  CHECK(ep_group != nullptr);
  CHECK(tp_group != nullptr);
  const int32_t tp_size = tp_group->world_size();
  const int32_t tp_rank = tp_group->rank();
  const int32_t ep_size = ep_group->world_size();
  CHECK_EQ(ep_size, cp_context.cp_group->world_size() * tp_size)
      << "DeepSeek V4 CP-aware MoE currently requires dp_size == 1 and a "
         "world-sized EP group.";

  const int32_t local_tokens =
      static_cast<int32_t>(local_hidden_states.size(0));
  const std::pair<int32_t, int32_t> local_range =
      split_range(local_tokens, tp_size, tp_rank);
  const int32_t local_offset = local_range.first;
  const int32_t local_unique_tokens = local_range.second;
  torch::Tensor unique_hidden = local_hidden_states.narrow(
      /*dim=*/0, local_offset, local_unique_tokens);

  std::vector<int32_t> unique_tokens_per_ep_rank;
  unique_tokens_per_ep_rank.reserve(static_cast<size_t>(ep_size));
  for (int32_t ep_rank = 0; ep_rank < ep_size; ++ep_rank) {
    const int32_t cp_rank = ep_rank / tp_size;
    const int32_t attention_tp_rank = ep_rank % tp_size;
    const int32_t cp_tokens =
        cp_context.geometry.tokens_per_rank[static_cast<size_t>(cp_rank)];
    unique_tokens_per_ep_rank.emplace_back(
        split_range(cp_tokens, tp_size, attention_tp_rank).second);
  }

  torch::Tensor gathered_hidden = parallel_state::gather(
      unique_hidden, ep_group, unique_tokens_per_ep_rank);
  std::optional<torch::Tensor> gathered_ids = std::nullopt;
  if (local_input_ids.has_value()) {
    torch::Tensor unique_ids = local_input_ids.value().narrow(
        /*dim=*/0, local_offset, local_unique_tokens);
    gathered_ids =
        parallel_state::gather(unique_ids, ep_group, unique_tokens_per_ep_rank);
  }

  FusedMoEImpl::RouteInfo route =
      moe_->prep_route(gathered_hidden, gathered_ids);
  std::vector<int64_t> gathered_shape = gathered_hidden.sizes().vec();
  torch::Tensor gathered_rows =
      gathered_hidden.reshape({-1, gathered_hidden.size(-1)}).contiguous();
  const int64_t row_factor = gathered_rows.size(0) / gathered_hidden.size(0);
  torch::Tensor shared_out = moe_->forward_shared(gathered_rows);
  torch::Tensor routed_out = moe_->forward_experts(
      gathered_rows, /*enable_all2all_communication=*/false, route);

  std::vector<int32_t> rows_per_ep_rank;
  rows_per_ep_rank.reserve(unique_tokens_per_ep_rank.size());
  int32_t max_rows = 0;
  for (int32_t token_count : unique_tokens_per_ep_rank) {
    const int32_t row_count = static_cast<int32_t>(token_count * row_factor);
    rows_per_ep_rank.emplace_back(row_count);
    max_rows = std::max(max_rows, row_count);
  }

  std::vector<torch::Tensor> padded_routed_parts;
  padded_routed_parts.reserve(rows_per_ep_rank.size());
  int64_t row_offset = 0;
  for (int32_t row_count : rows_per_ep_rank) {
    torch::Tensor part = routed_out.narrow(/*dim=*/0, row_offset, row_count);
    if (row_count < max_rows) {
      torch::Tensor padding = torch::zeros(
          {max_rows - row_count, routed_out.size(-1)}, routed_out.options());
      part = torch::cat({part, padding}, /*dim=*/0);
    }
    padded_routed_parts.emplace_back(part);
    row_offset += row_count;
  }
  torch::Tensor padded_routed =
      torch::cat(padded_routed_parts, /*dim=*/0).contiguous();
  torch::Tensor unique_output =
      parallel_state::reduce_scatter(padded_routed, ep_group);
  const int32_t local_unique_rows =
      rows_per_ep_rank[static_cast<size_t>(ep_group->rank())];
  unique_output = unique_output.narrow(/*dim=*/0, 0, local_unique_rows);

  const int64_t shared_offset =
      std::accumulate(rows_per_ep_rank.begin(),
                      rows_per_ep_rank.begin() + ep_group->rank(),
                      int64_t{0});
  if (shared_out.defined()) {
    unique_output.add_(shared_out.narrow(
        /*dim=*/0, shared_offset, local_unique_rows));
  }

  gathered_shape[0] = local_unique_tokens;
  unique_output = unique_output.reshape(gathered_shape);
  std::vector<int32_t> tp_token_counts;
  tp_token_counts.reserve(static_cast<size_t>(tp_size));
  for (int32_t rank = 0; rank < tp_size; ++rank) {
    tp_token_counts.emplace_back(
        split_range(local_tokens, tp_size, rank).second);
  }
  return parallel_state::gather(unique_output, tp_group, tp_token_counts);
}

}  // namespace layer
}  // namespace xllm
