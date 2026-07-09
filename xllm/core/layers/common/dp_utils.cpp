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

#include "dp_utils.h"

#include <algorithm>
#include <numeric>

#include "framework/parallel_state/parallel_state.h"

namespace xllm {
namespace layer {

namespace {

struct GatherPlan {
  int64_t padded_dp_tokens = 0;
  int64_t shard_tokens = 0;
};

bool are_dp_tokens_equal(const std::vector<int32_t>& dp_tokens) {
  return !dp_tokens.empty() &&
         std::all_of(dp_tokens.begin(),
                     dp_tokens.end(),
                     [first_token_num = dp_tokens.front()](int32_t num_tokens) {
                       return num_tokens == first_token_num;
                     });
}

int64_t align_tokens(int64_t tokens, int64_t align) {
  CHECK_GT(align, 0) << "align must be positive";
  int64_t rem = tokens % align;
  return rem == 0 ? tokens : tokens + align - rem;
}

int64_t get_tp_size(const ParallelArgs& args) {
  return args.tp_group_ ? args.tp_group_->world_size() : 1;
}

GatherPlan build_gather_plan(const std::vector<int32_t>& dp_tokens,
                             int64_t tp_size) {
  CHECK(!dp_tokens.empty()) << "dp_tokens is empty";
  CHECK_GT(tp_size, 0) << "tp_size must be positive";

  int64_t max_dp_tokens = *std::max_element(dp_tokens.begin(), dp_tokens.end());
  int64_t padded_dp_tokens =
      align_tokens(std::max(max_dp_tokens, tp_size), tp_size);
  return {padded_dp_tokens, padded_dp_tokens / tp_size};
}

}  // namespace

std::pair<torch::Tensor, PaddingInfo> pad_tokens(torch::Tensor x,
                                                 int64_t target_tokens) {
  int64_t current_tokens = x.size(0);
  CHECK_GE(target_tokens, current_tokens)
      << "target_tokens " << target_tokens << " must be >= current_tokens "
      << current_tokens;

  if (target_tokens == current_tokens) {
    return {x, {current_tokens, current_tokens, false}};
  }

  int64_t pad_len = target_tokens - current_tokens;
  auto pad_shape = x.sizes().vec();
  pad_shape[0] = pad_len;
  torch::Tensor pad_tensor = torch::zeros(pad_shape, x.options());
  torch::Tensor padded_x = torch::cat({x, pad_tensor}, 0);
  return {padded_x, {current_tokens, target_tokens, true}};
}

int64_t get_reduce_scatter_tokens(int64_t num_tokens,
                                  const ParallelArgs& parallel_args) {
  int64_t tp_size = get_tp_size(parallel_args);
  if (tp_size <= 1) {
    return num_tokens;
  }

  return align_tokens(num_tokens, tp_size);
}

std::pair<torch::Tensor, PaddingInfo> reduce_scatter_attn_input(
    torch::Tensor x,
    const torch::Tensor& residual,
    int64_t target_tokens,
    const ParallelArgs& parallel_args) {
  if (parallel_args.tp_group_ && parallel_args.tp_group_->rank() == 0) {
    x = x + residual;
  }

  auto pad_result = pad_tokens(x, target_tokens);
  x = pad_result.first;

  if (parallel_args.tp_group_ && parallel_args.tp_group_->world_size() > 1) {
    x = parallel_state::reduce_scatter(x, parallel_args.tp_group_);
  }

  return {x, pad_result.second};
}

int64_t get_dp_gather_tokens(const std::vector<int32_t>& dp_tokens,
                             const ParallelArgs& args) {
  return build_gather_plan(dp_tokens, get_tp_size(args)).padded_dp_tokens;
}

torch::Tensor gather_global_tokens(const torch::Tensor& input,
                                   const std::vector<int32_t>& dp_tokens,
                                   const ParallelArgs& args) {
  if (!args.process_group_ || args.process_group_->world_size() == 1) {
    return input;
  }

  CHECK(args.tp_group_ != nullptr) << "tp_group_ is not initialized";
  const int64_t tp_size = get_tp_size(args);
  const GatherPlan plan = build_gather_plan(dp_tokens, tp_size);
  const int64_t world_size = args.process_group_->world_size();
  CHECK_EQ(world_size, static_cast<int64_t>(dp_tokens.size()) * tp_size)
      << "world_size " << world_size << " does not match dp_size "
      << dp_tokens.size() << " * tp_size " << tp_size;
  CHECK_EQ(plan.padded_dp_tokens % tp_size, 0)
      << "padded_dp_tokens " << plan.padded_dp_tokens
      << " must be divisible by tp_size " << tp_size;
  CHECK_EQ(plan.shard_tokens, plan.padded_dp_tokens / tp_size)
      << "invalid shard_tokens " << plan.shard_tokens;
  CHECK_EQ(input.size(0), plan.shard_tokens)
      << "input tokens " << input.size(0) << " must match shard_tokens "
      << plan.shard_tokens;

  torch::Tensor gathered_tensors =
      args.process_group_->allgather_base_sync(input);
  torch::Tensor flat_gathered = gathered_tensors.flatten(0, 1).contiguous();

  int64_t total_tokens =
      std::accumulate(dp_tokens.begin(), dp_tokens.end(), int64_t{0});
  if (are_dp_tokens_equal(dp_tokens) && flat_gathered.size(0) == total_tokens) {
    return flat_gathered;
  }

  std::vector<int64_t> valid_rows;
  valid_rows.reserve(total_tokens);
  for (size_t dp_rank = 0; dp_rank < dp_tokens.size(); ++dp_rank) {
    int64_t valid_tokens = dp_tokens[dp_rank];
    int64_t rank_offset = static_cast<int64_t>(dp_rank) * plan.padded_dp_tokens;
    CHECK_LE(valid_tokens, plan.padded_dp_tokens)
        << "dp rank " << dp_rank << " valid_tokens " << valid_tokens
        << " exceeds padded_dp_tokens " << plan.padded_dp_tokens;
    for (int64_t token_idx = 0; token_idx < valid_tokens; ++token_idx) {
      valid_rows.push_back(rank_offset + token_idx);
    }
  }

  torch::Tensor row_index = torch::tensor(valid_rows,
                                          torch::TensorOptions()
                                              .dtype(torch::kLong)
                                              .device(flat_gathered.device()));
  return flat_gathered.index_select(0, row_index);
}

torch::Tensor unpad_tokens(torch::Tensor x, const PaddingInfo& pad_info) {
  if (!pad_info.active) {
    return x;
  }

  return x.slice(0, 0, pad_info.original_tokens);
}

bool need_dp_moe_gather(const ParallelArgs& args, bool enable_moe_all2all) {
  return args.dp_size() > 1 && args.ep_size() > 1 && !enable_moe_all2all;
}

bool need_selected_moe_dp_gather(const ParallelArgs& args) {
  return args.dp_size() > 1 && args.ep_size() > 1;
}

torch::Tensor gather_dp_tokens(const torch::Tensor& input,
                               const ModelInputParams& params,
                               const ParallelArgs& args) {
  if (args.dp_size() <= 1) {
    return input;
  }

  return parallel_state::gather(input,
                                args.dp_local_process_group_,
                                params.parallel.dp_global_token_nums);
}

SelectedMoeInputs gather_selected_moe_inputs(
    const torch::Tensor& hidden_states,
    const torch::Tensor& topk_weights,
    const torch::Tensor& topk_ids,
    const std::vector<int32_t>& dp_token_nums,
    const ParallelArgs& args) {
  if (!need_selected_moe_dp_gather(args)) {
    return {hidden_states, topk_weights, topk_ids, false};
  }

  CHECK(args.dp_local_process_group_ != nullptr)
      << "dp_local_process_group_ is not initialized";
  const int64_t rank = args.dp_local_process_group_->rank();
  CHECK_LT(rank, static_cast<int64_t>(dp_token_nums.size()))
      << "rank " << rank << " must be less than dp_token_nums size "
      << dp_token_nums.size();
  CHECK_EQ(hidden_states.size(0), static_cast<int64_t>(dp_token_nums[rank]))
      << "selected MoE local rows " << hidden_states.size(0)
      << " must match dp_token_nums[" << rank << "] " << dp_token_nums[rank];

  return {parallel_state::gather(
              hidden_states, args.dp_local_process_group_, dp_token_nums),
          parallel_state::gather(
              topk_weights, args.dp_local_process_group_, dp_token_nums),
          parallel_state::gather(
              topk_ids, args.dp_local_process_group_, dp_token_nums),
          true};
}

torch::Tensor slice_selected_moe_output(
    torch::Tensor output,
    const std::vector<int32_t>& dp_token_nums,
    const ParallelArgs& args) {
  if (!need_selected_moe_dp_gather(args)) {
    return output;
  }

  CHECK(args.dp_local_process_group_ != nullptr)
      << "dp_local_process_group_ is not initialized";
  const int64_t rank = args.dp_local_process_group_->rank();
  CHECK_LT(rank, static_cast<int64_t>(dp_token_nums.size()))
      << "rank " << rank << " must be less than dp_token_nums size "
      << dp_token_nums.size();

  int64_t start = 0;
  for (int64_t i = 0; i < rank; ++i) {
    start += dp_token_nums[i];
  }
  const int64_t end = start + dp_token_nums[rank];
  return output.slice(0, start, end);
}

torch::Tensor get_dp_local_slice(const torch::Tensor& input,
                                 const ModelInputParams& params,
                                 const ParallelArgs& args) {
  if (args.dp_size() <= 1) {
    return input;
  }

  const auto& dp_tokens = params.parallel.dp_global_token_nums;
  const int64_t dp_rank = args.dp_local_process_group_->rank();

  int64_t start = 0;
  for (int64_t i = 0; i < dp_rank; ++i) {
    start += dp_tokens[i];
  }
  int64_t end = start + dp_tokens[dp_rank];

  return input.slice(0, start, end);
}

bool all_dp_ranks_are_decode(const ModelInputParams& params) {
  if (params.parallel.dp_is_decode.empty()) {
    return params.parallel.dp_global_token_nums.size() <= 1;
  }

  return std::all_of(params.parallel.dp_is_decode.begin(),
                     params.parallel.dp_is_decode.end(),
                     [](int32_t val) { return val == 1; });
}

}  // namespace layer
}  // namespace xllm
