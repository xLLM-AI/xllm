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

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace xllm::layer::dsv4_eplb {

inline torch::Tensor select_decode_token_mask(
    const torch::Tensor& global_decode_token_mask,
    int64_t routed_token_count,
    const std::vector<int32_t>& dp_token_counts,
    int32_t dp_rank,
    bool routed_tokens_are_dp_gathered) {
  CHECK(global_decode_token_mask.defined());
  torch::Tensor flat_mask = global_decode_token_mask.reshape({-1});
  if (routed_tokens_are_dp_gathered || dp_token_counts.size() <= 1) {
    CHECK_EQ(flat_mask.numel(), routed_token_count)
        << "EPLB gathered decode mask must align with routed tokens.";
    return flat_mask;
  }
  CHECK(dp_rank >= 0 && dp_rank < static_cast<int32_t>(dp_token_counts.size()))
      << "EPLB decode mask DP rank is out of range.";
  const int64_t global_token_count = std::accumulate(
      dp_token_counts.begin(), dp_token_counts.end(), int64_t{0});
  CHECK_EQ(flat_mask.numel(), global_token_count)
      << "EPLB global decode mask size does not match DP token counts.";
  const int64_t real_local_token_count =
      static_cast<int64_t>(dp_token_counts[static_cast<size_t>(dp_rank)]);
  CHECK_GE(routed_token_count, real_local_token_count)
      << "EPLB local routed token count is smaller than its DP mask slice.";
  const int64_t begin = std::accumulate(
      dp_token_counts.begin(), dp_token_counts.begin() + dp_rank, int64_t{0});
  torch::Tensor local_mask =
      flat_mask.slice(/*dim=*/0, begin, begin + real_local_token_count);
  if (routed_token_count == real_local_token_count) {
    return local_mask;
  }
  torch::Tensor padding_mask = torch::zeros(
      {routed_token_count - real_local_token_count}, flat_mask.options());
  return torch::cat({local_mask, padding_mask}, /*dim=*/0);
}

inline torch::Tensor select_ep2_active_token_mask(
    const torch::Tensor& global_decode_token_mask,
    int64_t routed_token_count,
    const std::vector<int32_t>& dp_token_counts,
    int32_t dp_rank,
    bool all_dp_ranks_decode) {
  if (!all_dp_ranks_decode) {
    return torch::Tensor();
  }
  return select_decode_token_mask(global_decode_token_mask,
                                  routed_token_count,
                                  dp_token_counts,
                                  dp_rank,
                                  /*routed_tokens_are_dp_gathered=*/false);
}

inline torch::Tensor apply_decode_token_mask(
    const torch::Tensor& valid_weights,
    const torch::Tensor& decode_token_mask,
    int64_t topk) {
  CHECK_EQ(valid_weights.dim(), 1);
  CHECK_GT(topk, 0);
  CHECK_EQ(valid_weights.numel(), decode_token_mask.numel() * topk)
      << "EPLB decode mask must have one entry per routed token.";
  torch::Tensor expanded_mask =
      decode_token_mask.reshape({-1, 1}).expand({-1, topk}).reshape({-1});
  return valid_weights * expanded_mask.to(valid_weights.scalar_type());
}

inline torch::Tensor select_recorded_load_token_mask(
    const torch::Tensor& global_decode_token_mask,
    int64_t routed_token_count,
    const std::vector<int32_t>& dp_token_counts,
    int32_t dp_rank,
    bool routed_tokens_are_dp_gathered,
    bool decode_only,
    bool enable_graph) {
  if (!decode_only && !enable_graph) {
    return torch::Tensor();
  }
  CHECK(global_decode_token_mask.defined())
      << "EPLB filtered load requires a per-token decode mask.";
  return select_decode_token_mask(global_decode_token_mask,
                                  routed_token_count,
                                  dp_token_counts,
                                  dp_rank,
                                  routed_tokens_are_dp_gathered);
}

inline void record_dispatch_expert_load(
    const torch::Tensor& expert_token_counts,
    const torch::Tensor& expert_load_data,
    int32_t layer_id,
    bool is_graph_warmup = false) {
  if (is_graph_warmup) {
    return;
  }
  CHECK(expert_token_counts.defined());
  CHECK_EQ(expert_token_counts.dim(), 1);
  CHECK(expert_token_counts.scalar_type() == torch::kInt32 ||
        expert_token_counts.scalar_type() == torch::kInt64);
  CHECK(expert_load_data.defined());
  CHECK_EQ(expert_load_data.dim(), 2);
  CHECK_GE(layer_id, 0);
  CHECK_LT(layer_id, expert_load_data.size(0));
  CHECK_EQ(expert_token_counts.size(0), expert_load_data.size(1));
  expert_load_data.select(/*dim=*/0, layer_id)
      .copy_(expert_token_counts.to(torch::kInt64).cumsum(/*dim=*/0));
}

}  // namespace xllm::layer::dsv4_eplb
