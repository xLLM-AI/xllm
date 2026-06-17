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

#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/framework/multimodal/mm_batch_data.h"
#include "core/framework/parallel_state/parallel_state.h"

namespace xllm {

namespace {
int32_t compute_max_token_total(
    const std::vector<std::vector<int32_t>>& rank_to_items,
    const std::vector<int32_t>& token_nums) {
  auto token_total = [&](const std::vector<int32_t>& items) {
    int32_t total = 0;
    for (int32_t idx : items) {
      total += token_nums[idx];
    }
    return total;
  };
  int32_t max_total = 0;
  for (const auto& items : rank_to_items) {
    max_total = std::max(max_total, token_total(items));
  }
  return max_total;
}
}  // namespace

class ProcessGroup;

std::vector<int32_t> get_mm_token_nums(MMBatchData& mm_data, MMType type);

std::vector<torch::Tensor> split_by_token_nums(
    const torch::Tensor& tensor,
    const std::vector<int32_t>& token_nums);

std::vector<std::vector<int32_t>> build_lpt_assignment(
    const std::vector<int32_t>& patch_nums,
    int32_t world_size);

std::vector<int32_t> compute_patch_nums(const torch::Tensor& grid_thw);

std::pair<torch::Tensor, torch::Tensor> slice_local_input(
    const torch::Tensor& pixel_values,
    const torch::Tensor& grid_thw,
    const std::vector<int32_t>& patch_nums,
    const std::vector<int32_t>& local_index);

torch::Tensor all_gather_padded(const torch::Tensor& local_embeds,
                                int32_t max_token_total,
                                ProcessGroup* dp_group);

std::vector<torch::Tensor> restore_outputs(
    const torch::Tensor& gathered,
    const std::vector<int32_t>& token_nums,
    const std::vector<std::vector<int32_t>>& rank_to_items,
    int32_t max_token_total);

template <typename VisionEncoder>
std::vector<torch::Tensor> run_dp_encoder(
    VisionEncoder& visual,
    const torch::Tensor& pixel_values,
    const torch::Tensor& grid_thw,
    const std::vector<int32_t>& token_nums,
    int32_t feature_dim,
    ProcessGroup* dp_group) {
  CHECK(dp_group != nullptr) << "dp_group cannot be null.";
  std::vector<int32_t> patch_nums = compute_patch_nums(grid_thw);
  std::vector<std::vector<int32_t>> rank_to_items =
      build_lpt_assignment(patch_nums, dp_group->world_size());

  const int32_t local_rank = dp_group->rank();
  std::pair<torch::Tensor, torch::Tensor> local_input = slice_local_input(
      pixel_values, grid_thw, patch_nums, rank_to_items[local_rank]);
  torch::Tensor local_embeds;
  if (!rank_to_items[local_rank].empty()) {
    local_embeds = visual(local_input.first, local_input.second);
  } else {
    local_embeds = torch::empty({0, feature_dim}, pixel_values.options());
  }

  int32_t max_total = compute_max_token_total(rank_to_items, token_nums);
  torch::Tensor gathered = all_gather_padded(local_embeds, max_total, dp_group);
  return restore_outputs(gathered, token_nums, rank_to_items, max_total);
}

}  // namespace xllm
