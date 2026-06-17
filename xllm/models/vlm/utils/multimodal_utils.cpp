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

#include "models/vlm/utils/multimodal_utils.h"

#include <glog/logging.h>

#include <algorithm>
#include <numeric>

#include "core/framework/multimodal/mm_visitor.h"

namespace xllm {

std::vector<int32_t> get_mm_token_nums(MMBatchData& mm_data, MMType type) {
  MMTokenNumVisitor visitor(type);
  mm_data.foreach (visitor);
  return visitor.token_nums();
}

std::vector<std::vector<int32_t>> build_lpt_assignment(
    const std::vector<int32_t>& patch_nums,
    int32_t world_size) {
  const int32_t item_count = static_cast<int32_t>(patch_nums.size());
  std::vector<std::vector<int32_t>> rank_to_items(world_size);
  std::vector<int32_t> item_order(item_count);
  std::iota(item_order.begin(), item_order.end(), 0);
  auto larger_patch_num = [&patch_nums](int32_t lhs, int32_t rhs) {
    if (patch_nums[lhs] != patch_nums[rhs]) {
      return patch_nums[lhs] > patch_nums[rhs];
    }
    return lhs < rhs;
  };
  std::sort(item_order.begin(), item_order.end(), larger_patch_num);

  std::vector<int32_t> rank_loads(world_size, 0);
  for (int32_t item_index : item_order) {
    const auto target_it =
        std::min_element(rank_loads.begin(), rank_loads.end());
    const int32_t target_rank =
        static_cast<int32_t>(std::distance(rank_loads.begin(), target_it));
    rank_to_items[target_rank].push_back(item_index);
    rank_loads[target_rank] += patch_nums[item_index];
  }

  for (std::vector<int32_t>& item_indices : rank_to_items) {
    std::sort(item_indices.begin(), item_indices.end());
  }
  return rank_to_items;
}

std::vector<int32_t> compute_patch_nums(const torch::Tensor& grid_thw) {
  torch::Tensor grid_thw_cpu = grid_thw.cpu();
  const int64_t item_count = grid_thw_cpu.size(0);
  std::vector<int32_t> patch_nums;
  patch_nums.reserve(item_count);
  for (int64_t item_index = 0; item_index < item_count; ++item_index) {
    patch_nums.push_back(grid_thw_cpu[item_index].prod().item<int32_t>());
  }
  return patch_nums;
}

std::pair<torch::Tensor, torch::Tensor> slice_local_input(
    const torch::Tensor& pixel_values,
    const torch::Tensor& grid_thw,
    const std::vector<int32_t>& patch_nums,
    const std::vector<int32_t>& local_index) {
  if (local_index.empty()) {
    return std::make_pair(pixel_values.slice(0, 0, 0), grid_thw.slice(0, 0, 0));
  }

  std::vector<int64_t> patch_offsets(patch_nums.size() + 1, 0);
  for (size_t item_index = 0; item_index < patch_nums.size(); ++item_index) {
    patch_offsets[item_index + 1] =
        patch_offsets[item_index] +
        static_cast<int64_t>(patch_nums[item_index]);
  }

  std::vector<torch::Tensor> pixel_slices;
  std::vector<torch::Tensor> grid_slices;
  pixel_slices.reserve(local_index.size());
  grid_slices.reserve(local_index.size());
  for (int32_t item_index : local_index) {
    pixel_slices.push_back(pixel_values.slice(
        0, patch_offsets[item_index], patch_offsets[item_index + 1]));
    grid_slices.push_back(grid_thw.slice(0, item_index, item_index + 1));
  }

  return std::make_pair(torch::cat(pixel_slices, 0),
                        torch::cat(grid_slices, 0));
}

std::vector<torch::Tensor> split_by_token_nums(
    const torch::Tensor& tensor,
    const std::vector<int32_t>& token_nums) {
  std::vector<int64_t> split_sizes(token_nums.begin(), token_nums.end());
  return tensor.split(split_sizes, 0);
}

torch::Tensor all_gather_padded(const torch::Tensor& local_embeds,
                                int32_t max_token_total,
                                ProcessGroup* dp_group) {
  torch::Tensor padded = torch::zeros({max_token_total, local_embeds.size(1)},
                                      local_embeds.options());
  padded.slice(0, 0, local_embeds.size(0)).copy_(local_embeds);
  return parallel_state::gather(padded, dp_group, 0);
}

std::vector<torch::Tensor> restore_outputs(
    const torch::Tensor& gathered,
    const std::vector<int32_t>& token_nums,
    const std::vector<std::vector<int32_t>>& rank_to_items,
    int32_t max_token_total) {
  std::vector<torch::Tensor> outputs(token_nums.size());
  const int32_t world_size = static_cast<int32_t>(rank_to_items.size());
  for (int32_t rank = 0; rank < world_size; ++rank) {
    int32_t offset_in_rank = 0;
    for (int32_t item_idx : rank_to_items[rank]) {
      outputs[item_idx] = gathered.slice(
          0,
          rank * max_token_total + offset_in_rank,
          rank * max_token_total + offset_in_rank + token_nums[item_idx]);
      offset_in_rank += token_nums[item_idx];
    }
  }
  return outputs;
}

}  // namespace xllm
