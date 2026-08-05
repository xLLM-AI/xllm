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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace xllm {
namespace eplb {

inline int32_t effective_device_num(int32_t worker_num, int32_t ep_size) {
  CHECK_GT(worker_num, 0) << "EPLB worker_num must be positive.";
  CHECK_GT(ep_size, 0) << "EPLB ep_size must be positive.";
  CHECK_EQ(worker_num, ep_size)
      << "EPLB currently requires ep_size == worker_num, got ep_size="
      << ep_size << ", worker_num=" << worker_num;
  return worker_num;
}

inline int32_t local_physical_experts_num(int32_t num_logical_experts,
                                          int32_t eplb_device_num,
                                          int32_t redundant_experts_num) {
  CHECK_GT(num_logical_experts, 0)
      << "EPLB num_logical_experts must be positive.";
  CHECK_GT(eplb_device_num, 0) << "EPLB device_num must be positive.";
  CHECK_GE(redundant_experts_num, 0)
      << "EPLB redundant_experts_num must be non-negative.";
  CHECK_EQ(num_logical_experts % eplb_device_num, 0)
      << "EPLB logical experts must be divisible by device_num, got experts="
      << num_logical_experts << ", device_num=" << eplb_device_num;
  return num_logical_experts / eplb_device_num + redundant_experts_num;
}

inline int32_t eplb_rank_from_worker_rank(int32_t worker_rank,
                                          int32_t worker_num,
                                          int32_t eplb_device_num) {
  CHECK_GE(worker_rank, 0) << "EPLB worker_rank must be non-negative.";
  CHECK_LT(worker_rank, worker_num)
      << "EPLB worker_rank out of range, worker_rank=" << worker_rank
      << ", worker_num=" << worker_num;
  CHECK_GT(eplb_device_num, 0) << "EPLB device_num must be positive.";
  CHECK_EQ(worker_num, eplb_device_num)
      << "EPLB currently requires worker_num == device_num, got worker_num="
      << worker_num << ", device_num=" << eplb_device_num;
  return worker_rank;
}

inline torch::Tensor build_global_decode_token_mask(
    const std::vector<torch::Tensor>& local_masks,
    const std::vector<int32_t>& dp_token_counts) {
  CHECK_EQ(local_masks.size(), dp_token_counts.size())
      << "EPLB decode masks must align with DP token counts.";
  CHECK(!local_masks.empty()) << "EPLB requires at least one DP decode mask.";

  std::vector<torch::Tensor> flat_masks;
  flat_masks.reserve(local_masks.size());
  for (size_t dp_rank = 0; dp_rank < local_masks.size(); ++dp_rank) {
    CHECK(local_masks[dp_rank].defined())
        << "EPLB decode mask is undefined for DP rank " << dp_rank;
    CHECK_GE(dp_token_counts[dp_rank], 0)
        << "EPLB DP token count must be non-negative.";
    torch::Tensor flat_mask = local_masks[dp_rank].reshape({-1});
    CHECK_EQ(flat_mask.numel(), dp_token_counts[dp_rank])
        << "EPLB decode mask size mismatch for DP rank " << dp_rank;
    flat_masks.emplace_back(std::move(flat_mask));
  }
  return torch::cat(flat_masks, /*dim=*/0).contiguous();
}

inline torch::Tensor expand_decode_token_mask(
    const torch::Tensor& decode_token_mask,
    int32_t tokens_per_row) {
  CHECK_GT(tokens_per_row, 0)
      << "EPLB decode mask expansion factor must be positive.";
  if (!decode_token_mask.defined()) {
    return torch::Tensor();
  }
  torch::Tensor flat_mask = decode_token_mask.reshape({-1});
  if (tokens_per_row == 1) {
    return flat_mask;
  }
  return flat_mask.repeat_interleave(/*repeats=*/tokens_per_row, /*dim=*/0)
      .contiguous();
}

inline int32_t count_cross_device_migrations(
    const std::vector<int64_t>& previous_assignment,
    const std::vector<int64_t>& next_assignment,
    int32_t device_experts_num) {
  CHECK_GT(device_experts_num, 0);
  CHECK_EQ(previous_assignment.size(), next_assignment.size());
  CHECK_EQ(previous_assignment.size() % static_cast<size_t>(device_experts_num),
           0);
  int32_t migration_count = 0;
  for (size_t slot = 0; slot < next_assignment.size(); ++slot) {
    const size_t device = slot / static_cast<size_t>(device_experts_num);
    const size_t device_begin =
        device * static_cast<size_t>(device_experts_num);
    const size_t device_end =
        device_begin + static_cast<size_t>(device_experts_num);
    const int64_t target_expert = next_assignment[slot];
    const bool resident =
        std::find(previous_assignment.begin() + device_begin,
                  previous_assignment.begin() + device_end,
                  target_expert) != previous_assignment.begin() + device_end;
    if (!resident) {
      ++migration_count;
    }
  }
  return migration_count;
}

}  // namespace eplb
}  // namespace xllm
