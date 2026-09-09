/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "framework/parallel_state/parallel_state.h"
#include "framework/parallel_state/process_group.h"

namespace xllm::layer::mlu_v4_cp {

struct DeepseekV4CpGeometry {
  std::vector<std::vector<int64_t>> rows_by_rank;
  std::vector<std::vector<int64_t>> scatter_rows_by_rank;
  std::vector<std::vector<int32_t>> front_seq_lens_by_rank;
  std::vector<std::vector<int32_t>> back_seq_lens_by_rank;
  std::vector<int32_t> tokens_per_rank;
  std::vector<int64_t> restore_indices;
  int32_t total_tokens = 0;
};

struct DeepseekV4CpContext {
  DeepseekV4CpGeometry geometry;
  torch::Tensor local_row_indices;
  torch::Tensor restore_indices;
  torch::Tensor front_cu_seq_lens;
  torch::Tensor back_cu_seq_lens;
  torch::Tensor global_positions;
  torch::Tensor local_positions;
  int32_t cp_rank = 0;
  ProcessGroup* cp_group = nullptr;
};

std::vector<int32_t> query_lengths_from_cumulative(
    const std::vector<int32_t>& cumulative_q_seq_lens,
    int64_t total_tokens);

bool should_enable_zigzag_cp(const std::vector<int32_t>& q_seq_lens,
                             int32_t cp_size);

DeepseekV4CpGeometry build_cp_geometry(int32_t cp_size,
                                       const std::vector<int32_t>& q_seq_lens);

std::optional<DeepseekV4CpContext> build_cp_context(
    int32_t cp_size,
    int32_t cp_rank,
    ProcessGroup* cp_group,
    const std::vector<int32_t>& q_seq_lens,
    const torch::Tensor& global_positions);

torch::Tensor shard_rows(const torch::Tensor& global_tensor,
                         const DeepseekV4CpContext& context);

parallel_state::GatherAsyncCtx launch_gather_rows(
    const torch::Tensor& local_tensor,
    const DeepseekV4CpContext& context);

torch::Tensor finish_gather_restore(
    parallel_state::GatherAsyncCtx gather_context,
    const DeepseekV4CpContext& context);

torch::Tensor gather_restore(const torch::Tensor& local_tensor,
                             const DeepseekV4CpContext& context);

}  // namespace xllm::layer::mlu_v4_cp
