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

#include "layers/mlu/deepseek_v4/deepseek_v4_cp_context.h"

#include <glog/logging.h>

#include <algorithm>
#include <numeric>
#include <utility>

namespace xllm::layer::mlu_v4_cp {
namespace {

torch::Tensor make_cu_seq_lens(const std::vector<int32_t>& seq_lens,
                               const torch::TensorOptions& options) {
  std::vector<int32_t> cu_seq_lens = {0};
  cu_seq_lens.reserve(seq_lens.size() + 1);
  int32_t total_tokens = 0;
  for (int32_t seq_len : seq_lens) {
    total_tokens += seq_len;
    cu_seq_lens.emplace_back(total_tokens);
  }
  return torch::tensor(cu_seq_lens, options);
}

}  // namespace

std::vector<int32_t> query_lengths_from_cumulative(
    const std::vector<int32_t>& cumulative_q_seq_lens,
    int64_t total_tokens) {
  CHECK_GE(cumulative_q_seq_lens.size(), 2u)
      << "MLU query lengths require cumulative endpoints with a leading zero.";
  CHECK_EQ(cumulative_q_seq_lens.front(), 0)
      << "MLU cumulative query lengths must start at zero.";
  CHECK_EQ(cumulative_q_seq_lens.back(), total_tokens)
      << "MLU cumulative query lengths must end at the current token count.";

  std::vector<int32_t> query_lengths;
  query_lengths.reserve(cumulative_q_seq_lens.size() - 1);
  for (size_t index = 1; index < cumulative_q_seq_lens.size(); ++index) {
    const int32_t query_length =
        cumulative_q_seq_lens[index] - cumulative_q_seq_lens[index - 1];
    CHECK_GE(query_length, 0)
        << "MLU cumulative query lengths must be non-decreasing.";
    query_lengths.emplace_back(query_length);
  }
  return query_lengths;
}

bool should_enable_zigzag_cp(const std::vector<int32_t>& q_seq_lens,
                             int32_t cp_size) {
  if (cp_size <= 1 || q_seq_lens.empty()) {
    return false;
  }
  const int32_t max_query_len =
      *std::max_element(q_seq_lens.begin(), q_seq_lens.end());
  return max_query_len >= 2 * cp_size;
}

DeepseekV4CpGeometry build_cp_geometry(int32_t cp_size,
                                       const std::vector<int32_t>& q_seq_lens) {
  CHECK_GT(cp_size, 1) << "DeepSeek V4 CP requires cp_size > 1.";
  CHECK(!q_seq_lens.empty())
      << "DeepSeek V4 CP requires at least one prefill request.";
  CHECK(should_enable_zigzag_cp(q_seq_lens, cp_size))
      << "DeepSeek V4 CP requires a query with at least 2 * cp_size tokens.";

  DeepseekV4CpGeometry geometry;
  geometry.total_tokens =
      std::accumulate(q_seq_lens.begin(), q_seq_lens.end(), int32_t{0});
  geometry.rows_by_rank.resize(static_cast<size_t>(cp_size));
  geometry.scatter_rows_by_rank.resize(static_cast<size_t>(cp_size));
  geometry.front_seq_lens_by_rank.assign(
      static_cast<size_t>(cp_size), std::vector<int32_t>(q_seq_lens.size(), 0));
  geometry.back_seq_lens_by_rank.assign(
      static_cast<size_t>(cp_size), std::vector<int32_t>(q_seq_lens.size(), 0));

  std::vector<std::vector<int64_t>> front_rows(static_cast<size_t>(cp_size));
  std::vector<std::vector<int64_t>> back_rows(static_cast<size_t>(cp_size));
  int64_t request_offset = 0;
  const int32_t bucket_count = 2 * cp_size;
  for (size_t request_idx = 0; request_idx < q_seq_lens.size(); ++request_idx) {
    const int32_t query_len = q_seq_lens[request_idx];
    CHECK_GE(query_len, 0) << "query lengths must be non-negative.";
    const int32_t bucket_size =
        query_len == 0 ? 0 : (query_len + bucket_count - 1) / bucket_count;
    for (int32_t rank = 0; rank < cp_size; ++rank) {
      const int32_t front_start = rank * bucket_size;
      const int32_t back_start = (bucket_count - rank - 1) * bucket_size;
      const int32_t front_count =
          std::clamp(query_len - front_start, 0, bucket_size);
      const int32_t back_count =
          std::clamp(query_len - back_start, 0, bucket_size);
      geometry.front_seq_lens_by_rank[static_cast<size_t>(rank)][request_idx] =
          front_count;
      geometry.back_seq_lens_by_rank[static_cast<size_t>(rank)][request_idx] =
          back_count;
      for (int32_t token = 0; token < front_count; ++token) {
        front_rows[static_cast<size_t>(rank)].emplace_back(request_offset +
                                                           front_start + token);
      }
      for (int32_t token = 0; token < back_count; ++token) {
        back_rows[static_cast<size_t>(rank)].emplace_back(request_offset +
                                                          back_start + token);
      }
    }
    request_offset += query_len;
  }

  geometry.tokens_per_rank.reserve(static_cast<size_t>(cp_size));
  for (int32_t rank = 0; rank < cp_size; ++rank) {
    std::vector<int64_t>& rank_front = front_rows[static_cast<size_t>(rank)];
    std::vector<int64_t>& rank_back = back_rows[static_cast<size_t>(rank)];
    std::vector<int64_t>& rank_rows =
        geometry.rows_by_rank[static_cast<size_t>(rank)];
    std::vector<int64_t>& rank_scatter =
        geometry.scatter_rows_by_rank[static_cast<size_t>(rank)];

    const bool front_is_fake = rank_front.empty();
    if (front_is_fake) {
      rank_front.emplace_back(0);
      geometry.front_seq_lens_by_rank[static_cast<size_t>(rank)][0] = 1;
    }
    const bool back_is_fake = rank_back.empty();
    if (back_is_fake) {
      rank_back.emplace_back(0);
      geometry.back_seq_lens_by_rank[static_cast<size_t>(rank)][0] = 1;
    }

    rank_rows.insert(rank_rows.end(), rank_front.begin(), rank_front.end());
    rank_rows.insert(rank_rows.end(), rank_back.begin(), rank_back.end());
    rank_scatter = rank_rows;
    if (front_is_fake) {
      rank_scatter[0] = geometry.total_tokens;
    }
    if (back_is_fake) {
      rank_scatter.back() = geometry.total_tokens;
    }
    geometry.tokens_per_rank.emplace_back(
        static_cast<int32_t>(rank_rows.size()));
  }

  geometry.restore_indices.assign(static_cast<size_t>(geometry.total_tokens),
                                  -1);
  int64_t gathered_position = 0;
  for (const std::vector<int64_t>& scatter_rows :
       geometry.scatter_rows_by_rank) {
    for (int64_t global_row : scatter_rows) {
      if (global_row < geometry.total_tokens) {
        CHECK_EQ(geometry.restore_indices[static_cast<size_t>(global_row)], -1)
            << "DeepSeek V4 CP assigned a token to more than one rank.";
        geometry.restore_indices[static_cast<size_t>(global_row)] =
            gathered_position;
      }
      ++gathered_position;
    }
  }
  for (int64_t restore_position : geometry.restore_indices) {
    CHECK_GE(restore_position, 0)
        << "DeepSeek V4 CP did not assign every global token.";
  }
  return geometry;
}

std::optional<DeepseekV4CpContext> build_cp_context(
    int32_t cp_size,
    int32_t cp_rank,
    ProcessGroup* cp_group,
    const std::vector<int32_t>& q_seq_lens,
    const torch::Tensor& global_positions) {
  if (!should_enable_zigzag_cp(q_seq_lens, cp_size)) {
    return std::nullopt;
  }
  CHECK(cp_group != nullptr) << "DeepSeek V4 CP requires a CP process group.";
  CHECK_EQ(cp_group->world_size(), cp_size)
      << "CP process group size must match cp_size.";
  CHECK_GE(cp_rank, 0);
  CHECK_LT(cp_rank, cp_size);

  DeepseekV4CpContext context;
  context.geometry = build_cp_geometry(cp_size, q_seq_lens);
  CHECK(global_positions.defined())
      << "DeepSeek V4 CP requires global positions.";
  CHECK_EQ(global_positions.dim(), 1)
      << "DeepSeek V4 CP positions must be one-dimensional.";
  CHECK_EQ(global_positions.size(0), context.geometry.total_tokens)
      << "DeepSeek V4 CP positions must match the total query token count.";
  context.cp_rank = cp_rank;
  context.cp_group = cp_group;
  context.global_positions = global_positions;
  const torch::Device device = global_positions.device();
  const torch::TensorOptions int64_options =
      torch::TensorOptions().dtype(torch::kInt64).device(device);
  const torch::TensorOptions int32_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  context.local_row_indices =
      torch::tensor(context.geometry.rows_by_rank[static_cast<size_t>(cp_rank)],
                    int64_options);
  context.restore_indices =
      torch::tensor(context.geometry.restore_indices, int64_options);
  context.front_cu_seq_lens = make_cu_seq_lens(
      context.geometry.front_seq_lens_by_rank[static_cast<size_t>(cp_rank)],
      int32_options);
  context.back_cu_seq_lens = make_cu_seq_lens(
      context.geometry.back_seq_lens_by_rank[static_cast<size_t>(cp_rank)],
      int32_options);
  context.local_positions = global_positions.index_select(
      /*dim=*/0, context.local_row_indices);
  return context;
}

torch::Tensor shard_rows(const torch::Tensor& global_tensor,
                         const DeepseekV4CpContext& context) {
  if (!global_tensor.defined()) {
    return global_tensor;
  }
  return global_tensor.index_select(/*dim=*/0, context.local_row_indices);
}

parallel_state::GatherAsyncCtx launch_gather_rows(
    const torch::Tensor& local_tensor,
    const DeepseekV4CpContext& context) {
  if (!local_tensor.defined()) {
    return {};
  }
  return parallel_state::launch_gather(local_tensor.contiguous(),
                                       context.cp_group,
                                       context.geometry.tokens_per_rank);
}

torch::Tensor finish_gather_restore(
    parallel_state::GatherAsyncCtx gather_context,
    const DeepseekV4CpContext& context) {
  if (!gather_context.stacked.defined()) {
    return torch::Tensor();
  }
  torch::Tensor gathered =
      parallel_state::finish_gather(std::move(gather_context));
  return gathered.index_select(/*dim=*/0, context.restore_indices);
}

torch::Tensor gather_restore(const torch::Tensor& local_tensor,
                             const DeepseekV4CpContext& context) {
  if (!local_tensor.defined()) {
    return local_tensor;
  }
  return finish_gather_restore(launch_gather_rows(local_tensor, context),
                               context);
}

}  // namespace xllm::layer::mlu_v4_cp
