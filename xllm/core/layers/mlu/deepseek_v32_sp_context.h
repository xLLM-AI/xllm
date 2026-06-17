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
#include <numeric>
#include <optional>
#include <vector>

#include "core/common/global_flags.h"
#include "core/framework/config/parallel_config.h"
#include "framework/batch/batch_forward_type.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/parallel_state/process_group.h"
#include "layers/mlu/deepseek_v32_sp_plan.h"

namespace xllm::layer::v32_sp {

using PaddedGatherHandle = xllm::parallel_state::GatherAsyncCtx;

inline torch::Tensor make_sp_prefix(const std::vector<int32_t>& seq_lens,
                                    const torch::TensorOptions& options) {
  std::vector<int32_t> cu_lens = {0};
  cu_lens.reserve(seq_lens.size() + 1);
  int32_t token_num = 0;
  for (int32_t seq_len : seq_lens) {
    token_num += seq_len;
    cu_lens.push_back(token_num);
  }
  return torch::tensor(cu_lens, options);
}

inline AttentionMetadata build_local_prefill_attention_metadata(
    const AttentionMetadata& base_attn_metadata,
    const std::vector<DeepseekV32SPSegment>& segments) {
  const auto int32_options =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  std::vector<int32_t> seg_q_tokens;
  std::vector<int32_t> seg_suffix_k_lens;
  seg_q_tokens.reserve(segments.size());
  seg_suffix_k_lens.reserve(segments.size());

  int32_t max_query_len = 0;
  int32_t max_seq_len = 0;
  for (const auto& segment : segments) {
    CHECK_LE(segment.suffix_k_len, segment.ctx_k_len)
        << "deepseek_v32 sequence parallel expects suffix_k_len <= ctx_k_len.";
    seg_q_tokens.push_back(segment.q_tokens);
    seg_suffix_k_lens.push_back(segment.suffix_k_len);
    max_query_len = std::max(max_query_len, segment.q_tokens);
    max_seq_len = std::max(max_seq_len, segment.suffix_k_len);
  }

  AttentionMetadata local_attn_metadata = base_attn_metadata;
  const torch::Device device = base_attn_metadata.q_cu_seq_lens.device();
  local_attn_metadata.q_cu_seq_lens =
      make_sp_prefix(seg_q_tokens, int32_options).to(device);
  local_attn_metadata.kv_cu_seq_lens =
      make_sp_prefix(seg_suffix_k_lens, int32_options).to(device);
  // Local SP metadata stays on the live suffix view. Cached prefix tokens are
  // addressed later through seg_ctx_lens + block_table in the indexer path.
  local_attn_metadata.kv_seq_lens =
      torch::tensor(seg_suffix_k_lens, int32_options).to(device);
  local_attn_metadata.max_query_len = max_query_len;
  local_attn_metadata.max_seq_len = max_seq_len;
  return local_attn_metadata;
}

inline std::vector<int32_t> build_seq_offsets(
    const std::vector<int32_t>& seq_lens) {
  std::vector<int32_t> offsets;
  offsets.reserve(seq_lens.size());
  int32_t offset = 0;
  for (int32_t seq_len : seq_lens) {
    offsets.push_back(offset);
    offset += seq_len;
  }
  return offsets;
}

inline torch::Tensor build_segment_length_matrix(
    const std::vector<DeepseekV32SPSegment>& segments,
    int32_t DeepseekV32SPSegment::* length_field,
    const torch::Device& device) {
  std::vector<int32_t> values;
  values.reserve(segments.size() * 2);
  for (const auto& segment : segments) {
    values.push_back(0);
    values.push_back(segment.*length_field);
  }

  auto cpu_options =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  return torch::tensor(values, cpu_options)
      .view({static_cast<int64_t>(segments.size()), 2})
      .to(device)
      .contiguous();
}

inline torch::Tensor build_segment_ctx_lens_tensor(
    const std::vector<DeepseekV32SPSegment>& segments,
    const torch::Device& device) {
  std::vector<int32_t> values;
  values.reserve(segments.size());
  for (const auto& segment : segments) {
    values.push_back(segment.ctx_k_len);
  }

  auto cpu_options =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  return torch::tensor(values, cpu_options).to(device).contiguous();
}

struct DeepseekV32SPContext {
  DeepseekV32SPCommPlan comm_plan;
  AttentionMetadata local_attn_metadata;
  BatchForwardType batch_forward_type;
  std::vector<DeepseekV32SPSegment> local_segments;
  std::vector<int32_t> seg_q_starts_cpu;
  std::vector<int32_t> req_q_offsets_cpu;
  std::vector<int32_t> req_ctx_offsets_cpu;

  torch::Tensor seg_q_cu_lens_2col;
  torch::Tensor seg_suffix_k_cu_lens_2col;
  torch::Tensor seg_ctx_k_cu_lens_2col;
  torch::Tensor seg_ctx_lens_1col;

  torch::Tensor gathered_reorder_index;
  torch::Tensor gathered_slot_mapping;

  int32_t total_tokens = 0;
  int32_t rank = 0;
  ProcessGroup* process_group = nullptr;
};

inline torch::Tensor reorder_by_index(const torch::Tensor& tensor,
                                      const torch::Tensor& reorder_index);

inline std::optional<DeepseekV32SPContext> build_deepseek_v32_sp_context(
    const AttentionMetadata& base_attn_metadata,
    BatchForwardType batch_forward_type,
    const torch::Tensor& tokens,
    ProcessGroup* sp_group,
    int32_t curr_rank,
    int32_t world_size) {
  if (!::xllm::ParallelConfig::get_instance().enable_prefill_sp() ||
      !batch_forward_type.no_decode() || world_size <= 1) {
    return std::nullopt;
  }

  CHECK(sp_group != nullptr)
      << "deepseek_v32 sequence parallel requires sp_group.";
  CHECK_EQ(tokens.dim(), 1)
      << "deepseek_v32 sequence parallel expects 1D tokens.";
  CHECK_GE(curr_rank, 0) << "curr_rank must be non-negative.";
  CHECK_LT(curr_rank, world_size) << "curr_rank must be less than world_size.";

  const std::vector<int32_t> q_seq_lens =
      extract_q_seq_lens(base_attn_metadata);
  const std::vector<int32_t> ctx_seq_lens =
      extract_ctx_seq_lens(base_attn_metadata);
  CHECK(!q_seq_lens.empty())
      << "deepseek_v32 sequence parallel requires non-empty prefill requests.";
  for (int32_t seq_len : q_seq_lens) {
    if (seq_len < world_size) {
      return std::nullopt;
    }
  }

  const int32_t total_tokens = static_cast<int32_t>(tokens.numel());
  CHECK_EQ(total_tokens,
           std::accumulate(q_seq_lens.begin(), q_seq_lens.end(), int32_t{0}))
      << "tokens.numel() must match total prefill tokens.";

  DeepseekV32SPContext context;
  const auto all_segments =
      build_all_sp_segments(world_size, q_seq_lens, ctx_seq_lens);
  const auto local_segments = build_local_sp_segments(curr_rank, all_segments);
  const auto runtime_artifacts = build_sp_runtime_artifacts(
      curr_rank, world_size, all_segments, total_tokens);
  const torch::Device runtime_device =
      base_attn_metadata.block_table.defined()
          ? base_attn_metadata.block_table.device()
          : base_attn_metadata.q_cu_seq_lens.device();
  context.comm_plan = runtime_artifacts.comm_plan;
  context.local_segments = local_segments;
  context.local_attn_metadata = build_local_prefill_attention_metadata(
      base_attn_metadata, local_segments);
  context.seg_q_starts_cpu =
      build_seq_offsets(extract_q_seq_lens(context.local_attn_metadata));
  context.req_q_offsets_cpu = build_seq_offsets(q_seq_lens);
  context.req_ctx_offsets_cpu = build_seq_offsets(ctx_seq_lens);
  context.seg_q_cu_lens_2col = build_segment_length_matrix(
      local_segments, &DeepseekV32SPSegment::q_tokens, runtime_device);
  context.seg_suffix_k_cu_lens_2col = build_segment_length_matrix(
      local_segments, &DeepseekV32SPSegment::suffix_k_len, runtime_device);
  context.seg_ctx_k_cu_lens_2col = build_segment_length_matrix(
      local_segments, &DeepseekV32SPSegment::ctx_k_len, runtime_device);
  context.seg_ctx_lens_1col =
      build_segment_ctx_lens_tensor(local_segments, runtime_device);
  context.batch_forward_type = batch_forward_type;
  context.total_tokens = total_tokens;
  context.rank = curr_rank;
  context.process_group = sp_group;

  CHECK_EQ(context.seg_q_starts_cpu.size(), context.local_segments.size())
      << "deepseek_v32 sequence parallel expects one q start per segment.";
  CHECK_EQ(context.seg_q_cu_lens_2col.size(0),
           static_cast<int64_t>(context.local_segments.size()))
      << "deepseek_v32 sequence parallel expects one q cu-lens row per "
         "segment.";
  CHECK_EQ(context.seg_suffix_k_cu_lens_2col.size(0),
           static_cast<int64_t>(context.local_segments.size()))
      << "deepseek_v32 sequence parallel expects one suffix-k cu-lens row "
         "per segment.";
  CHECK_EQ(context.seg_ctx_k_cu_lens_2col.size(0),
           static_cast<int64_t>(context.local_segments.size()))
      << "deepseek_v32 sequence parallel expects one ctx-k cu-lens row per "
         "segment.";
  CHECK_EQ(context.seg_ctx_lens_1col.size(0),
           static_cast<int64_t>(context.local_segments.size()))
      << "deepseek_v32 sequence parallel expects one ctx len per segment.";
  CHECK_EQ(context.req_q_offsets_cpu.size(), q_seq_lens.size())
      << "deepseek_v32 sequence parallel expects one request-q offset per "
         "request.";
  CHECK_EQ(context.req_ctx_offsets_cpu.size(), ctx_seq_lens.size())
      << "deepseek_v32 sequence parallel expects one request-ctx offset per "
         "request.";

  const auto int64_options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  context.gathered_reorder_index =
      torch::tensor(runtime_artifacts.gathered_reorder_index_cpu, int64_options)
          .to(tokens.device());
  if (base_attn_metadata.slot_mapping.defined()) {
    context.gathered_slot_mapping = reorder_by_index(
        base_attn_metadata.slot_mapping, context.gathered_reorder_index);
  }
  return context;
}

inline torch::Tensor reorder_by_index(const torch::Tensor& tensor,
                                      const torch::Tensor& reorder_index) {
  if (!tensor.defined()) {
    return tensor;
  }
  CHECK_GT(tensor.dim(), 0)
      << "deepseek_v32 sequence parallel expects non-scalar tensors.";

  torch::Tensor index = reorder_index;
  if (index.scalar_type() != torch::kLong) {
    index = index.to(torch::kLong);
  }
  if (index.device() != tensor.device()) {
    index = index.to(tensor.device());
  }
  return tensor.index_select(0, index);
}

inline torch::Tensor reorder_to_local_shard(
    const torch::Tensor& tensor,
    const DeepseekV32SPContext& context) {
  const int32_t token_num = context.comm_plan.tokens_per_rank.at(context.rank);
  return reorder_by_index(
      tensor,
      context.gathered_reorder_index.narrow(
          0, context.comm_plan.token_num_offset, token_num));
}

inline torch::Tensor all_gather_across_ranks(
    const torch::Tensor& local_tensor,
    const DeepseekV32SPContext& context) {
  if (!local_tensor.defined()) {
    return local_tensor;
  }
  return xllm::parallel_state::gather(
      local_tensor, context.process_group, context.comm_plan.tokens_per_rank);
}

inline PaddedGatherHandle launch_all_gather_across_ranks(
    const torch::Tensor& local_tensor,
    const DeepseekV32SPContext& context) {
  if (!local_tensor.defined()) {
    return {};
  }
  return xllm::parallel_state::launch_gather(
      local_tensor, context.process_group, context.comm_plan.tokens_per_rank);
}

inline torch::Tensor finish_all_gather_across_ranks(PaddedGatherHandle handle) {
  return xllm::parallel_state::finish_gather(std::move(handle));
}

inline torch::Tensor restore_gathered_to_global_order(
    const torch::Tensor& gathered_tensor,
    const DeepseekV32SPContext& context) {
  if (!gathered_tensor.defined()) {
    return gathered_tensor;
  }
  CHECK_GT(gathered_tensor.dim(), 0)
      << "deepseek_v32 sequence parallel expects non-scalar tensors.";

  std::vector<int64_t> output_sizes;
  output_sizes.reserve(gathered_tensor.dim());
  output_sizes.push_back(context.total_tokens);
  for (int64_t dim = 1; dim < gathered_tensor.dim(); ++dim) {
    output_sizes.push_back(gathered_tensor.size(dim));
  }
  torch::Tensor restored =
      torch::zeros(output_sizes, gathered_tensor.options());

  CHECK_EQ(gathered_tensor.size(0), context.gathered_reorder_index.size(0))
      << "unexpected packed tensor length for sequence parallel restore.";
  torch::Tensor restore_index = context.gathered_reorder_index;

  if (restore_index.device() != gathered_tensor.device()) {
    restore_index = restore_index.to(gathered_tensor.device());
  }
  if (restore_index.scalar_type() != torch::kLong) {
    restore_index = restore_index.to(torch::kLong);
  }
  restored.index_copy_(0, restore_index, gathered_tensor);
  return restored;
}

inline torch::Tensor gather_and_restore_global(
    const torch::Tensor& local_tensor,
    const DeepseekV32SPContext& context) {
  return restore_gathered_to_global_order(
      all_gather_across_ranks(local_tensor, context), context);
}

inline torch::Tensor slice_local_packed(const torch::Tensor& packed_tensor,
                                        const DeepseekV32SPContext& context) {
  if (!packed_tensor.defined()) {
    return packed_tensor;
  }
  CHECK_GT(packed_tensor.dim(), 0)
      << "deepseek_v32 sequence parallel expects non-scalar tensors.";
  const int32_t start = context.comm_plan.token_num_offset;
  const int32_t token_num = context.comm_plan.tokens_per_rank.at(context.rank);
  CHECK_LE(start + token_num, packed_tensor.size(0))
      << "packed tensor rows are smaller than local slice range.";
  return packed_tensor.slice(0, start, start + token_num);
}

inline torch::Tensor pad_to_sp_rows(const torch::Tensor& local_tensor,
                                    const DeepseekV32SPContext& context) {
  if (!local_tensor.defined()) {
    return local_tensor;
  }
  const int32_t target_rows =
      context.comm_plan.padded_tokens_per_rank.at(context.rank);
  CHECK_GT(local_tensor.dim(), 0)
      << "deepseek_v32 sequence parallel expects non-scalar tensors.";
  CHECK_LE(local_tensor.size(0), target_rows)
      << "local tensor rows exceed padded target.";
  if (local_tensor.size(0) == target_rows) {
    return local_tensor.contiguous();
  }
  auto pad_shape = local_tensor.sizes().vec();
  pad_shape[0] = target_rows - local_tensor.size(0);
  torch::Tensor pad_tensor = torch::zeros(pad_shape, local_tensor.options());
  return torch::cat({local_tensor.contiguous(), pad_tensor}, /*dim=*/0);
}

}  // namespace xllm::layer::v32_sp
