/* Copyright 2026 The xLLM Authors.

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

#include "core/runtime/mtp_async_state.h"

#include <glog/logging.h>

#include <vector>

namespace xllm::mtp_async {
namespace {

torch::Tensor gather_sequence_rows(const torch::Tensor& values,
                                   const torch::Tensor& indices) {
  CHECK_GE(values.dim(), 2);
  CHECK_EQ(values.size(0), indices.numel());
  torch::Tensor gather_index =
      indices.to(torch::dtype(torch::kLong).device(indices.device()))
          .view({-1, 1});
  for (int64_t dim = 2; dim < values.dim(); ++dim) {
    gather_index = gather_index.unsqueeze(-1);
  }
  std::vector<int64_t> expanded_shape = values.sizes().vec();
  expanded_shape[1] = 1;
  gather_index = gather_index.expand(expanded_shape);
  return values.gather(/*dim=*/1, gather_index).squeeze(/*dim=*/1);
}

}  // namespace

CombinedDraftExecutionPath classify_combined_draft_execution_path(
    std::string_view model_type) {
  if (model_type == "qwen3_5_mtp" || model_type == "qwen3_5_moe_mtp") {
    return CombinedDraftExecutionPath::QWEN3_5_PAGED_ATTENTION;
  }
  if (model_type == "glm_moe_dsa_mtp") {
    return CombinedDraftExecutionPath::GLM5_DSA;
  }
  return CombinedDraftExecutionPath::UNSUPPORTED;
}

bool can_use_combined_decode(bool enable_schedule_overlap,
                             bool execution_path_supported) {
  return enable_schedule_overlap && execution_path_supported;
}

AcceptedState build_accepted_state(const torch::Tensor& accepted_tokens,
                                   const torch::Tensor& accepted_embeddings,
                                   const torch::Tensor& embedding_placeholder,
                                   const torch::Tensor& base_positions,
                                   const torch::Tensor& base_kv_seq_lens) {
  CHECK_EQ(accepted_tokens.dim(), 2);
  CHECK_EQ(accepted_embeddings.dim(), 3);
  const int64_t batch_size = accepted_tokens.size(0);
  CHECK_EQ(accepted_embeddings.size(0), batch_size);
  CHECK_GE(base_positions.numel(), batch_size);
  CHECK_GE(base_kv_seq_lens.numel(), batch_size);

  AcceptedState state;
  state.accepted_lengths =
      accepted_tokens.ge(0).sum(/*dim=*/1).to(torch::kLong);
  state.all_draft_accepted =
      state.accepted_lengths.eq(accepted_tokens.size(/*dim=*/1));
  torch::Tensor last_indices = (state.accepted_lengths - 1).clamp_min(0);
  torch::Tensor previous_indices = (state.accepted_lengths - 2).clamp_min(0);
  state.last_tokens = gather_sequence_rows(accepted_tokens, last_indices);
  torch::Tensor gathered_previous_tokens =
      gather_sequence_rows(accepted_tokens, previous_indices);
  torch::Tensor has_previous = state.accepted_lengths.gt(1);
  state.previous_tokens =
      torch::where(has_previous, gathered_previous_tokens, state.last_tokens);
  state.last_embeddings =
      gather_sequence_rows(accepted_embeddings, last_indices);
  torch::Tensor gathered_previous_embeddings =
      gather_sequence_rows(accepted_embeddings, previous_indices);
  torch::Tensor placeholder = embedding_placeholder;
  if (placeholder.dim() == 1) {
    placeholder = placeholder.unsqueeze(0);
  }
  placeholder = placeholder.expand_as(gathered_previous_embeddings);
  state.previous_embeddings = torch::where(has_previous.view({batch_size, 1}),
                                           gathered_previous_embeddings,
                                           placeholder);

  state.base_positions =
      base_positions.flatten().slice(0, 0, batch_size).to(torch::kLong) +
      state.accepted_lengths;
  state.base_kv_seq_lens =
      base_kv_seq_lens.flatten().slice(0, 0, batch_size).to(torch::kLong) +
      state.accepted_lengths;
  return state;
}

torch::Tensor make_row_positions(const AcceptedState& state,
                                 const torch::Tensor& offsets) {
  return state.base_positions.unsqueeze(1) +
         offsets.to(state.base_positions.options()).unsqueeze(0);
}

torch::Tensor make_kv_seq_lens(const AcceptedState& state,
                               const torch::Tensor& offsets,
                               bool use_chunked_prefill) {
  if (use_chunked_prefill) {
    return state.base_kv_seq_lens;
  }
  return (state.base_kv_seq_lens.unsqueeze(1) +
          offsets.to(state.base_kv_seq_lens.options()).unsqueeze(0))
      .flatten();
}

torch::Tensor make_repair_cache_positions(const AcceptedState& state) {
  return torch::where(state.all_draft_accepted,
                      state.base_positions - 1,
                      state.base_positions + 1);
}

torch::Tensor map_positions_to_cache_slots(const torch::Tensor& block_tables,
                                           const torch::Tensor& positions,
                                           int32_t block_size) {
  CHECK_EQ(positions.dim(), 2);
  CHECK(block_tables.defined());
  CHECK_GT(block_size, 0);
  const int64_t batch_size = positions.size(0);
  torch::Tensor position_long =
      positions.to(torch::dtype(torch::kLong).device(positions.device()));
  torch::Tensor block_indices =
      torch::floor_divide(position_long, block_size)
          .to(torch::dtype(torch::kLong).device(position_long.device()));
  torch::Tensor block_ids = block_tables.slice(/*dim=*/0, 0, batch_size)
                                .to(torch::kLong)
                                .gather(/*dim=*/1, block_indices);
  return (block_ids * block_size + position_long.remainder(block_size))
      .to(torch::kInt)
      .flatten();
}

torch::Tensor select_rows(const torch::Tensor& values,
                          const torch::Tensor& selected_rows) {
  if (!values.defined() || !selected_rows.defined() ||
      selected_rows.numel() == 0) {
    return values;
  }
  CHECK_GE(values.dim(), 1);
  if (values.size(0) == selected_rows.numel()) {
    return values;
  }
  if (selected_rows.device().is_cpu()) {
    CHECK_GT(values.size(0), selected_rows.max().item<int64_t>())
        << "MTP selected row index exceeds available rows.";
  }
  torch::Tensor index =
      selected_rows.to(torch::dtype(torch::kLong).device(values.device()))
          .contiguous();
  return values.index_select(/*dim=*/0, index);
}

}  // namespace xllm::mtp_async
