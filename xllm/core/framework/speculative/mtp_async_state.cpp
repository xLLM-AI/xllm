/* Copyright 2026 The xLLM Authors.

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

#include "core/framework/speculative/mtp_async_state.h"

#include <glog/logging.h>

#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/model/model_type.h"

namespace xllm::mtp_async {
namespace {

torch::Tensor gather_sequence_rows(const torch::Tensor& values,
                                   const torch::Tensor& indices) {
  torch::Tensor gather_index = indices.view({-1, 1});
  for (int64_t dim = 2; dim < values.dim(); ++dim) {
    gather_index = gather_index.unsqueeze(-1);
  }
  std::vector<int64_t> expanded_shape = values.sizes().vec();
  expanded_shape[1] = 1;
  gather_index = gather_index.expand(expanded_shape);
  return values.gather(/*dim=*/1, gather_index).squeeze(/*dim=*/1);
}

}  // namespace

TargetSpecVerifyMode classify_target_spec_verify_mode(
    std::string_view model_type,
    bool is_python_model) {
  if (is_qwen3_5_target_model_type(model_type)) {
    return TargetSpecVerifyMode::EXPANDED_VERIFY;
  }
  if (is_glm5_next_target_model_type(model_type)) {
    return is_python_model ? TargetSpecVerifyMode::EXPANDED_VERIFY
                           : TargetSpecVerifyMode::CAUSAL_CHUNKED_PREFILL;
  }
  if (is_python_model &&
      (model_type == "deepseek_v32" || model_type == "deepseek_v4" ||
       model_type == "deepseek_v4_dspark" ||
       is_deepseek_v41_family(model_type) || model_type == "glm_moe_dsa")) {
    return TargetSpecVerifyMode::EXPANDED_VERIFY;
  }
  if (model_type == "mimo") {
    return TargetSpecVerifyMode::CAUSAL_CHUNKED_PREFILL;
  }
  return TargetSpecVerifyMode::GENERIC;
}

CombinedDraftExecutionPath classify_combined_draft_execution_path(
    std::string_view model_type) {
  if (model_type == "qwen3_5_mtp" || model_type == "qwen3_5_moe_mtp") {
    return CombinedDraftExecutionPath::QWEN3_5_PAGED_ATTENTION;
  }
  return CombinedDraftExecutionPath::UNSUPPORTED;
}

DraftPrefixPolicy classify_draft_prefix_policy(
    TargetSpecVerifyMode target_mode,
    CombinedDraftExecutionPath draft_path) {
  if (target_mode == TargetSpecVerifyMode::EXPANDED_VERIFY &&
      draft_path == CombinedDraftExecutionPath::QWEN3_5_PAGED_ATTENTION) {
    return DraftPrefixPolicy::VERIFIER_CONDITIONED_FIRST_PASS;
  }
  return DraftPrefixPolicy::TAIL_REPAIR;
}

bool supports_combined_draft_configuration(
    CombinedDraftExecutionPath execution_path,
    std::string_view npu_backend,
    int32_t dp_size) {
  switch (execution_path) {
    case CombinedDraftExecutionPath::QWEN3_5_PAGED_ATTENTION:
      return npu_backend == "TORCH" && dp_size <= 1;
    case CombinedDraftExecutionPath::UNSUPPORTED:
      return false;
  }
  return false;
}

torch::Tensor extract_target_base_kv_seq_lens(
    const torch::Tensor& validate_kv_seq_lens,
    int64_t batch_size,
    int64_t num_validate_tokens,
    bool use_chunked_prefill) {
  CHECK(validate_kv_seq_lens.defined());
  CHECK_GT(batch_size, 0);
  CHECK_GT(num_validate_tokens, 0);
  torch::Tensor flattened = validate_kv_seq_lens.flatten();
  if (use_chunked_prefill) {
    CHECK_GE(flattened.numel(), batch_size);
    return flattened.slice(/*dim=*/0, /*start=*/0, /*end=*/batch_size) -
           (num_validate_tokens - 1);
  }

  const int64_t expanded_rows = batch_size * num_validate_tokens;
  CHECK_GE(flattened.numel(), expanded_rows);
  torch::Tensor rows =
      flattened.slice(/*dim=*/0, /*start=*/0, /*end=*/expanded_rows)
          .view({batch_size, num_validate_tokens});
  return rows.select(/*dim=*/1, /*index=*/0).contiguous();
}

AcceptedState build_accepted_state(const torch::Tensor& accepted_tokens,
                                   const torch::Tensor& accepted_embeddings,
                                   const torch::Tensor& embedding_placeholder,
                                   const torch::Tensor& base_positions,
                                   const torch::Tensor& base_kv_seq_lens) {
  CHECK_EQ(accepted_tokens.dim(), 2);
  const int64_t batch_size = accepted_tokens.size(0);
  CHECK_GT(accepted_tokens.size(/*dim=*/1), 0);
  CHECK_GE(base_positions.numel(), batch_size);
  CHECK_GE(base_kv_seq_lens.numel(), batch_size);
  CHECK_EQ(accepted_embeddings.dim(), 3);
  CHECK_EQ(accepted_embeddings.size(/*dim=*/0), batch_size);

  AcceptedState state;
  const torch::Tensor accepted_lengths =
      accepted_tokens.ge(/*other=*/0).sum(/*dim=*/1);
  torch::Tensor last_indices = (accepted_lengths - 1).clamp_min(/*min=*/0);
  state.last_tokens = gather_sequence_rows(accepted_tokens, last_indices);
  state.base_positions = base_positions.flatten()
                             .slice(/*dim=*/0, /*start=*/0, batch_size)
                             .to(torch::kLong) +
                         accepted_lengths;
  state.base_kv_seq_lens = base_kv_seq_lens.flatten()
                               .slice(/*dim=*/0, /*start=*/0, batch_size)
                               .to(torch::kLong) +
                           accepted_lengths;
  state.all_draft_accepted =
      accepted_lengths.eq(accepted_tokens.size(/*dim=*/1));

  torch::Tensor previous_indices = (accepted_lengths - 2).clamp_min(/*min=*/0);
  torch::Tensor gathered_previous_tokens =
      gather_sequence_rows(accepted_tokens, previous_indices);
  torch::Tensor has_previous = accepted_lengths.gt(/*other=*/1);
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

  return state;
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

}  // namespace xllm::mtp_async
