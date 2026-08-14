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

#include "core/framework/speculative/mtp_async_input_builder.h"

#include <glog/logging.h>

#if defined(USE_NPU)
#include "kernels/npu/xllm_ops/xllm_ops_api.h"
#endif

#include "core/framework/speculative/mtp_async_state.h"
#include "core/runtime/forward_params.h"
#include "layers/common/expanded_decode_metadata_builder.h"

namespace xllm::mtp_async {
namespace {

torch::Tensor build_device_cache_slots(const ForwardInput& input,
                                       const torch::Tensor& positions,
                                       int32_t block_size) {
  CHECK_EQ(positions.dim(), 2);
  if (!input.input_params.multi_block_tables.empty()) {
    return torch::zeros_like(positions, positions.options().dtype(torch::kInt))
        .flatten();
  }
  return map_positions_to_cache_slots(
      input.input_params.attention.device.block_tables, positions, block_size);
}

void apply_device_row_metadata(ForwardInput& input,
                               const ForwardInput& block_table_source,
                               const AcceptedState& state,
                               const torch::Tensor& offsets,
                               int32_t block_size,
                               bool use_chunked_prefill) {
  torch::Tensor row_positions = make_row_positions(state, offsets);
  input.positions = row_positions.flatten().to(input.positions.options());
  input.input_params.attention.device.new_cache_slots =
      build_device_cache_slots(block_table_source, row_positions, block_size);
  torch::Tensor kv_seq_lens =
      make_kv_seq_lens(state, offsets, use_chunked_prefill);
  input.input_params.attention.device.kv_seq_lens =
      kv_seq_lens.to(input.input_params.attention.device.kv_seq_lens.options());
}

#if defined(USE_NPU)
void expand_decode_attention_metadata(ForwardInput& draft_input,
                                      const ForwardInput& block_table_source,
                                      const torch::Tensor& kv_seq_lens,
                                      int32_t block_size) {
  layer::ExpandedDecodeMetadataBuilder::populate(
      draft_input.input_params,
      block_table_source.input_params,
      kv_seq_lens,
      block_size);
}

void apply_mtp_prepare_output(
    ForwardInput& draft_input,
    const ForwardInput& block_table_source,
    const kernel::npu::MtpPrepareNextDraftOutput& output,
    bool use_chunked_prefill,
    bool rebuild_expanded_decode_metadata,
    int32_t block_size) {
  CHECK_EQ(output.token_ids.dim(), 1);
  CHECK_EQ(output.positions.dim(), 1);
  CHECK_EQ(output.cache_slots.dim(), 1);
  CHECK_EQ(output.embeddings.dim(), 2);
  CHECK_EQ(output.kv_seq_lens.dim(), 1);
  const int64_t token_count = output.token_ids.numel();
  CHECK_EQ(output.positions.numel(), token_count);
  CHECK_EQ(output.cache_slots.numel(), token_count);
  CHECK_EQ(output.embeddings.size(0), token_count);

  draft_input.token_ids = output.token_ids;
  draft_input.input_params.embedding.input_embedding = output.embeddings;
  draft_input.positions = output.positions;
  if (use_chunked_prefill) {
    draft_input.input_params.attention.device.kv_seq_lens = output.kv_seq_lens;
  } else {
    draft_input.input_params.attention.device.kv_seq_lens =
        torch::stack({output.kv_seq_lens - 1, output.kv_seq_lens}, /*dim=*/1)
            .flatten();
  }
  draft_input.input_params.attention.device.new_cache_slots =
      output.cache_slots;
  if (!use_chunked_prefill && rebuild_expanded_decode_metadata) {
    expand_decode_attention_metadata(
        draft_input, block_table_source, output.kv_seq_lens, block_size);
    const auto& attention = draft_input.input_params.attention.device;
    CHECK_EQ(attention.kv_seq_lens.numel(), token_count);
    CHECK_EQ(attention.new_cache_slots.numel(), token_count);
    CHECK_EQ(attention.block_tables.size(0), token_count);
    CHECK_EQ(attention.paged_kv_indptr.numel(), token_count + 1);
    CHECK_EQ(attention.paged_kv_last_page_len.numel(), token_count);
    CHECK_EQ(draft_input.input_params.graph.expanded_kv_seq_lens.numel(),
             token_count);
    CHECK_EQ(draft_input.input_params.graph.expanded_block_tables.size(0),
             token_count);
  }
}
#endif

}  // namespace

void prepare_next_draft_from_accepted_state(
    ForwardInput& draft_input,
    const ForwardInput& block_table_source,
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& accepted_embeddings,
    const torch::Tensor& embedding_placeholder,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens,
    bool use_chunked_prefill,
    bool rebuild_expanded_decode_metadata,
    int32_t block_size) {
#if defined(USE_NPU)
  if (block_table_source.input_params.multi_block_tables.empty()) {
    const auto output = kernel::npu::try_mtp_prepare_next_draft(
        accepted_tokens,
        accepted_embeddings,
        embedding_placeholder,
        base_positions,
        base_kv_seq_lens,
        block_table_source.input_params.attention.device.block_tables,
        block_size);
    if (output.has_value()) {
      apply_mtp_prepare_output(draft_input,
                               block_table_source,
                               *output,
                               use_chunked_prefill,
                               rebuild_expanded_decode_metadata,
                               block_size);
      return;
    }
  }
#endif

  AcceptedState state = build_accepted_state(accepted_tokens,
                                             accepted_embeddings,
                                             embedding_placeholder,
                                             base_positions,
                                             base_kv_seq_lens);
  // Generate offsets on device to avoid a synchronizing host-to-device copy.
  torch::Tensor extend_offsets = torch::arange(
      /*start=*/-1,
      /*end=*/1,
      torch::TensorOptions()
          .dtype(torch::kLong)
          .device(accepted_tokens.device()));
  apply_device_row_metadata(draft_input,
                            block_table_source,
                            state,
                            extend_offsets,
                            block_size,
                            use_chunked_prefill);

  // On rejection, redirect the shape-stabilizing repair row to a future
  // scratch position so it cannot overwrite valid draft KV state.
  torch::Tensor previous_cache_positions = make_repair_cache_positions(state);
  torch::Tensor cache_positions =
      torch::stack({previous_cache_positions, state.base_positions},
                   /*dim=*/1);
  draft_input.input_params.attention.device.new_cache_slots =
      build_device_cache_slots(block_table_source, cache_positions, block_size);
  draft_input.token_ids =
      torch::stack({state.previous_tokens, state.last_tokens}, /*dim=*/1)
          .flatten()
          .to(draft_input.token_ids.options());
  draft_input.input_params.embedding.input_embedding =
      torch::stack({state.previous_embeddings, state.last_embeddings},
                   /*dim=*/1)
          .flatten(/*start_dim=*/0, /*end_dim=*/1);
#if defined(USE_NPU)
  if (!use_chunked_prefill && rebuild_expanded_decode_metadata) {
    expand_decode_attention_metadata(
        draft_input,
        block_table_source,
        state.base_kv_seq_lens.to(base_kv_seq_lens.options()),
        block_size);
  }
#endif
}

void prepare_later_draft_from_device_base(
    ForwardInput& draft_input,
    const ForwardInput& block_table_source,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens,
    int32_t position_offset,
    int32_t block_size) {
  CHECK(base_positions.defined());
  CHECK(base_kv_seq_lens.defined());
  CHECK_EQ(base_positions.dim(), 1);
  CHECK_EQ(base_kv_seq_lens.dim(), 1);
  CHECK_EQ(base_positions.numel(), base_kv_seq_lens.numel());
  CHECK_GT(position_offset, 0);

  torch::Tensor row_positions =
      (base_positions + position_offset).unsqueeze(/*dim=*/1);
  draft_input.positions =
      row_positions.flatten().to(draft_input.positions.options());
  draft_input.input_params.attention.device.new_cache_slots =
      build_device_cache_slots(block_table_source, row_positions, block_size);
  draft_input.input_params.attention.device.kv_seq_lens =
      (base_kv_seq_lens + position_offset)
          .to(draft_input.input_params.attention.device.kv_seq_lens.options());
}

void prepare_target_verify_from_accepted_state(
    ForwardInput& validate_input,
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens,
    int32_t block_size) {
  CHECK(validate_input.token_ids.defined());
  CHECK(validate_input.positions.defined());
  CHECK_EQ(accepted_tokens.dim(), 2);
  const int64_t batch_size = accepted_tokens.size(0);
  const int64_t validate_width = accepted_tokens.size(1);
  CHECK_EQ(validate_input.token_ids.numel(), batch_size * validate_width);
  CHECK_EQ(validate_input.positions.numel(), batch_size * validate_width);

  AcceptedTokenMetadata metadata = build_accepted_token_metadata(
      accepted_tokens, base_positions, base_kv_seq_lens);
  torch::Tensor template_position_rows =
      validate_input.positions.view({batch_size, validate_width});
  torch::Tensor position_delta =
      metadata.base_positions -
      template_position_rows.select(/*dim=*/1, /*index=*/0).to(torch::kLong);
  torch::Tensor position_rows =
      template_position_rows.to(torch::kLong) + position_delta.unsqueeze(1);
  validate_input.positions =
      position_rows.flatten().to(validate_input.positions.options());
  if (validate_input.input_params.multi_block_tables.empty()) {
    const torch::Tensor& expanded_block_tables =
        validate_input.input_params.attention.device.block_tables;
    CHECK(expanded_block_tables.defined());
    CHECK_EQ(expanded_block_tables.dim(), 2);
    CHECK_EQ(expanded_block_tables.size(0), batch_size * validate_width);
    torch::Tensor sequence_block_tables =
        expanded_block_tables
            .view({batch_size, validate_width, expanded_block_tables.size(1)})
            .select(/*dim=*/1, /*index=*/0);
    validate_input.input_params.attention.device.new_cache_slots =
        map_positions_to_cache_slots(
            sequence_block_tables, position_rows, block_size);
  } else {
    validate_input.input_params.attention.device.new_cache_slots =
        torch::zeros_like(position_rows,
                          position_rows.options().dtype(torch::kInt))
            .flatten();
  }

  torch::Tensor template_kv_rows =
      validate_input.input_params.attention.device.kv_seq_lens.view(
          {batch_size, validate_width});
  torch::Tensor kv_delta =
      metadata.base_kv_seq_lens -
      template_kv_rows.select(/*dim=*/1, /*index=*/0).to(torch::kLong);
  validate_input.input_params.attention.device.kv_seq_lens =
      (template_kv_rows.to(torch::kLong) + kv_delta.unsqueeze(1))
          .flatten()
          .to(validate_input.input_params.attention.device.kv_seq_lens
                  .options());

  torch::Tensor token_rows =
      validate_input.token_ids.view({batch_size, validate_width});
  token_rows.select(/*dim=*/1, /*index=*/0)
      .copy_(metadata.last_tokens.to(validate_input.token_ids.options()),
             /*non_blocking=*/true);
  validate_input.device_tensors_ready = true;
}

}  // namespace xllm::mtp_async
