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

#include "core/framework/speculative/mtp_async_input_builder.h"

#include <glog/logging.h>

#include "core/util/tensor_helper.h"

#if defined(USE_NPU)
#include "kernels/npu/tilelang/mtp_prepare_next_draft.h"
#endif

#include "core/framework/speculative/mtp_async_state.h"
#include "core/runtime/vlm_forward_params.h"
#include "layers/common/expanded_decode_metadata_builder.h"

namespace xllm::mtp_async {
namespace {

template <typename Input>
torch::Tensor build_device_cache_slots(const Input& input,
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
#if defined(USE_NPU)
template <typename Source>
void expand_decode_attention_metadata(LlmForwardInput& draft_input,
                                      const Source& block_table_source,
                                      const torch::Tensor& kv_seq_lens,
                                      int32_t block_size) {
  layer::ExpandedDecodeMetadataBuilder::populate(
      ModelParamsView(draft_input.input_params),
      block_table_source.input_params,
      kv_seq_lens,
      block_size);
}

template <typename Source>
void apply_mtp_prepare_output(
    LlmForwardInput& draft_input,
    const Source& block_table_source,
    const kernel::npu::MtpPrepareNextDraftOutput& output,
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
  const bool expanded_lengths = output.kv_seq_lens.numel() == token_count;
  if (expanded_lengths) {
    draft_input.input_params.attention.device.kv_seq_lens = output.kv_seq_lens;
  } else {
    draft_input.input_params.attention.device.kv_seq_lens =
        torch::stack({output.kv_seq_lens - 1, output.kv_seq_lens}, /*dim=*/1)
            .flatten();
  }
  draft_input.input_params.attention.device.new_cache_slots =
      output.cache_slots;
  if (rebuild_expanded_decode_metadata) {
    const torch::Tensor base_lengths =
        expanded_lengths
            ? output.kv_seq_lens.view({-1, 2}).select(1, 1).contiguous()
            : output.kv_seq_lens;
    expand_decode_attention_metadata(
        draft_input, block_table_source, base_lengths, block_size);
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

void prepare_draft_prefix_from_target(LlmForwardInput& draft_input,
                                      const torch::Tensor& accepted_tokens,
                                      const torch::Tensor& target_hidden,
                                      const torch::Tensor& base_positions,
                                      const torch::Tensor& base_kv_seq_lens,
                                      int32_t block_size) {
  CHECK_EQ(accepted_tokens.dim(), 2);
  CHECK_EQ(target_hidden.dim(), 3);
  const int64_t batch_size = accepted_tokens.size(0);
  const int64_t width = accepted_tokens.size(1);
  CHECK_GT(batch_size, 0);
  CHECK_GE(width, 2);
  CHECK_LE(width, 17);
  CHECK_EQ(target_hidden.size(0), batch_size);
  CHECK_EQ(target_hidden.size(1), width);
  CHECK_EQ(base_positions.numel(), batch_size);
  CHECK_EQ(base_kv_seq_lens.numel(), batch_size);
  CHECK_EQ(draft_input.positions.numel(), batch_size * width);
  CHECK_EQ(draft_input.token_ids.numel(), batch_size * width);
  auto& attention = draft_input.input_params.attention.device;
  CHECK(attention.block_tables.defined());
  CHECK_EQ(attention.block_tables.size(0), batch_size * width);
  CHECK_EQ(attention.kv_seq_lens.numel(), batch_size * width);
  CHECK(accepted_tokens.device() == target_hidden.device());
  CHECK(accepted_tokens.device() == base_positions.device());
  CHECK(accepted_tokens.device() == base_kv_seq_lens.device());
  CHECK(draft_input.sampling_params.logits_indices().defined());
  CHECK_EQ(draft_input.sampling_params.logits_indices().numel(), batch_size);

  const torch::Tensor offsets =
      torch::arange(width, accepted_tokens.options().dtype(torch::kLong));
  const torch::Tensor positions =
      base_positions.to(torch::kLong).view({batch_size, 1}) + offsets;
  const torch::Tensor kv_lengths =
      base_kv_seq_lens.to(torch::kLong).view({batch_size, 1}) + offsets;
  const torch::Tensor sequence_tables =
      attention.block_tables
          .view({batch_size, width, attention.block_tables.size(1)})
          .select(1, 0);
  attention.new_cache_slots =
      map_positions_to_cache_slots(sequence_tables, kv_lengths - 1, block_size);
  attention.kv_seq_lens =
      kv_lengths.flatten().to(attention.kv_seq_lens.options());
  draft_input.positions =
      positions.flatten().to(draft_input.positions.options());
  draft_input.token_ids = accepted_tokens.clamp_min(0).flatten().to(
      draft_input.token_ids.options());
  draft_input.input_params.embedding.input_embedding =
      target_hidden.flatten(/*start_dim=*/0, /*end_dim=*/1);
  const torch::Tensor lengths =
      accepted_tokens.ge(0).sum(/*dim=*/1, /*keepdim=*/false, torch::kLong);
  draft_input.sampling_params.logits_indices() =
      (torch::arange(batch_size, lengths.options()) * width + lengths - 1)
          .to(draft_input.sampling_params.logits_indices().options().device(
              accepted_tokens.device()));
  draft_input.runtime.device_tensors_ready = true;
}

template <typename Source>
void prepare_next_draft_from_accepted_state(
    LlmForwardInput& draft_input,
    const Source& block_table_source,
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& accepted_embeddings,
    const torch::Tensor& embedding_placeholder,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens,
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
  const torch::Tensor positions = make_positions(state, extend_offsets);
  draft_input.positions =
      positions.flatten().to(draft_input.positions.options());
  draft_input.input_params.attention.device.kv_seq_lens =
      make_kv_seq_lens(state, extend_offsets)
          .to(draft_input.input_params.attention.device.kv_seq_lens.options());

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
  if (rebuild_expanded_decode_metadata) {
    expand_decode_attention_metadata(
        draft_input,
        block_table_source,
        state.base_kv_seq_lens.to(base_kv_seq_lens.options()),
        block_size);
  }
#endif
}

template void prepare_next_draft_from_accepted_state(LlmForwardInput&,
                                                     const LlmForwardInput&,
                                                     const torch::Tensor&,
                                                     const torch::Tensor&,
                                                     const torch::Tensor&,
                                                     const torch::Tensor&,
                                                     const torch::Tensor&,
                                                     bool,
                                                     int32_t);
template void prepare_next_draft_from_accepted_state(LlmForwardInput&,
                                                     const VlmForwardInput&,
                                                     const torch::Tensor&,
                                                     const torch::Tensor&,
                                                     const torch::Tensor&,
                                                     const torch::Tensor&,
                                                     const torch::Tensor&,
                                                     bool,
                                                     int32_t);

}  // namespace xllm::mtp_async
