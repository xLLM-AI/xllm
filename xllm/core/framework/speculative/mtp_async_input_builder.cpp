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

#include <algorithm>

#include "core/util/tensor_helper.h"

#if defined(USE_NPU)
#include "kernels/npu/tilelang/mtp_prepare_next_draft.h"
#endif

#include "core/framework/speculative/mtp_async_state.h"
#include "core/runtime/vlm_forward_params.h"
#include "layers/common/expanded_decode_metadata_builder.h"

namespace xllm::mtp_async {
namespace {
void shift_host_rows(std::vector<int32_t>& values,
                     const torch::Tensor& base_values,
                     int64_t batch_size,
                     int64_t row_width,
                     bool step_major_layout) {
  if (values.empty()) {
    return;
  }
  CHECK_EQ(base_values.numel(), batch_size);
  CHECK(values.size() == static_cast<size_t>(batch_size) ||
        values.size() == static_cast<size_t>(batch_size * row_width));
  // tensor_to_vector() owns the device-to-host copy and reads only after the
  // copy has completed. Calling to(torch::kCPU) here first can expose a
  // pending asynchronous NPU copy as stale host memory.
  const std::vector<int64_t> base = tensor_to_vector<int64_t>(base_values);
  if (values.size() == static_cast<size_t>(batch_size)) {
    for (int64_t seq_id = 0; seq_id < batch_size; ++seq_id) {
      values[static_cast<size_t>(seq_id)] =
          static_cast<int32_t>(base[static_cast<size_t>(seq_id)]);
    }
    return;
  }
  for (int64_t seq_id = 0; seq_id < batch_size; ++seq_id) {
    const int64_t first_index = step_major_layout ? seq_id : seq_id * row_width;
    const int32_t delta =
        static_cast<int32_t>(base[static_cast<size_t>(seq_id)]) -
        values[static_cast<size_t>(first_index)];
    for (int64_t row = 0; row < row_width; ++row) {
      const int64_t index = step_major_layout ? row * batch_size + seq_id
                                              : seq_id * row_width + row;
      values[static_cast<size_t>(index)] += delta;
    }
  }
}

void shift_host_tensor(torch::Tensor& tensor,
                       const torch::Tensor& base_values,
                       int64_t batch_size,
                       int64_t row_width,
                       bool step_major_layout) {
  if (!tensor.defined() || tensor.numel() == 0) {
    return;
  }
  CHECK(tensor.device().is_cpu());
  CHECK_EQ(tensor.scalar_type(), torch::kInt);
  std::vector<int32_t> values = tensor_to_vector<int32_t>(tensor);
  shift_host_rows(
      values, base_values, batch_size, row_width, step_major_layout);
  std::copy(values.begin(), values.end(), tensor.data_ptr<int32_t>());
}

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
template <typename Source>
void apply_device_row_metadata(LlmForwardInput& input,
                               const Source& block_table_source,
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
  const bool expanded_lengths = output.kv_seq_lens.numel() == token_count;
  if (expanded_lengths && use_chunked_prefill) {
    draft_input.input_params.attention.device.kv_seq_lens =
        output.kv_seq_lens.view({-1, 2}).select(1, 1).contiguous();
  } else if (use_chunked_prefill || expanded_lengths) {
    draft_input.input_params.attention.device.kv_seq_lens = output.kv_seq_lens;
  } else {
    draft_input.input_params.attention.device.kv_seq_lens =
        torch::stack({output.kv_seq_lens - 1, output.kv_seq_lens}, /*dim=*/1)
            .flatten();
  }
  draft_input.input_params.attention.device.new_cache_slots =
      output.cache_slots;
  if (!use_chunked_prefill && rebuild_expanded_decode_metadata) {
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
  CHECK(draft_input.sampling_params.selected_token_idxes.defined());
  CHECK_EQ(draft_input.sampling_params.selected_token_idxes.numel(),
           batch_size);

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
  draft_input.sampling_params.selected_token_idxes =
      (torch::arange(batch_size, lengths.options()) * width + lengths - 1)
          .to(draft_input.sampling_params.selected_token_idxes.options().device(
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
    bool use_chunked_prefill,
    bool rebuild_expanded_decode_metadata,
    int32_t block_size,
    bool require_fused_npu_kernel,
    kernel::npu::MtpPrepareNextDraftWorkspace* reusable_workspace) {
#if defined(USE_NPU)
  if (require_fused_npu_kernel &&
      !block_table_source.input_params.multi_block_tables.empty()) {
    LOG(FATAL) << "unified MTP requires a single paged block table for fused "
                  "next-draft preparation";
  }
  if (block_table_source.input_params.multi_block_tables.empty()) {
    const auto output = kernel::npu::try_mtp_prepare_next_draft(
        accepted_tokens,
        accepted_embeddings,
        embedding_placeholder,
        base_positions,
        base_kv_seq_lens,
        block_table_source.input_params.attention.device.block_tables,
        block_size,
        reusable_workspace);
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
  if (require_fused_npu_kernel) {
    LOG(FATAL)
        << "unified MTP requires the fused NPU next-draft prepare kernel";
  }

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
template <typename Source>
void prepare_later_draft_from_device_base(LlmForwardInput& draft_input,
                                          const Source& block_table_source,
                                          const torch::Tensor& base_positions,
                                          const torch::Tensor& base_kv_seq_lens,
                                          int32_t position_offset,
                                          int32_t block_size) {
  CHECK(base_positions.defined());
  CHECK(base_kv_seq_lens.defined());
  CHECK_EQ(base_positions.dim(), 1);
  CHECK_EQ(base_kv_seq_lens.dim(), 1);
  CHECK_EQ(base_positions.numel(), base_kv_seq_lens.numel());
  CHECK_GE(position_offset, 0);

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
template <typename Input>
void prepare_target_verify_from_accepted_state(
    Input& validate_input,
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens,
    int32_t block_size,
    bool use_chunked_prefill,
    bool step_major_layout) {
  CHECK(validate_input.token_ids.defined());
  CHECK(validate_input.positions.defined());
  CHECK_EQ(accepted_tokens.dim(), 2);
  const int64_t batch_size = accepted_tokens.size(0);
  const int64_t validate_width = accepted_tokens.size(1);
  CHECK_EQ(validate_input.token_ids.numel(), batch_size * validate_width);
  CHECK_EQ(validate_input.positions.numel(), batch_size * validate_width);

  AcceptedTokenMetadata metadata = build_accepted_token_metadata(
      accepted_tokens, base_positions, base_kv_seq_lens);
  shift_host_tensor(validate_input.positions_host,
                    metadata.base_positions,
                    batch_size,
                    validate_width,
                    step_major_layout);
  shift_host_rows(validate_input.input_params.attention.host.kv_seq_lens,
                  use_chunked_prefill
                      ? metadata.base_kv_seq_lens + (validate_width - 1)
                      : metadata.base_kv_seq_lens,
                  batch_size,
                  validate_width,
                  step_major_layout);
  if (validate_input.input_params.graph.expanded_kv_seq_lens.defined()) {
    shift_host_rows(validate_input.input_params.graph.expanded_kv_seq_lens_vec,
                    metadata.base_kv_seq_lens,
                    batch_size,
                    validate_width,
                    step_major_layout);
  }
  if (!validate_input.input_params.attention.host.kv_seq_lens.empty()) {
    validate_input.input_params.meta.kv_max_seq_len = *std::max_element(
        validate_input.input_params.attention.host.kv_seq_lens.begin(),
        validate_input.input_params.attention.host.kv_seq_lens.end());
  }
  torch::Tensor template_position_rows;
  if (step_major_layout) {
    template_position_rows =
        validate_input.positions.view({validate_width, batch_size})
            .transpose(0, 1);
  } else {
    template_position_rows =
        validate_input.positions.view({batch_size, validate_width});
  }
  torch::Tensor position_delta =
      metadata.base_positions -
      template_position_rows.select(/*dim=*/1, /*index=*/0).to(torch::kLong);
  torch::Tensor position_rows =
      template_position_rows.to(torch::kLong) + position_delta.unsqueeze(1);
  validate_input.positions =
      (step_major_layout ? position_rows.transpose(0, 1) : position_rows)
          .flatten()
          .to(validate_input.positions.options());
  if (validate_input.input_params.multi_block_tables.empty()) {
    const auto& graph = validate_input.input_params.graph;
    const torch::Tensor& expanded_block_tables =
        graph.expanded_block_tables.defined()
            ? graph.expanded_block_tables
            : validate_input.input_params.attention.device.block_tables;
    CHECK(expanded_block_tables.defined());
    CHECK_EQ(expanded_block_tables.dim(), 2);
    torch::Tensor sequence_block_tables;
    if (expanded_block_tables.size(0) == batch_size) {
      sequence_block_tables = expanded_block_tables;
    } else if (expanded_block_tables.size(0) == batch_size * validate_width &&
               step_major_layout) {
      sequence_block_tables =
          expanded_block_tables
              .view({validate_width, batch_size, expanded_block_tables.size(1)})
              .select(/*dim=*/0, /*index=*/0);
    } else if (expanded_block_tables.size(0) == batch_size * validate_width) {
      sequence_block_tables =
          expanded_block_tables
              .view({batch_size, validate_width, expanded_block_tables.size(1)})
              .select(/*dim=*/1, /*index=*/0);
    } else {
      LOG(FATAL) << "target verify block tables have "
                 << expanded_block_tables.size(0) << " rows; expected "
                 << batch_size << " or " << batch_size * validate_width;
    }
    torch::Tensor slot_rows =
        map_positions_to_cache_slots(
            sequence_block_tables, position_rows, block_size)
            .view({batch_size, validate_width});
    validate_input.input_params.attention.device.new_cache_slots =
        (step_major_layout ? slot_rows.transpose(0, 1) : slot_rows).flatten();
  } else {
    validate_input.input_params.attention.device.new_cache_slots =
        torch::zeros_like(position_rows,
                          position_rows.options().dtype(torch::kInt))
            .flatten();
  }

  torch::Tensor template_kv_rows;
  const torch::Tensor kv_seq_lens =
      validate_input.input_params.attention.device.kv_seq_lens.flatten();
  if (kv_seq_lens.numel() == batch_size) {
    torch::Tensor kv_delta =
        metadata.base_kv_seq_lens.flatten().slice(0, 0, batch_size) -
        kv_seq_lens.to(torch::kLong);
    if (use_chunked_prefill) {
      kv_delta = kv_delta + (validate_width - 1);
    }
    validate_input.input_params.attention.device.kv_seq_lens =
        (kv_seq_lens.to(torch::kLong) + kv_delta)
            .to(validate_input.input_params.attention.device.kv_seq_lens
                    .options());
  } else {
    CHECK_EQ(kv_seq_lens.numel(), batch_size * validate_width)
        << "target verify KV lengths have " << kv_seq_lens.numel()
        << " values; expected " << batch_size << " or "
        << batch_size * validate_width;
    if (step_major_layout) {
      template_kv_rows =
          kv_seq_lens.view({validate_width, batch_size}).transpose(0, 1);
    } else {
      template_kv_rows = kv_seq_lens.view({batch_size, validate_width});
    }
  }
  if (template_kv_rows.defined()) {
    if (step_major_layout) {
      template_kv_rows =
          validate_input.input_params.attention.device.kv_seq_lens
              .view({validate_width, batch_size})
              .transpose(0, 1);
    } else {
      template_kv_rows =
          validate_input.input_params.attention.device.kv_seq_lens.view(
              {batch_size, validate_width});
    }
    torch::Tensor kv_delta =
        metadata.base_kv_seq_lens -
        template_kv_rows.select(/*dim=*/1, /*index=*/0).to(torch::kLong);
    torch::Tensor corrected_kv_rows =
        template_kv_rows.to(torch::kLong) + kv_delta.unsqueeze(1);
    validate_input.input_params.attention.device.kv_seq_lens =
        (step_major_layout ? corrected_kv_rows.transpose(0, 1)
                           : corrected_kv_rows)
            .flatten()
            .to(validate_input.input_params.attention.device.kv_seq_lens
                    .options());
  }

#if defined(USE_NPU)
  if (validate_input.input_params.graph.expanded_kv_seq_lens.defined()) {
    auto& params = validate_input.input_params;
    const torch::Tensor offsets =
        torch::arange(validate_width, metadata.base_kv_seq_lens.options());
    const torch::Tensor rows =
        metadata.base_kv_seq_lens.unsqueeze(1) + offsets.unsqueeze(0);
    const torch::Tensor expanded_kv_lens =
        (step_major_layout ? rows.transpose(0, 1) : rows)
            .flatten()
            .to(params.graph.expanded_kv_seq_lens.options());
    // The Python backend selects expanded_decode ahead of generic metadata.
    // Refresh lengths and page lists together after acceptance crosses a page.
    layer::ExpandedDecodeMetadataBuilder::populate_expanded_layout(
        ModelParamsView(params),
        expanded_kv_lens,
        params.graph.expanded_block_tables,
        params.graph.expanded_kv_seq_lens_vec,
        block_size);
  }
#endif

  torch::Tensor token_rows;
  if (step_major_layout) {
    token_rows = validate_input.token_ids.view({validate_width, batch_size})
                     .transpose(0, 1);
  } else {
    token_rows = validate_input.token_ids.view({batch_size, validate_width});
  }
  token_rows.select(/*dim=*/1, /*index=*/0)
      .copy_(metadata.last_tokens.to(validate_input.token_ids.options()),
             /*non_blocking=*/true);
  validate_input.runtime.device_tensors_ready = true;
}

template void prepare_next_draft_from_accepted_state(
    LlmForwardInput&,
    const LlmForwardInput&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    bool,
    bool,
    int32_t,
    bool,
    kernel::npu::MtpPrepareNextDraftWorkspace*);
template void prepare_later_draft_from_device_base(LlmForwardInput&,
                                                   const LlmForwardInput&,
                                                   const torch::Tensor&,
                                                   const torch::Tensor&,
                                                   int32_t,
                                                   int32_t);
template void prepare_target_verify_from_accepted_state(LlmForwardInput&,
                                                        const torch::Tensor&,
                                                        const torch::Tensor&,
                                                        const torch::Tensor&,
                                                        int32_t,
                                                        bool,
                                                        bool);
template void prepare_next_draft_from_accepted_state(
    LlmForwardInput&,
    const VlmForwardInput&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    bool,
    bool,
    int32_t,
    bool,
    kernel::npu::MtpPrepareNextDraftWorkspace*);
template void prepare_later_draft_from_device_base(LlmForwardInput&,
                                                   const VlmForwardInput&,
                                                   const torch::Tensor&,
                                                   const torch::Tensor&,
                                                   int32_t,
                                                   int32_t);
template void prepare_target_verify_from_accepted_state(VlmForwardInput&,
                                                        const torch::Tensor&,
                                                        const torch::Tensor&,
                                                        const torch::Tensor&,
                                                        int32_t,
                                                        bool,
                                                        bool);

}  // namespace xllm::mtp_async
