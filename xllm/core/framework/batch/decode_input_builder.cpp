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

#include "core/framework/batch/decode_input_builder.h"

#include <glog/logging.h>

#include <limits>
#include <numeric>

#include "core/framework/config/model_config.h"
#include "core/framework/model/model_input_types.h"
#include "core/kv_cache/layout/paged_kv_layout.h"
#include "core/kv_cache/layout/seq_lens_layout.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/vlm_forward_params.h"
#include "core/util/tensor_helper.h"

namespace xllm {

namespace {

void fill_multi_block_table_slices(DecodeBatchContext& ctx) {
  ctx.multi_block_tables.resize(ctx.multi_block_tables_owner.size());
  for (size_t m = 0; m < ctx.multi_block_tables_owner.size(); ++m) {
    const torch::Tensor& manager_table = ctx.multi_block_tables_owner[m];
    CHECK(manager_table.defined());
    CHECK_EQ(manager_table.dim(), 2);
    ctx.multi_block_tables[m].reserve(static_cast<size_t>(ctx.num_sequences));
    const Slice<int32_t> flat_block_table = tensor_slice(manager_table);
    const size_t row_stride = static_cast<size_t>(manager_table.size(1));
    for (int32_t seq_id = 0; seq_id < ctx.num_sequences; ++seq_id) {
      ctx.multi_block_tables[m].emplace_back(ubatch_slices(
          flat_block_table, static_cast<size_t>(seq_id), row_stride));
    }
  }
}

}  // namespace

template <typename Input>
DecodeBatchContext make_decode_batch_context(const Input& input) {
  DecodeBatchContext ctx;
  ctx.num_sequences = input.input_params.meta.num_sequences;
  CHECK_GE(ctx.num_sequences, 0) << "invalid num_sequences";
  ctx.build_execution_batch = ModelConfig::is_python_model_impl(
      ModelConfig::get_instance().model_impl());

  if (input.token_ids_host.defined()) {
    ctx.token_ids = tensor_slice(input.token_ids_host);
  }
  CHECK(input.positions_host.defined())
      << "positions_host must be defined for decode input build";
  ctx.positions = tensor_slice(input.positions_host);
  CHECK_GE(static_cast<int32_t>(ctx.positions.size()), ctx.num_sequences)
      << "positions size is smaller than num_sequences, positions_size="
      << ctx.positions.size() << ", num_sequences=" << ctx.num_sequences;

  ctx.kv_seq_lens = input.input_params.attention.host.kv_seq_lens;
  ctx.q_seq_lens = input.input_params.attention.host.q_seq_lens;
  if (!input.input_params.multi_block_tables.empty()) {
    ctx.multi_block_tables_owner.reserve(
        input.input_params.multi_block_tables.size());
    for (const torch::Tensor& block_table :
         input.input_params.multi_block_tables) {
      ctx.multi_block_tables_owner.emplace_back(to_cpu_contiguous(block_table));
    }
    fill_multi_block_table_slices(ctx);
    return ctx;
  }

  CHECK(input.input_params.attention.host.block_tables.defined())
      << "host block_tables must be defined for decode input build";
  ctx.block_tables_owner =
      to_cpu_contiguous(input.input_params.attention.host.block_tables);
  CHECK_EQ(ctx.block_tables_owner.dim(), 2);
  CHECK_LE(ctx.num_sequences, ctx.block_tables_owner.size(0))
      << "num_sequences exceeds block table rows, num_sequences="
      << ctx.num_sequences
      << ", block_table_rows=" << ctx.block_tables_owner.size(0);
  ctx.block_table_stride = static_cast<int32_t>(ctx.block_tables_owner.size(1));
  ctx.block_tables = tensor_slice(ctx.block_tables_owner);
  return ctx;
}

int32_t token_at(const DecodeBatchContext& ctx, int32_t seq_id) {
  CHECK_LT(static_cast<size_t>(seq_id), ctx.token_ids.size());
  return ctx.token_ids[seq_id];
}

std::vector<int32_t> build_grouped_prefill_swa_slots(
    const DecodeBatchContext& ctx,
    int32_t block_size) {
  CHECK(!ctx.multi_block_tables.empty())
      << "grouped prefill SWA slots require manager 0";

  std::vector<int32_t> slots;
  slots.reserve(ctx.positions.size());
  for (int32_t seq_id = 0; seq_id < ctx.num_sequences; ++seq_id) {
    const int32_t kv_len = CuSeqLens::at(ctx.kv_seq_lens, seq_id);
    const int32_t q_len = CuSeqLens::at(ctx.q_seq_lens, seq_id);
    CHECK_GE(kv_len, q_len) << "grouped prefill kv length must cover the query";
    const int32_t query_start = kv_len - q_len;
    const Slice<int32_t>& swa_block_table =
        ctx.multi_block_tables.front()[static_cast<size_t>(seq_id)];
    for (int32_t query_idx = 0; query_idx < q_len; ++query_idx) {
      slots.emplace_back(compute_slot_mapping</*kRing=*/true>(
          swa_block_table, query_start + query_idx, block_size));
    }
  }
  return slots;
}

void reserve_entries(DecodeBuildBuffers& buffers,
                     const DecodeBatchContext& ctx,
                     size_t num_rows,
                     DecodeFields fields) {
  buffers.fields = fields;
  buffers.out_token_ids.reserve(num_rows);
  buffers.out_positions.reserve(num_rows);
  buffers.out_new_cache_slots.reserve(num_rows);
  if (ctx.build_execution_batch) {
    buffers.out_num_scheduled_tokens.reserve(
        static_cast<size_t>(ctx.num_sequences));
  }
  if (fields >= DecodeFields::kSlotsAndKvLen) {
    buffers.out_kv_seq_lens.reserve(CuSeqLens::capacity_for(num_rows));
  }
  if (fields >= DecodeFields::kAllFields) {
    buffers.out_q_seq_lens.reserve(CuSeqLens::capacity_for(num_rows));
    buffers.out_q_cu_seq_lens.reserve(num_rows);
    if (!ctx.multi_block_tables.empty()) {
      buffers.out_multi_block_tables.resize(ctx.multi_block_tables.size());
      for (std::vector<std::vector<int32_t>>& manager_tables :
           buffers.out_multi_block_tables) {
        manager_tables.reserve(num_rows);
      }
    }
    const size_t block_table_stride =
        static_cast<size_t>(ctx.block_table_stride);
    if (block_table_stride > 0) {
      buffers.out_block_table_stride = ctx.block_table_stride;
      buffers.out_block_tables.reserve(num_rows * block_table_stride);
      buffers.out_paged_kv_indices.reserve(num_rows * block_table_stride);
      buffers.out_paged_kv_indptr.reserve(num_rows + 1);
      buffers.out_paged_kv_last_page_len.reserve(num_rows);
    }
  }
}

void reserve_plan_entries(DecodeBuildBuffers& buffers,
                          const DecodeCommitPlan& plan,
                          const DecodeBatchContext& ctx,
                          size_t num_rows) {
  const bool chunked = is_chunked_layout(plan.layout);
  reserve_entries(
      buffers,
      ctx,
      num_rows,
      chunked ? DecodeFields::kSlotsOnly : DecodeFields::kAllFields);
  if (chunked) {
    const size_t num_requests = static_cast<size_t>(ctx.num_sequences);
    buffers.out_kv_seq_lens.reserve(CuSeqLens::capacity_for(num_requests));
    buffers.out_q_seq_lens.reserve(CuSeqLens::capacity_for(num_requests));
    buffers.out_q_cu_seq_lens.reserve(CuSeqLens::capacity_for(num_requests) +
                                      (plan.leading_zero_cu ? 1 : 0));
  }
}

void append_decode_entry(const DecodeBatchContext& ctx,
                         const QueryEntrySpec& row,
                         int32_t block_size,
                         DecodeBuildBuffers& buf) {
  CHECK_GE(row.seq_id, 0);
  CHECK_LT(row.seq_id, ctx.num_sequences);
  const int32_t kv_len_base = CuSeqLens::at(ctx.kv_seq_lens, row.seq_id);
  const int32_t model_position =
      ctx.positions[row.seq_id] + row.position_offset;
  const int32_t cache_position = kv_len_base + row.position_offset - 1;
  CHECK_GE(model_position, 0) << "invalid decode model position";
  CHECK_GE(cache_position, 0) << "invalid decode cache position";

  Slice<int32_t> flat_block_table_slice;
  int32_t cache_slot = 0;
  if (!row.padding_block_cache_slot) {
    if (!ctx.multi_block_tables.empty()) {
      flat_block_table_slice =
          ctx.multi_block_tables.front()[static_cast<size_t>(row.seq_id)];
      cache_slot = compute_slot_mapping</*kRing=*/true>(
          flat_block_table_slice, cache_position, block_size);
    } else {
      flat_block_table_slice =
          ubatch_slices(ctx.block_tables,
                        static_cast<size_t>(row.seq_id),
                        static_cast<size_t>(ctx.block_table_stride));
      cache_slot = compute_slot_mapping</*kRing=*/false>(
          flat_block_table_slice, cache_position, block_size);
    }
  }

  if (row.append_token) {
    buf.out_token_ids.emplace_back(row.token_id);
  }
  buf.out_positions.emplace_back(model_position);
  buf.out_new_cache_slots.emplace_back(cache_slot);

  if (ctx.build_execution_batch) {
    if (buf.last_sequence_id != row.seq_id) {
      buf.last_sequence_id = row.seq_id;
      buf.out_num_scheduled_tokens.emplace_back(0);
    }
    ++buf.out_num_scheduled_tokens.back();
  }

  const int32_t row_kv_seq_len = row.kv_len_offset.has_value()
                                     ? kv_len_base + *row.kv_len_offset
                                     : cache_position + 1;
  if (buf.fields >= DecodeFields::kSlotsAndKvLen) {
    CuSeqLens::append_kv_len_and_max(
        buf.out_kv_seq_lens, row_kv_seq_len, buf.kv_max_seq_len);
  }
  if (buf.fields >= DecodeFields::kAllFields) {
    if (!ctx.multi_block_tables.empty()) {
      for (size_t m = 0; m < ctx.multi_block_tables.size(); ++m) {
        const Slice<int32_t>& block_table_slice =
            ctx.multi_block_tables[m][row.seq_id];
        buf.out_multi_block_tables[m].emplace_back(block_table_slice.begin(),
                                                   block_table_slice.end());
      }
    } else {
      buf.out_block_tables.insert(buf.out_block_tables.end(),
                                  flat_block_table_slice.begin(),
                                  flat_block_table_slice.end());
      append_paged_kv_metadata(row_kv_seq_len,
                               flat_block_table_slice,
                               block_size,
                               buf.out_paged_kv_indices,
                               buf.out_paged_kv_indptr,
                               buf.out_paged_kv_last_page_len);
    }
    // A per-token row contributes one qo_indptr entry; the graph-bound
    // leading-zero layout is applied once per request by the chunked epilogue.
    CuSeqLens::append_q_len_and_qo_indptr(buf.out_q_seq_lens,
                                          buf.out_q_cu_seq_lens,
                                          /*length=*/1,
                                          /*leading_zero=*/false);
  }
}

TokenWithOffset resolve_token_with_position_offset(
    int32_t input_token_id,
    const Slice<int64_t>& last_step_tokens,
    int32_t last_step_decode_num) {
  CHECK_GT(last_step_decode_num, 0);
  if (input_token_id >= 0) {
    TokenWithOffset direct;
    direct.token_id = input_token_id;
    direct.position_offset = 0;
    return direct;
  }

  const int32_t placeholder_idx = -input_token_id - 1;
  const int32_t base_idx = placeholder_idx * last_step_decode_num;
  CHECK_LE(base_idx + last_step_decode_num,
           static_cast<int32_t>(last_step_tokens.size()));

  TokenWithOffset resolved;
  // position_offset ends at accepted_count - 1: the -1 seed means "no
  // candidate accepted yet", and token_id holds the last non-negative token.
  resolved.position_offset = -1;
  for (int32_t i = 0; i < last_step_decode_num; ++i) {
    const int64_t candidate = last_step_tokens[base_idx + i];
    if (candidate >= 0) {
      CHECK_LE(candidate,
               static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
      resolved.token_id = static_cast<int32_t>(candidate);
      resolved.position_offset += 1;
    }
  }
  return resolved;
}

template <typename Input>
void set_token_position_tensors(Input& input,
                                const std::vector<int32_t>& token_ids,
                                const std::vector<int32_t>& positions) {
  input.runtime.device_tensors_ready = false;
  input.token_ids_host = make_pinned_cpu_tensor(token_ids);
  input.positions_host = make_pinned_cpu_tensor(positions);
  input.token_ids = safe_to(input.token_ids_host,
                            input.token_ids.options(),
                            /*non_blocking=*/true);
  input.positions = safe_to(input.positions_host,
                            input.positions.options(),
                            /*non_blocking=*/true);
  input.runtime.device_tensors_ready = true;
}

template void set_token_position_tensors(LlmForwardInput&,
                                         const std::vector<int32_t>&,
                                         const std::vector<int32_t>&);
template void set_token_position_tensors(VlmForwardInput&,
                                         const std::vector<int32_t>&,
                                         const std::vector<int32_t>&);

template DecodeBatchContext make_decode_batch_context(const LlmForwardInput&);
template DecodeBatchContext make_decode_batch_context(const VlmForwardInput&);

DecodeBuildBuffers expand_decode_inputs(const DecodeBatchContext& ctx,
                                        const ExpandDecodePlan& plan) {
  const bool chunked = is_chunked_layout(plan.layout);
  const bool block_parallel = plan.layout == QueryLayout::kBlockParallel;
  if (block_parallel) {
    CHECK(!ctx.multi_block_tables.empty())
        << "block-parallel rows require grouped KV tables";
  }
  const int32_t num_sequences = ctx.num_sequences;
  const int32_t num_val_tokens = plan.num_val_tokens;
  const int32_t total = num_sequences * num_val_tokens;
  DecodeBuildBuffers buf;

  reserve_plan_entries(buf, plan, ctx, static_cast<size_t>(total));

  auto emit_row = [&](int32_t seq_id, int32_t val_idx) {
    QueryEntrySpec row;
    row.seq_id = seq_id;
    row.token_id = val_idx == 0 ? token_at(ctx, seq_id)
                                : plan.mask_token_id.value_or(-val_idx);
    row.position_offset = val_idx;
    row.kv_len_offset =
        block_parallel ? std::make_optional(num_val_tokens - 1) : std::nullopt;
    append_decode_entry(ctx, row, plan.block_size, buf);
  };

  if (plan.step_major) {
    for (int32_t val_idx = 0; val_idx < num_val_tokens; ++val_idx) {
      for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
        emit_row(seq_id, val_idx);
      }
    }
  } else {
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      for (int32_t val_idx = 0; val_idx < num_val_tokens; ++val_idx) {
        emit_row(seq_id, val_idx);
      }
    }
  }
  if (chunked) {
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      append_chunked_request(
          buf,
          plan,
          CuSeqLens::at(ctx.kv_seq_lens, seq_id) + num_val_tokens - 1);
    }
  }

  return buf;
}

DecodeBuildBuffers make_idle_entries(const ExpandDecodePlan& plan) {
  const int32_t query_width = plan.num_val_tokens;
  CHECK(plan.mask_token_id.has_value()) << "idle rows require a mask token id";
  // The idle rank stands in for one synthetic sequence whose query tokens all
  // land on padding block 0 and never touch a real cache slot. Rows are filled
  // directly rather than through append_decode_entry, so the only buffer state
  // commit needs is the single-column padding block table.
  DecodeBuildBuffers buf;
  buf.out_block_table_stride = 1;

  buf.out_token_ids.assign(static_cast<size_t>(query_width),
                           *plan.mask_token_id);
  buf.out_positions.resize(static_cast<size_t>(query_width));
  std::iota(buf.out_positions.begin(), buf.out_positions.end(), 0);
  buf.out_new_cache_slots.assign(static_cast<size_t>(query_width),
                                 kSkippedSlotId);
  buf.out_block_tables.emplace_back(kPaddingBlockId);
  CuSeqLens::append_kv_len_and_max(buf.out_kv_seq_lens, 1, buf.kv_max_seq_len);
  // Idle ranks are replayed by a graph-bound ACL path, which always binds
  // rows + 1 cu entries.
  CuSeqLens::append_q_len_and_qo_indptr(
      buf.out_q_seq_lens, buf.out_q_cu_seq_lens, query_width, true);
  buf.out_num_scheduled_tokens = {query_width};
  append_paged_kv_metadata(1,
                           buf.out_block_tables,
                           plan.block_size,
                           buf.out_paged_kv_indices,
                           buf.out_paged_kv_indptr,
                           buf.out_paged_kv_last_page_len);
  return buf;
}

template <typename Params>
void commit_decode_metadata(Params& input_params,
                            DecodeBuildBuffers& buf,
                            const DecodeCommitPlan& plan) {
  const int32_t num_entries = CuSeqLens::entry_count(buf.out_q_seq_lens);
  input_params.attention.host.q_seq_lens = std::move(buf.out_q_seq_lens);
  input_params.attention.host.q_cu_seq_lens = std::move(buf.out_q_cu_seq_lens);
  commit_kv_and_slots(input_params, buf);
  if (!buf.out_multi_block_tables.empty()) {
    input_params.multi_block_tables =
        pad_and_stack_grouped_tables(buf.out_multi_block_tables);
    input_params.attention.host.block_tables = torch::Tensor();
  } else if (buf.out_block_table_stride > 0) {
    const int32_t block_table_rows = static_cast<int32_t>(
        buf.out_block_tables.size() / buf.out_block_table_stride);
    input_params.attention.host.block_tables =
        make_pinned_cpu_tensor(buf.out_block_tables)
            .view({block_table_rows, buf.out_block_table_stride});
    input_params.multi_block_tables.clear();
    if (!buf.out_paged_kv_indptr.empty()) {
      input_params.attention.device.paged_kv_indptr =
          make_pinned_cpu_tensor(buf.out_paged_kv_indptr);
      input_params.attention.device.paged_kv_indices =
          make_pinned_cpu_tensor(buf.out_paged_kv_indices);
      input_params.attention.device.paged_kv_last_page_len =
          make_pinned_cpu_tensor(buf.out_paged_kv_last_page_len);
    } else {
      input_params.attention.device.paged_kv_indptr = torch::Tensor();
      input_params.attention.device.paged_kv_indices = torch::Tensor();
      input_params.attention.device.paged_kv_last_page_len = torch::Tensor();
    }
  }
  apply_commit_layout_meta(input_params, plan, num_entries);
  if (!buf.out_num_scheduled_tokens.empty()) {
    auto& execution_batch = input_params.meta.execution_batch;
    execution_batch.num_tokens = static_cast<int64_t>(buf.out_positions.size());
    execution_batch.num_scheduled_tokens =
        std::move(buf.out_num_scheduled_tokens);
    execution_batch.num_reqs =
        static_cast<int32_t>(execution_batch.num_scheduled_tokens.size());
    fill_request_layout(execution_batch,
                        std::span<const int32_t>(buf.out_positions));
    execution_batch.is_prefilling.assign(
        execution_batch.num_reqs,
        input_params.meta.batch_forward_type.no_decode());
  }
}

template void commit_decode_metadata(LlmModelParams&,
                                     DecodeBuildBuffers&,
                                     const DecodeCommitPlan&);
template void commit_decode_metadata(VlmModelParams&,
                                     DecodeBuildBuffers&,
                                     const DecodeCommitPlan&);

template <typename Input>
void commit_decode_input(Input& input,
                         DecodeBuildBuffers& buf,
                         const DecodeCommitPlan& plan,
                         const torch::Device& target_device) {
  set_token_position_tensors(input, buf.out_token_ids, buf.out_positions);
  commit_decode_metadata(input.input_params, buf, plan);
  input.input_params.attention.rebuild_device_buffer(target_device);
}

template void commit_decode_input(LlmForwardInput&,
                                  DecodeBuildBuffers&,
                                  const DecodeCommitPlan&,
                                  const torch::Device&);
template void commit_decode_input(VlmForwardInput&,
                                  DecodeBuildBuffers&,
                                  const DecodeCommitPlan&,
                                  const torch::Device&);

}  // namespace xllm
