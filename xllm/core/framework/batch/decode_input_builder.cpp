/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/common/constants.h"
#include "core/framework/config/model_config.h"
#include "core/framework/model/model_input_types.h"
#include "core/framework/model/seq_lens.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/vlm_forward_params.h"
#include "core/util/tensor_helper.h"

namespace xllm {

template <typename Input>
DecodeBatchContext make_decode_batch_context(const Input& input) {
  DecodeBatchContext ctx;
  ctx.num_requests = input.input_params.meta.num_sequences;
  ctx.build_execution_batch = ModelConfig::is_python_model_impl(
      ModelConfig::get_instance().model_impl());

  if (input.token_ids_host.defined()) {
    ctx.token_ids = tensor_slice(input.token_ids_host);
  }
  ctx.num_rope_axes =
      input.positions_host.dim() == 2 ? input.positions_host.size(0) : 1;
  CHECK(ctx.num_rope_axes == 1 || ctx.num_rope_axes == 3)
      << "positions must be one- or three-axis, axes=" << ctx.num_rope_axes;
  ctx.positions = tensor_slice(input.positions_host);
  CHECK_GE(static_cast<int32_t>(ctx.positions.size()), ctx.num_requests)
      << "positions size is smaller than num_requests, positions_size="
      << ctx.positions.size() << ", num_requests=" << ctx.num_requests;

  ctx.kv_seq_lens = input.input_params.attention.host.kv_seq_lens;
  ctx.q_seq_lens = input.input_params.attention.host.q_seq_lens;
  if (!input.input_params.multi_block_tables.empty()) {
    ctx.multi_block_tables_owner.reserve(
        input.input_params.multi_block_tables.size());
    for (const torch::Tensor& block_table :
         input.input_params.multi_block_tables) {
      ctx.multi_block_tables_owner.emplace_back(to_cpu_contiguous(block_table));
    }
    ctx.multi_block_tables.resize(ctx.multi_block_tables_owner.size());
    for (size_t m = 0; m < ctx.multi_block_tables_owner.size(); ++m) {
      const torch::Tensor& manager_table = ctx.multi_block_tables_owner[m];
      CHECK_EQ(manager_table.dim(), 2);
      ctx.multi_block_tables[m].reserve(static_cast<size_t>(ctx.num_requests));
      const Slice<int32_t> flat_block_table = tensor_slice(manager_table);
      const size_t request_stride = static_cast<size_t>(manager_table.size(1));
      for (int32_t req_id = 0; req_id < ctx.num_requests; ++req_id) {
        const size_t request_offset =
            static_cast<size_t>(req_id) * request_stride;
        ctx.multi_block_tables[m].emplace_back(flat_block_table.slice(
            request_offset, request_offset + request_stride));
      }
    }
    return ctx;
  }

  ctx.block_tables_owner =
      to_cpu_contiguous(input.input_params.attention.host.block_tables);
  CHECK_EQ(ctx.block_tables_owner.dim(), 2);
  CHECK_LE(ctx.num_requests, ctx.block_tables_owner.size(0))
      << "num_requests exceeds block table rows, num_requests="
      << ctx.num_requests
      << ", num_block_tables=" << ctx.block_tables_owner.size(0);
  ctx.block_table_stride = static_cast<int32_t>(ctx.block_tables_owner.size(1));
  ctx.block_tables = tensor_slice(ctx.block_tables_owner);
  return ctx;
}

std::vector<int32_t> build_grouped_prefill_swa_slots(
    const DecodeBatchContext& ctx,
    int32_t block_size) {
  CHECK(!ctx.multi_block_tables.empty())
      << "grouped prefill SWA slots require manager 0";

  std::vector<int32_t> slots;
  slots.reserve(ctx.positions.size());
  for (int32_t req_id = 0; req_id < ctx.num_requests; ++req_id) {
    const int32_t kv_len = ctx.kv_seq_lens[req_id];
    const int32_t q_len = ctx.q_seq_lens[req_id];
    CHECK_GE(kv_len, q_len) << "grouped prefill kv length must cover the query";
    const int32_t query_start = kv_len - q_len;
    const Slice<int32_t>& swa_block_table =
        ctx.multi_block_tables.front()[static_cast<size_t>(req_id)];
    for (int32_t query_idx = 0; query_idx < q_len; ++query_idx) {
      slots.emplace_back(util::compute_slot_mapping</*Ring=*/true>(
          swa_block_table, query_start + query_idx, block_size));
    }
  }
  return slots;
}

void prepare_decode_buffers(DecodeBuildBuffers& buffers,
                            const DecodeBatchContext& ctx,
                            size_t num_tokens,
                            bool emit_kv_len,
                            bool emit_block_tables,
                            bool emit_token) {
  buffers.emit_kv_len = emit_kv_len;
  buffers.emit_block_tables = emit_block_tables;
  buffers.build_execution_batch = ctx.build_execution_batch;
  buffers.out_token_ids.reserve(emit_token ? num_tokens : 0);
  buffers.out_positions.reserve(num_tokens);
  buffers.out_new_cache_slots.reserve(num_tokens);
  if (buffers.build_execution_batch) {
    buffers.out_num_scheduled_tokens.reserve(
        static_cast<size_t>(ctx.num_requests));
  }
  if (emit_kv_len) {
    buffers.out_kv_seq_lens.reserve(num_tokens);
  }
  if (emit_block_tables) {
    buffers.out_q_seq_lens.reserve(num_tokens);
    if (!ctx.multi_block_tables.empty()) {
      buffers.out_multi_block_tables.resize(ctx.multi_block_tables.size());
      for (std::vector<std::vector<int32_t>>& manager_tables :
           buffers.out_multi_block_tables) {
        manager_tables.reserve(num_tokens);
      }
    }
    const size_t block_table_stride =
        static_cast<size_t>(ctx.block_table_stride);
    if (block_table_stride > 0) {
      buffers.out_block_table_stride = ctx.block_table_stride;
      buffers.out_block_tables.reserve(num_tokens * block_table_stride);
      buffers.out_paged_kv_indices.reserve(num_tokens * block_table_stride);
      buffers.out_paged_kv_indptr.reserve(num_tokens + 1);
      buffers.out_paged_kv_last_page_len.reserve(num_tokens);
    }
  }
}

void append_decode_token(const DecodeBatchContext& ctx,
                         const DecodeTokenInput& token_input,
                         int32_t block_size,
                         DecodeBuildBuffers& buf) {
  CHECK_GE(token_input.req_id, 0);
  CHECK_LT(token_input.req_id, ctx.num_requests);
  const int32_t kv_len_base = ctx.kv_seq_lens[token_input.req_id];
  const int32_t model_position =
      ctx.positions[token_input.req_id] + token_input.position_offset;
  const int32_t cache_position = kv_len_base + token_input.position_offset - 1;
  CHECK_GE(model_position, 0) << "invalid decode model position";
  CHECK_GE(cache_position, 0) << "invalid decode cache position";

  Slice<int32_t> flat_block_table_slice;
  int32_t cache_slot = 0;
  if (!ctx.multi_block_tables.empty()) {
    flat_block_table_slice =
        ctx.multi_block_tables.front()[static_cast<size_t>(token_input.req_id)];
    cache_slot = util::compute_slot_mapping</*Ring=*/true>(
        flat_block_table_slice, cache_position, block_size);
  } else {
    const size_t request_stride = static_cast<size_t>(ctx.block_table_stride);
    const size_t request_offset =
        static_cast<size_t>(token_input.req_id) * request_stride;
    flat_block_table_slice =
        ctx.block_tables.slice(request_offset, request_offset + request_stride);
    cache_slot = util::compute_slot_mapping</*Ring=*/false>(
        flat_block_table_slice, cache_position, block_size);
  }
  if (token_input.cache_slot_override) {
    cache_slot = *token_input.cache_slot_override;
  }

  if (token_input.append_token) {
    buf.out_token_ids.emplace_back(token_input.token_id);
  }
  buf.out_positions.emplace_back(model_position);
  buf.out_new_cache_slots.emplace_back(cache_slot);

  if (buf.build_execution_batch) {
    if (buf.last_req_id != token_input.req_id) {
      buf.last_req_id = token_input.req_id;
      buf.out_num_scheduled_tokens.emplace_back(0);
    }
    ++buf.out_num_scheduled_tokens.back();
  }

  const int32_t kv_seq_len = token_input.kv_len_offset.has_value()
                                 ? kv_len_base + *token_input.kv_len_offset
                                 : cache_position + 1;
  if (buf.emit_kv_len) {
    buf.out_kv_seq_lens.emplace_back(kv_seq_len);
  }
  if (buf.emit_block_tables) {
    if (!ctx.multi_block_tables.empty()) {
      for (size_t m = 0; m < ctx.multi_block_tables.size(); ++m) {
        const Slice<int32_t>& block_table_slice =
            ctx.multi_block_tables[m][token_input.req_id];
        buf.out_multi_block_tables[m].emplace_back(block_table_slice.begin(),
                                                   block_table_slice.end());
      }
    } else {
      buf.out_block_tables.insert(buf.out_block_tables.end(),
                                  flat_block_table_slice.begin(),
                                  flat_block_table_slice.end());
      util::append_paged_kv_entry(flat_block_table_slice,
                                  kv_seq_len,
                                  block_size,
                                  buf.out_paged_kv_indptr,
                                  buf.out_paged_kv_indices,
                                  buf.out_paged_kv_last_page_len);
    }
    buf.out_q_seq_lens.emplace_back(1);
  }
}

std::pair<int32_t, int32_t> resolve_token_with_position_offset(
    int32_t input_token_id,
    const Slice<int64_t>& last_step_tokens,
    int32_t last_step_decode_num) {
  CHECK_GT(last_step_decode_num, 0);
  if (input_token_id >= 0) {
    return {input_token_id, 0};
  }

  const int32_t placeholder_idx = -input_token_id - 1;
  const int32_t base_idx = placeholder_idx * last_step_decode_num;
  CHECK_LE(base_idx + last_step_decode_num,
           static_cast<int32_t>(last_step_tokens.size()));

  int32_t token_id = 0;
  int32_t position_offset = -1;
  for (int32_t i = 0; i < last_step_decode_num; ++i) {
    const int64_t candidate = last_step_tokens[base_idx + i];
    if (candidate >= 0) {
      CHECK_LE(candidate,
               static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
      token_id = static_cast<int32_t>(candidate);
      ++position_offset;
    }
  }
  return {token_id, position_offset};
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

DecodeBuildBuffers expand_decode_inputs(
    const DecodeBatchContext& ctx,
    const DecodeExpansionOptions& decode_options) {
  const int32_t num_reqs = ctx.num_requests;
  const int32_t num_val_tokens = decode_options.num_val_tokens;
  const int32_t total = num_reqs * num_val_tokens;
  DecodeBuildBuffers buf;

  prepare_decode_buffers(buf,
                         ctx,
                         static_cast<size_t>(total),
                         /*emit_kv_len=*/!decode_options.per_request,
                         /*emit_block_tables=*/!decode_options.per_request);
  if (decode_options.per_request) {
    const size_t num_requests = static_cast<size_t>(num_reqs);
    buf.out_kv_seq_lens.reserve(num_requests);
    buf.out_q_seq_lens.reserve(num_requests);
  }

  auto append_token_input = [&](int32_t req_id, int32_t val_idx) {
    DecodeTokenInput token_input;
    token_input.req_id = req_id;
    token_input.token_id =
        val_idx == 0 ? ctx.token_ids[req_id]
                     : decode_options.mask_token_id.value_or(-val_idx);
    token_input.position_offset = val_idx;
    token_input.kv_len_offset = decode_options.causal
                                    ? std::nullopt
                                    : std::make_optional(num_val_tokens - 1);
    append_decode_token(ctx, token_input, decode_options.block_size, buf);
  };

  if (decode_options.step_major) {
    for (int32_t val_idx = 0; val_idx < num_val_tokens; ++val_idx) {
      for (int32_t req_id = 0; req_id < num_reqs; ++req_id) {
        append_token_input(req_id, val_idx);
      }
    }
  } else {
    for (int32_t req_id = 0; req_id < num_reqs; ++req_id) {
      for (int32_t val_idx = 0; val_idx < num_val_tokens; ++val_idx) {
        append_token_input(req_id, val_idx);
      }
    }
  }
  if (decode_options.per_request) {
    for (int32_t req_id = 0; req_id < num_reqs; ++req_id) {
      buf.out_kv_seq_lens.emplace_back(ctx.kv_seq_lens[req_id] +
                                       num_val_tokens - 1);
      buf.out_q_seq_lens.emplace_back(num_val_tokens);
    }
  }

  return buf;
}

template <typename Params>
void build_decode_metadata(Params& input_params,
                           DecodeBuildBuffers& buf,
                           const DecodeInputOptions& decode_options) {
  input_params.attention.host.q_cu_seq_lens =
      decode_options.leading_zero_cu ? util::query_start_loc(buf.out_q_seq_lens)
                                     : util::cu_num_tokens(buf.out_q_seq_lens);
  input_params.attention.host.q_seq_lens = std::move(buf.out_q_seq_lens);
  update_kv_lengths_and_slots(input_params, buf);
  if (!buf.out_multi_block_tables.empty()) {
    input_params.multi_block_tables =
        util::gather_block_tables(buf.out_multi_block_tables);
    input_params.attention.host.block_tables = torch::Tensor();
  } else if (buf.out_block_table_stride > 0) {
    const int32_t num_block_tables = static_cast<int32_t>(
        buf.out_block_tables.size() / buf.out_block_table_stride);
    input_params.attention.host.block_tables =
        make_pinned_cpu_tensor(buf.out_block_tables)
            .view({num_block_tables, buf.out_block_table_stride});
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
  update_decode_batch_metadata(
      input_params,
      decode_options,
      static_cast<int32_t>(input_params.attention.host.q_seq_lens.size()));
  if (buf.build_execution_batch) {
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

template void build_decode_metadata(LlmModelParams&,
                                    DecodeBuildBuffers&,
                                    const DecodeInputOptions&);
template void build_decode_metadata(VlmModelParams&,
                                    DecodeBuildBuffers&,
                                    const DecodeInputOptions&);

void build_idle_decode_input(LlmForwardInput& input,
                             const DecodeExpansionOptions& decode_options) {
  const int32_t query_width = decode_options.num_val_tokens;
  CHECK(decode_options.mask_token_id.has_value());
  DecodeBuildBuffers buf;
  buf.out_block_table_stride = 1;
  buf.build_execution_batch = true;
  buf.out_token_ids.assign(static_cast<size_t>(query_width),
                           *decode_options.mask_token_id);
  buf.out_positions.resize(static_cast<size_t>(query_width));
  std::iota(buf.out_positions.begin(), buf.out_positions.end(), 0);
  buf.out_new_cache_slots.assign(static_cast<size_t>(query_width), kPadSlotId);
  buf.out_block_tables.emplace_back(kPaddingBlockId);
  buf.out_kv_seq_lens.emplace_back(1);
  buf.out_q_seq_lens.emplace_back(query_width);
  util::append_paged_kv_entry(buf.out_block_tables,
                              /*kv_seq_len=*/1,
                              decode_options.block_size,
                              buf.out_paged_kv_indptr,
                              buf.out_paged_kv_indices,
                              buf.out_paged_kv_last_page_len);

  input.runtime.device_tensors_ready = false;
  input.token_ids_host = make_cpu_tensor(buf.out_token_ids);
  input.positions_host = make_cpu_tensor(buf.out_positions);
  input.token_ids = input.token_ids_host;
  input.positions = input.positions_host;
  DecodeExpansionOptions idle_options = decode_options;
  idle_options.leading_zero_cu = true;
  build_decode_metadata(input.input_params, buf, idle_options);
}

template <typename Input>
void build_decode_input(Input& input,
                        DecodeBuildBuffers& buf,
                        const DecodeInputOptions& decode_options,
                        const torch::Device& target_device) {
  set_token_position_tensors(input, buf.out_token_ids, buf.out_positions);
  build_decode_metadata(input.input_params, buf, decode_options);
  input.input_params.attention.rebuild_device_buffer(target_device);
}

template void build_decode_input(LlmForwardInput&,
                                 DecodeBuildBuffers&,
                                 const DecodeInputOptions&,
                                 const torch::Device&);
template void build_decode_input(VlmForwardInput&,
                                 DecodeBuildBuffers&,
                                 const DecodeInputOptions&,
                                 const torch::Device&);

}  // namespace xllm
