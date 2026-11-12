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

#pragma once

#include <glog/logging.h>
#include <torch/types.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "core/framework/batch/batch_forward_type.h"
#include "core/framework/model/seq_lens.h"
#include "core/util/slice.h"

namespace xllm {

class LlmForwardInput;
class VlmForwardInput;
class LlmModelParams;
class VlmModelParams;

struct DecodeBuildBuffers {
  std::vector<int32_t> out_token_ids;
  std::vector<int32_t> out_positions;
  std::vector<int32_t> out_kv_seq_lens;
  std::vector<int32_t> out_q_seq_lens;
  std::vector<int32_t> out_new_cache_slots;
  std::vector<int32_t> out_block_tables;
  std::vector<int32_t> out_paged_kv_indptr;
  std::vector<int32_t> out_paged_kv_indices;
  std::vector<int32_t> out_paged_kv_last_page_len;
  std::vector<std::vector<std::vector<int32_t>>> out_multi_block_tables;
  std::vector<int32_t> out_num_scheduled_tokens;
  int32_t last_req_id = -1;
  int32_t out_block_table_stride = 0;
  bool emit_kv_len = true;
  bool emit_block_tables = true;
  bool build_execution_batch = false;
};

struct DecodeBatchContext {
  torch::Tensor block_tables_owner;
  std::vector<torch::Tensor> multi_block_tables_owner;
  Slice<int32_t> token_ids;
  Slice<int32_t> positions;
  Slice<int32_t> kv_seq_lens;
  Slice<int32_t> q_seq_lens;
  Slice<int32_t> block_tables;
  std::vector<std::vector<Slice<int32_t>>> multi_block_tables;
  int32_t num_requests = 0;
  int32_t num_rope_axes = 1;
  int32_t block_table_stride = 0;
  bool build_execution_batch = false;
};

void prepare_decode_buffers(DecodeBuildBuffers& buffers,
                            const DecodeBatchContext& ctx,
                            size_t num_tokens,
                            bool emit_kv_len,
                            bool emit_block_tables,
                            bool emit_token = true);

struct DecodeTokenInput {
  int32_t req_id = 0;
  int32_t token_id = 0;
  int32_t position_offset = 0;
  bool append_token = true;
  std::optional<int32_t> kv_len_offset;
  std::optional<int32_t> cache_slot_override;
};

template <typename Input>
DecodeBatchContext make_decode_batch_context(const Input& input);

void append_decode_token(const DecodeBatchContext& ctx,
                         const DecodeTokenInput& token_input,
                         int32_t block_size,
                         DecodeBuildBuffers& buf);

std::pair<int32_t, int32_t> resolve_token_with_position_offset(
    int32_t input_token_id,
    const Slice<int64_t>& last_step_tokens,
    int32_t last_step_decode_num);

std::vector<int32_t> build_grouped_prefill_swa_slots(
    const DecodeBatchContext& ctx,
    int32_t block_size);
template <typename Params>
void update_kv_lengths_and_slots(Params& params, DecodeBuildBuffers& buf) {
  params.meta.kv_max_seq_len = util::max_len(buf.out_kv_seq_lens);
  params.attention.host.kv_seq_lens = std::move(buf.out_kv_seq_lens);
  params.attention.host.new_cache_slots = std::move(buf.out_new_cache_slots);
}

struct DecodeInputOptions {
  int32_t num_val_tokens = 1;
  bool per_request = false;
  BatchForwardType forward_type = BatchForwardType::DECODE;
  // Emit query_start_loc (leading 0) instead of cu_num_tokens (zero-free).
  bool leading_zero_cu = false;
  bool causal = true;
};

struct DecodeExpansionOptions : DecodeInputOptions {
  int32_t block_size = 0;
  std::optional<int32_t> mask_token_id;
  bool step_major = false;
  bool sample_from_anchor = false;
};

DecodeBuildBuffers expand_decode_inputs(
    const DecodeBatchContext& ctx,
    const DecodeExpansionOptions& decode_options);

inline DecodeExpansionOptions make_verify_layout(int32_t num_val_tokens,
                                                 int32_t block_size,
                                                 bool chunked,
                                                 bool step_major = false) {
  DecodeExpansionOptions decode_options;
  decode_options.num_val_tokens = num_val_tokens;
  decode_options.block_size = block_size;
  decode_options.per_request = chunked;
  decode_options.forward_type =
      chunked ? BatchForwardType::CHUNKED_PREFILL : BatchForwardType::DECODE;
  decode_options.leading_zero_cu = chunked;
  decode_options.step_major = step_major;
  return decode_options;
}

template <typename Params>
void update_decode_batch_metadata(Params& input_params,
                                  const DecodeInputOptions& decode_options,
                                  int32_t num_requests) {
  input_params.meta.num_sequences = num_requests;
  if (!decode_options.per_request && !decode_options.causal &&
      input_params.meta.actual_num_sequences > 0) {
    input_params.meta.actual_num_sequences *= decode_options.num_val_tokens;
  }
  input_params.meta.q_max_seq_len =
      decode_options.per_request ? decode_options.num_val_tokens : 1;
  input_params.meta.batch_forward_type = decode_options.forward_type;
}

template <typename Params>
void build_decode_metadata(Params& input_params,
                           DecodeBuildBuffers& buf,
                           const DecodeInputOptions& decode_options);
extern template void build_decode_metadata(
    LlmModelParams& input_params,
    DecodeBuildBuffers& buf,
    const DecodeInputOptions& decode_options);
extern template void build_decode_metadata(
    VlmModelParams& input_params,
    DecodeBuildBuffers& buf,
    const DecodeInputOptions& decode_options);

template <typename Input>
void set_token_position_tensors(Input& input,
                                const std::vector<int32_t>& token_ids,
                                const std::vector<int32_t>& positions);
extern template void set_token_position_tensors(LlmForwardInput&,
                                                const std::vector<int32_t>&,
                                                const std::vector<int32_t>&);
extern template void set_token_position_tensors(VlmForwardInput&,
                                                const std::vector<int32_t>&,
                                                const std::vector<int32_t>&);

void build_idle_decode_input(LlmForwardInput& input,
                             const DecodeExpansionOptions& decode_options);

template <typename Input>
void build_decode_input(Input& input,
                        DecodeBuildBuffers& buf,
                        const DecodeInputOptions& decode_options,
                        const torch::Device& target_device);
extern template void build_decode_input(LlmForwardInput&,
                                        DecodeBuildBuffers&,
                                        const DecodeInputOptions&,
                                        const torch::Device&);
extern template void build_decode_input(VlmForwardInput&,
                                        DecodeBuildBuffers&,
                                        const DecodeInputOptions&,
                                        const torch::Device&);

}  // namespace xllm
