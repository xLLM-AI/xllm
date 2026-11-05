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

#pragma once

#include <glog/logging.h>
#include <torch/types.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "core/framework/batch/batch_forward_type.h"
#include "core/kv_cache/layout/seq_lens_layout.h"
#include "core/util/slice.h"

namespace xllm {

class LlmForwardInput;
class VlmForwardInput;
class LlmModelParams;
class VlmModelParams;

enum class DecodeFields : int8_t {
  kSlotsOnly = 0,
  kSlotsAndKvLen = 1,
  kAllFields = 2,
};

struct DecodeBuildBuffers {
  int32_t kv_max_seq_len = 0;
  std::vector<int32_t> out_token_ids;
  std::vector<int32_t> out_positions;
  std::vector<int32_t> out_kv_seq_lens;
  std::vector<int32_t> out_q_seq_lens;
  std::vector<int32_t> out_q_cu_seq_lens;
  std::vector<int32_t> out_new_cache_slots;
  std::vector<int32_t> out_block_tables;
  std::vector<int32_t> out_paged_kv_indptr;
  std::vector<int32_t> out_paged_kv_indices;
  std::vector<int32_t> out_paged_kv_last_page_len;
  std::vector<std::vector<std::vector<int32_t>>> out_multi_block_tables;
  std::vector<int32_t> out_num_scheduled_tokens;
  int32_t last_sequence_id = -1;
  int32_t out_block_table_stride = 0;
  DecodeFields fields = DecodeFields::kAllFields;
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
  int32_t num_sequences = 0;
  int32_t block_table_stride = 0;
  bool build_execution_batch = false;
};

void reserve_entries(DecodeBuildBuffers& buffers,
                     const DecodeBatchContext& ctx,
                     size_t num_rows,
                     DecodeFields fields);

struct QueryEntrySpec {
  int32_t seq_id = 0;
  int32_t token_id = 0;
  int32_t position_offset = 0;
  bool append_token = true;
  std::optional<int32_t> kv_len_offset;
  bool padding_block_cache_slot = false;
};

struct TokenWithOffset {
  int32_t token_id = 0;
  int32_t position_offset = 0;
};

template <typename Input>
DecodeBatchContext make_decode_batch_context(const Input& input);

int32_t token_at(const DecodeBatchContext& ctx, int32_t seq_id);

void append_decode_entry(const DecodeBatchContext& ctx,
                         const QueryEntrySpec& row,
                         int32_t block_size,
                         DecodeBuildBuffers& buf);

TokenWithOffset resolve_token_with_position_offset(
    int32_t input_token_id,
    const Slice<int64_t>& last_step_tokens,
    int32_t last_step_decode_num);

std::vector<int32_t> build_grouped_prefill_swa_slots(
    const DecodeBatchContext& ctx,
    int32_t block_size);
template <typename Params>
void commit_kv_and_slots(Params& params, DecodeBuildBuffers& buf) {
  params.meta.kv_max_seq_len = buf.kv_max_seq_len;
  params.attention.host.kv_seq_lens = std::move(buf.out_kv_seq_lens);
  params.attention.host.new_cache_slots = std::move(buf.out_new_cache_slots);
}

enum class QueryLayout : int8_t {
  kFlat = 0,
  // One entry per request carrying an N-token query.
  kChunked = 1,
  // One entry per query token sharing the full block KV length.
  kBlockParallel = 2,
  // Chunked row geometry whose commit is a plain DECODE step (DSpark
  // native-SAS): one N-token query row per request, no chunked-prefill mask.
  kChunkedDecode = 3,
};

inline bool is_chunked_layout(QueryLayout layout) {
  return layout == QueryLayout::kChunked ||
         layout == QueryLayout::kChunkedDecode;
}

// Every layout commits as DECODE except plain chunked rows, which commit as
// CHUNKED_PREFILL.
inline BatchForwardType commit_forward_type(QueryLayout layout) {
  return layout == QueryLayout::kChunked ? BatchForwardType::CHUNKED_PREFILL
                                         : BatchForwardType::DECODE;
}

struct DecodeCommitPlan {
  int32_t num_val_tokens = 1;
  QueryLayout layout = QueryLayout::kFlat;
  // The ACL graph-bound ABI binds rows + 1 cu entries instead of a bare
  // qo_indptr; only the graph-bound draft and verify paths set this.
  bool leading_zero_cu = false;
};

void reserve_plan_entries(DecodeBuildBuffers& buffers,
                          const DecodeCommitPlan& plan,
                          const DecodeBatchContext& ctx,
                          size_t num_rows);

// Each sequence emits num_val_tokens rows: the committed token plus its
// continuation slots.
struct ExpandDecodePlan : DecodeCommitPlan {
  int32_t block_size = 0;
  std::optional<int32_t> mask_token_id;
  bool step_major = false;
};

DecodeBuildBuffers expand_decode_inputs(const DecodeBatchContext& ctx,
                                        const ExpandDecodePlan& plan);

// Verify-step plan shared by the speculative and MTP validate paths: chunked
// rows when the verifier consumes one N-token query per sequence, flat rows
// otherwise.
inline ExpandDecodePlan make_verify_plan(int32_t num_val_tokens,
                                         int32_t block_size,
                                         QueryLayout layout,
                                         bool leading_zero_cu,
                                         bool step_major = false) {
  ExpandDecodePlan plan;
  plan.num_val_tokens = num_val_tokens;
  plan.block_size = block_size;
  plan.layout = layout;
  plan.leading_zero_cu = leading_zero_cu;
  plan.step_major = step_major;
  // The leading-zero cu ABI is only consumed by the chunked epilogue; on a
  // flat layout the flag would be silently ignored, so reject it up front.
  CHECK(!leading_zero_cu || is_chunked_layout(layout))
      << "leading-zero cu requires the chunked layout";
  return plan;
}

// Synthetic width-W masked query rows for an idle DP rank. Requires
// plan.mask_token_id. The rows always use the leading-zero cu layout, because
// a graph-bound ACL replay binds rows + 1 entries regardless of row geometry.
DecodeBuildBuffers make_idle_entries(const ExpandDecodePlan& plan);

inline void append_chunked_request(DecodeBuildBuffers& buf,
                                   const DecodeCommitPlan& plan,
                                   int32_t kv_len) {
  CuSeqLens::append_kv_len_and_max(
      buf.out_kv_seq_lens, kv_len, buf.kv_max_seq_len);
  CuSeqLens::append_q_len_and_qo_indptr(buf.out_q_seq_lens,
                                        buf.out_q_cu_seq_lens,
                                        plan.num_val_tokens,
                                        plan.leading_zero_cu);
}

// Takes the number of committed query entries the caller built. Block-parallel
// expansion is already reflected in that count, so only the per-sequence
// `actual_num_sequences` still needs scaling here.
template <typename Params>
void apply_commit_layout_meta(Params& input_params,
                              const DecodeCommitPlan& plan,
                              int32_t num_entries) {
  input_params.meta.num_sequences = num_entries;
  if (plan.layout == QueryLayout::kBlockParallel) {
    if (input_params.meta.actual_num_sequences > 0) {
      input_params.meta.actual_num_sequences *= plan.num_val_tokens;
    }
  }
  input_params.meta.q_max_seq_len =
      is_chunked_layout(plan.layout) ? plan.num_val_tokens : 1;
  input_params.meta.batch_forward_type = commit_forward_type(plan.layout);
}

template <typename Params>
void commit_decode_metadata(Params& input_params,
                            DecodeBuildBuffers& buf,
                            const DecodeCommitPlan& plan);
extern template void commit_decode_metadata(LlmModelParams& input_params,
                                            DecodeBuildBuffers& buf,
                                            const DecodeCommitPlan& plan);
extern template void commit_decode_metadata(VlmModelParams& input_params,
                                            DecodeBuildBuffers& buf,
                                            const DecodeCommitPlan& plan);

// Stages the token/position vectors as pinned host tensors and starts the
// non-blocking H2D copy. Sets `device_tensors_ready`.
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

template <typename Input>
void commit_decode_input(Input& input,
                         DecodeBuildBuffers& buf,
                         const DecodeCommitPlan& plan,
                         const torch::Device& target_device);
extern template void commit_decode_input(LlmForwardInput&,
                                         DecodeBuildBuffers&,
                                         const DecodeCommitPlan&,
                                         const torch::Device&);
extern template void commit_decode_input(VlmForwardInput&,
                                         DecodeBuildBuffers&,
                                         const DecodeCommitPlan&,
                                         const torch::Device&);

}  // namespace xllm
