/* Copyright 2025-2026 The xLLM Authors.

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

#include <cstdint>
#include <utility>
#include <vector>

#include "core/framework/model/mtp_topk_state.h"
#include "util/slice.h"

namespace xllm {

struct ModelInputParams;
struct ForwardInput;
struct SamplingParameters;

namespace specBuilder {

struct DecodeBuildMeta {
  int32_t kv_max_seq_len = 0;
};

// Aggregated output vectors produced by row builders.
// Callers convert these vectors to tensors and write back to input params.
struct DecodeBuildBuffers {
  DecodeBuildMeta meta;
  std::vector<int32_t> out_token_ids;
  std::vector<int32_t> out_positions;
  std::vector<int32_t> out_kv_seq_lens;
  std::vector<int32_t> out_q_seq_lens;
  std::vector<int32_t> out_q_cu_seq_lens;
  std::vector<int32_t> out_new_cache_slots;
  std::vector<int32_t> out_block_tables;
  std::vector<std::vector<std::vector<int32_t>>> out_multi_block_tables;
  int32_t out_block_table_rows = 0;
  int32_t out_block_table_stride = 0;
};

// CPU-side row context shared by decode row builders.
// It owns the contiguous block table tensor so row slices stay valid.
struct DecodeRowContext {
  torch::Tensor block_tables_owner;
  std::vector<torch::Tensor> multi_block_tables_owner;
  Slice<int32_t> token_ids;
  Slice<int32_t> positions;
  Slice<int32_t> kv_seq_lens;
  Slice<int32_t> block_tables;
  std::vector<std::vector<Slice<int32_t>>> multi_block_tables;
  int32_t num_sequences = 0;
  int32_t block_table_stride = 0;
  bool model_managed_multiblock = false;
};

// Declarative spec for one emitted decode row.
// A row can selectively append token/kv/q/block fields depending on caller
// needs.
struct RowSpec {
  int32_t seq_id = 0;
  // When true, token_id is ignored and token comes from input
  // token_ids[seq_id].
  bool use_input_token = false;
  int32_t token_id = 0;
  int32_t position_offset = 0;
  bool append_token = true;
  bool append_kv_len = true;
  bool append_q_len_one = false;
  bool append_block_table = false;
};

// Resolved token and relative position offset for placeholder token handling.
struct TokenWithOffset {
  int32_t token_id = 0;
  int32_t position_offset = 0;
};

// Builds a reusable row context from ForwardInput host-side fields.
DecodeRowContext make_decode_row_context(const ForwardInput& input);

// Appends one logical decode row into output buffers.
void append_decode_row(const DecodeRowContext& ctx,
                       const RowSpec& row,
                       int32_t block_size,
                       DecodeBuildBuffers& buf);

// Resolves direct/placeholder input token to a real token and position offset.
// Placeholder tokens use negative ids (-1, -2, ...) and are read from last-step
// outputs.
TokenWithOffset resolve_token_with_position_offset(
    int32_t input_token_id,
    int32_t seq_id,
    const Slice<int64_t>& last_step_tokens,
    int32_t last_step_decode_num);

// Resolves a token from last-step output and appends one decode row.
void append_decode_row_from_last_step(const DecodeRowContext& ctx,
                                      int32_t seq_id,
                                      int32_t input_token_id,
                                      const Slice<int64_t>& last_step_tokens,
                                      int32_t last_step_decode_num,
                                      int32_t block_size,
                                      DecodeBuildBuffers& buf);

// Computes one cache slot id from absolute position and block table mapping.
int32_t calc_slot_id(int32_t position,
                     const Slice<int32_t>& block_table_slice,
                     int32_t block_size);

// Computes sequence kv length with platform-specific seq-lens layout handling.
int32_t calc_kv_len(const Slice<int32_t>& kv_seq_lens_slice,
                    int32_t seq_id,
                    int32_t offset);

// Appends one q/kv length element using current backend layout policy.
void append_seq_len_by_layout(std::vector<int32_t>& vec, int32_t len);

// Appends one q length and the matching cumulative q length.
void append_q_seq_len(std::vector<int32_t>& q_seq_lens,
                      std::vector<int32_t>& q_cu_seq_lens,
                      int32_t len);

// Appends kv_len into output vector and updates kv_max_seq_len.
void update_kv_seq_lens_and_max(std::vector<int32_t>& kv_seq_lens_vec,
                                int32_t kv_len,
                                int32_t& kv_max_seq_len);

// Builds q_cu_seq_lens tensor from upstream-provided host values.
// When include_leading_zero is true, returns query_start_loc-style
// [0, cumsum...].
torch::Tensor build_q_cu_seq_lens_tensor(const ModelInputParams& params,
                                         torch::Device device = torch::kCPU,
                                         bool include_leading_zero = false);

// Updates common decode-side ModelInputParams fields from built buffers.
void update_input_params(ModelInputParams& input_params,
                         DecodeBuildBuffers& buf,
                         int32_t q_max_seq_len,
                         std::vector<int32_t> q_seq_lens_vec,
                         std::vector<int32_t> q_cu_seq_lens_vec,
                         int32_t kv_max_seq_len,
                         std::vector<int32_t> kv_seq_lens_vec,
                         bool update_block_tables = false);

// Packs a host int32 vector into a pinned CPU tensor for async H2D staging.
torch::Tensor make_cpu_int_tensor(const std::vector<int32_t>& values);

// Stages token_ids/positions into both the host and device tensors of `input`
// with async H2D copies, toggling device_tensors_ready around the write.
void set_token_position_tensors(ForwardInput& input,
                                const std::vector<int32_t>& token_ids,
                                const std::vector<int32_t>& positions,
                                const torch::TensorOptions& token_options,
                                const torch::TensorOptions& position_options);

// Reuses the NPU MTP draft-step row-selection flow for every backend. Backend
// state implementations only interpret index_select_rows() for their payload.
MtpTopkStatePtr select_mtp_topk_state_for_next_step(
    const MtpTopkStatePtr& state,
    const SamplingParameters& sampling_params);

namespace draftProbs {

// Compress draft probs to selected-only format [batch_size] for cache storage.
// Input draft_probs may be dense [batch_size, vocab_size] or selected-only
// [batch_size] / [batch_size, 1].
torch::Tensor compress_for_cache(const torch::Tensor& draft_probs,
                                 const torch::Tensor& draft_token_ids);

// Build validate inputs from per-step draft token ids/probs.
// Returns:
//   - draft_token_ids: [batch_size, n_speculative_tokens]
//   - draft_probs:
//       * selected-only [batch_size, n_speculative_tokens], if
//         enable_opt_validate_probs=true
//       * recovered-dense [batch_size, n_speculative_tokens, vocab_size], if
//         enable_opt_validate_probs=false
//       * undefined, if draft_probs_required=false
std::pair<torch::Tensor, torch::Tensor> build_validate_tensors(
    const std::vector<torch::Tensor>& draft_token_ids_steps,
    const std::vector<torch::Tensor>& draft_probs_steps,
    int32_t batch_size,
    int32_t vocab_size,
    bool enable_opt_validate_probs,
    bool draft_probs_required = true);

// Build validate inputs from an already-stacked draft block. Block-diffusion
// drafters (DFlash) emit the whole block in one forward, so there are no
// per-step tensors to assemble — this avoids the per-step select/view/cat
// round trip of build_validate_tensors.
//   token_ids_block / probs_block: [batch_size, n_speculative_tokens]
// Returns draft_token_ids [batch, n_spec] (int64). When
// enable_opt_validate_probs is true the selected-only probs_block is returned
// as-is [batch, n_spec]; otherwise the selected probs are scattered into a
// dense [batch, n_spec, vocab_size] tensor matching MTP's
// build_validate_tensors output, which the default path and the fused
// rejection kernel require.
std::pair<torch::Tensor, torch::Tensor> build_validate_tensors_from_block(
    const torch::Tensor& token_ids_block,
    const torch::Tensor& probs_block,
    int32_t vocab_size,
    bool enable_opt_validate_probs);

}  // namespace draftProbs

}  // namespace specBuilder

}  // namespace xllm
