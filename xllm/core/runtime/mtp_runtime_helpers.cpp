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

#include "core/runtime/mtp_runtime_helpers.h"

#include "core/kv_cache/layout/seq_lens_layout.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/vlm_forward_params.h"
#include "core/util/slice.h"
#include "core/util/tensor_helper.h"

namespace xllm::mtp_detail {
void check_mtp_decode_states(
    const std::vector<EmbeddingCache::DecodeState>& states,
    const std::vector<std::string>& request_ids,
    const torch::Tensor& token_ids_host,
    bool allow_overlap_fake_token) {
  CHECK(!request_ids.empty())
      << "MTP decode requires request ids for bootstrap state validation";
  CHECK_EQ(states.size(), request_ids.size())
      << "MTP decode request/state count mismatch";
  CHECK_GE(token_ids_host.numel(), static_cast<int64_t>(states.size()))
      << "MTP decode token/state count mismatch";

  Slice<int32_t> token_ids = tensor_slice(token_ids_host);
  for (int32_t i = 0; i < static_cast<int32_t>(states.size()); ++i) {
    const EmbeddingCache::DecodeState& state = states[i];
    const int32_t token_id = token_ids[i];
    CHECK(state.valid) << "MTP decode missing target state, request_id="
                       << request_ids[i];
    CHECK_EQ(state.request_id, request_ids[i])
        << "MTP decode target state request mismatch";
    CHECK(state.embedding.defined())
        << "MTP decode target state embedding is undefined, request_id="
        << request_ids[i];
    if (token_id < 0) {
      CHECK(allow_overlap_fake_token)
          << "MTP decode fake token is only allowed with schedule overlap, "
          << "request_id=" << request_ids[i];
      CHECK_GE(state.token_id, 0)
          << "MTP decode fake token requires a valid cached target token, "
          << "request_id=" << request_ids[i];
      continue;
    }
    CHECK_EQ(state.token_id, token_id)
        << "MTP decode target state token mismatch, request_id="
        << request_ids[i];
  }
}

MtpTopkStatePtr select_mtp_topk_state_for_next_step(
    const MtpTopkStatePtr& state,
    const SamplingParameters& sampling_params) {
  if (state == nullptr) {
    return state;
  }
  const int64_t num_seqs = sampling_params.num_sample_seqs();
  if (num_seqs == 0 || state->num_rows() == num_seqs) {
    return state;
  }
  const torch::Tensor& selected_idxes = sampling_params.selected_token_idxes();
  if (selected_idxes.device().is_cpu()) {
    CHECK_GT(state->num_rows(), selected_idxes.max().item<int64_t>())
        << "MTP selected top-k index exceeds top-k rows.";
  }
  torch::Tensor index =
      selected_idxes.to(torch::dtype(torch::kLong).device(state->device()))
          .contiguous();
  return state->index_select_rows(index);
}

namespace {

// Clamps the position/kv-length pair to the allocated block-table capacity
// and normalizes scheduler KV-length lag.
void clamp_chunked_verify_decode_step(int32_t allocated_kv_len,
                                      int32_t verify_lens,
                                      int64_t block_table_rows,
                                      int32_t seq_id,
                                      int32_t& current_position,
                                      int32_t& current_kv_len,
                                      int32_t& expected_kv_len) {
  if (seq_id >= block_table_rows) {
    return;
  }
  const int32_t max_valid_position = allocated_kv_len - verify_lens;
  if (current_position > max_valid_position) {
    CHECK_GE(max_valid_position, 0)
        << "decode context block table is too small for validation, "
        << "seq_id=" << seq_id << ", allocated_kv_len=" << allocated_kv_len
        << ", verify_lens=" << verify_lens;
    CHECK_LE(current_position - max_valid_position, verify_lens)
        << "decode context position exceeds allocated blocks, seq_id=" << seq_id
        << ", current_position=" << current_position
        << ", current_kv_len=" << current_kv_len
        << ", allocated_kv_len=" << allocated_kv_len;
    current_position = max_valid_position;
    expected_kv_len = current_position + 1;
    current_kv_len = std::min(current_kv_len, expected_kv_len);
  }
  if (current_kv_len < expected_kv_len) {
    CHECK_LE(expected_kv_len - current_kv_len, verify_lens)
        << "decode context kv_len lag is too large, seq_id=" << seq_id
        << ", current_position=" << current_position
        << ", current_kv_len=" << current_kv_len;
    current_kv_len = expected_kv_len;
  }
  if (current_kv_len > expected_kv_len) {
    // First decode step: the prompt KV length with a zero-initialized
    // position; align the position to the KV context.
    current_position = current_kv_len - 1;
    expected_kv_len = current_kv_len;
  }
}

}  // namespace

ExpectedKvPolicy derive_expected_kv_policy(bool clamped_verify,
                                           bool positions_decoupled,
                                           int32_t verify_lens,
                                           int64_t logical_block_size) {
  if (clamped_verify) {
    CHECK_GT(verify_lens, 0);
    CHECK_GT(logical_block_size, 0);
    return {.kind = ExpectedKvPolicy::Kind::kChunkedVerifyClamped,
            .verify_lens = verify_lens,
            .logical_block_size = static_cast<int32_t>(logical_block_size)};
  }
  if (positions_decoupled) {
    return {.kind = ExpectedKvPolicy::Kind::kDecoupledPassThrough};
  }
  return {.kind = ExpectedKvPolicy::Kind::kPositionCoupled};
}

template <typename Input>
void update_decode_step_input(
    Input& input,
    const std::vector<EmbeddingCache::DecodeState>& last_states,
    bool enable_cache_correction,
    const ExpectedKvPolicy& kv_policy) {
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  CHECK_EQ(last_states.size(), static_cast<size_t>(num_sequences))
      << "decode context state count mismatch";

  std::vector<int32_t> token_ids_vec;
  std::vector<int32_t> positions_vec;
  std::vector<int32_t> kv_seq_lens_vec;
  token_ids_vec.reserve(num_sequences);
  positions_vec.reserve(num_sequences);
  kv_seq_lens_vec.reserve(
      CuSeqLens::capacity_for(static_cast<size_t>(num_sequences)));

  const Slice<int32_t> input_token_ids = tensor_slice(input.token_ids_host);
  const Slice<int32_t> input_positions = tensor_slice(input.positions_host);
  const torch::Tensor& block_tables =
      input.input_params.attention.host.block_tables;
  const bool clamp_active =
      kv_policy.kind == ExpectedKvPolicy::Kind::kChunkedVerifyClamped &&
      block_tables.defined() && block_tables.dim() == 2;
  const int32_t allocated_kv_len =
      clamp_active ? static_cast<int32_t>(block_tables.size(1)) *
                         kv_policy.logical_block_size
                   : 0;

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    CHECK_LT(static_cast<size_t>(seq_id), input_token_ids.size());
    CHECK_LT(static_cast<size_t>(seq_id), input_positions.size());
    const EmbeddingCache::DecodeState& state = last_states[seq_id];
    const int32_t input_token_id = input_token_ids[seq_id];
    const bool input_is_fake_token = input_token_id < 0;
    const bool rewrite_fake_token =
        enable_cache_correction && input_is_fake_token;
    const bool use_cache_correction = rewrite_fake_token && state.valid;
    const int32_t position_offset =
        use_cache_correction ? state.position_offset : 0;
    int32_t current_position = input_positions[seq_id] + position_offset;
    int32_t current_kv_len =
        CuSeqLens::at(input.input_params.attention.host.kv_seq_lens, seq_id) +
        position_offset;
    int32_t expected_kv_len =
        kv_policy.kind == ExpectedKvPolicy::Kind::kDecoupledPassThrough
            ? current_kv_len
            : current_position + 1;
    if (clamp_active) {
      clamp_chunked_verify_decode_step(allocated_kv_len,
                                       kv_policy.verify_lens,
                                       block_tables.size(0),
                                       seq_id,
                                       current_position,
                                       current_kv_len,
                                       expected_kv_len);
    }

    CHECK_EQ(expected_kv_len, current_kv_len)
        << "decode context position/kv_len mismatch, seq_id=" << seq_id
        << ", current_position=" << current_position
        << ", current_kv_len=" << current_kv_len;

    token_ids_vec.emplace_back(rewrite_fake_token ? state.token_id
                                                  : input_token_id);
    positions_vec.emplace_back(current_position);
    CuSeqLens::append(kv_seq_lens_vec, current_kv_len);
  }

  input.token_ids_host = make_pinned_cpu_tensor(token_ids_vec);
  input.positions_host = make_pinned_cpu_tensor(positions_vec);
  input.input_params.attention.host.kv_seq_lens = std::move(kv_seq_lens_vec);
  input.runtime.device_tensors_ready = false;
}

template void update_decode_step_input<LlmForwardInput>(
    LlmForwardInput&,
    const std::vector<EmbeddingCache::DecodeState>&,
    bool,
    const ExpectedKvPolicy&);
template void update_decode_step_input<VlmForwardInput>(
    VlmForwardInput&,
    const std::vector<EmbeddingCache::DecodeState>&,
    bool,
    const ExpectedKvPolicy&);

}  // namespace xllm::mtp_detail
