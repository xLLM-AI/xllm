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

#include "rec_multi_round_batch_input_builder.h"

#include <c10/core/DeviceType.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "core/framework/config/beam_search_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/util/rec_model_utils.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/request/sequence.h"
#include "framework/request/sequences_group.h"
#include "framework/sampling/sampling_params.h"
#include "runtime/params_utils.h"
#include "util/blocking_counter.h"
#include "util/slice.h"
#include "util/tensor_helper.h"
#include "util/threadpool.h"
#include "util/utils.h"

namespace xllm {
namespace {

std::vector<int32_t> build_q_cu_seq_lens_vec(
    const std::vector<int32_t>& q_seq_lens) {
  std::vector<int32_t> q_cu_seq_lens;
  if (q_seq_lens.empty()) {
    return q_cu_seq_lens;
  }
#if defined(USE_NPU)
  q_cu_seq_lens.reserve(q_seq_lens.size());
  int32_t cum_seq_len = 0;
  for (int32_t q_len : q_seq_lens) {
    cum_seq_len += q_len;
    q_cu_seq_lens.emplace_back(cum_seq_len);
  }
#else
  CHECK(q_seq_lens.front() == 0)
      << "q_seq_lens must be cumulative with leading zero";
  q_cu_seq_lens.assign(q_seq_lens.begin() + 1, q_seq_lens.end());
#endif
  return q_cu_seq_lens;
}

}  // namespace

RecMultiRoundBatchInputBuilder::RecMultiRoundBatchInputBuilder(
    const std::vector<SequencesGroup*>& sequence_groups,
    const std::vector<uint32_t>& allowed_max_tokens,
    const std::vector<torch::Tensor>& input_embeddings_vec,
    const std::vector<MMData>& mm_data_vec,
    std::vector<BlockTransferInfo>* swap_block_transfer_infos,
    const uint64_t batch_id,
    const ModelArgs* args,
    BatchForwardType batch_forward_type,
    MPMCThreadPool* thread_pool)
    : allowed_max_tokens_(allowed_max_tokens),
      input_embeddings_vec_(input_embeddings_vec),
      mm_data_vec_(mm_data_vec),
      args_(args),
      batch_forward_type_(batch_forward_type),
      swap_block_transfer_infos_(swap_block_transfer_infos),
      thread_pool_(thread_pool),
      batch_id_(batch_id) {
  // Extract sequences from sequence_groups
  sequences_.clear();
  for (auto* seq_group : sequence_groups) {
    const auto& group_sequences = seq_group->sequences();
    for (const auto& seq_ptr : group_sequences) {
      sequences_.push_back(seq_ptr.get());
    }
  }

  num_sequences_ = static_cast<int32_t>(sequences_.size());
  CHECK_GT(num_sequences_, 0);

  if (args_ != nullptr) {
    use_mrope_ = (args_->rope_scaling_rope_type() == "mrope");
  }

  // Initialize RecMultiRound specific state
  rec_multi_round_state_.total_steps = get_rec_multi_round_decode_rounds();
  rec_multi_round_state_.base_state.batch_forward_type = batch_forward_type_;
}

void RecMultiRoundBatchInputBuilder::process_single_sequence(
    int32_t seq_index,
    BuilderState* state_ptr,
    std::unordered_set<int32_t>* write_block_ids_ptr) {
  RecMultiRoundBuilderState& state = rec_multi_round_state_;
  BuilderState& base_state = state.base_state;

  auto* sequence = sequences_[seq_index];
  const auto token_ids = sequence->tokens();
  const uint32_t n_tokens = token_ids.size();
  const uint32_t n_kv_cache_tokens = sequence->kv_state().kv_cache_tokens_num();

  // Validate and calculate sequence lengths
  CHECK(allowed_max_tokens_[seq_index] > 0);
  uint32_t q_seq_len =
      std::min(n_tokens - n_kv_cache_tokens, allowed_max_tokens_[seq_index]);
  uint32_t seq_len = q_seq_len + n_kv_cache_tokens;

  // add decode data;
  uint32_t decode_q_seq_len = 1;
  uint32_t decode_seq_len = n_kv_cache_tokens + 1;

  CHECK_GT(q_seq_len, 0) << "at least one token should be processed. "
                         << "n_tokens: " << n_tokens
                         << ", n_kv_cache_tokens: " << n_kv_cache_tokens
                         << ", current_max_tokens_capacity: "
                         << sequence->kv_state().current_max_tokens_capacity()
                         << ", allowed_max_tokens: "
                         << allowed_max_tokens_[seq_index];

  // Update state
  int32_t offset = is_mtp_decode_ ? -1 : 0;
  base_state.max_seq_len = std::max(base_state.max_seq_len, seq_len);
  base_state.q_max_seq_len = std::max(base_state.q_max_seq_len, q_seq_len);
#if defined(USE_NPU)
  base_state.seq_lens.push_back(seq_len);
  base_state.q_seq_lens.push_back(q_seq_len);
#elif defined(USE_MLU) || defined(USE_CUDA) || defined(USE_ILU) || \
    defined(USE_DCU)
  base_state.seq_lens.push_back(base_state.seq_lens.back() + seq_len);
  base_state.q_seq_lens.push_back(base_state.q_seq_lens.back() + q_seq_len);
#endif

  // Call our enhanced method to process tokens and positions
  // This handles both regular decode and step-level decode cases
  extract_tokens_and_positions(sequence, n_kv_cache_tokens, seq_len, &state);

  // not Setup KV cache for prefill

  setup_kv_cache_info(sequence,
                      n_kv_cache_tokens,
                      decode_seq_len,
                      decode_q_seq_len,
                      &base_state,
                      write_block_ids_ptr);

  // Track prefill sequences
  if (sequence->stage() == SequenceStage::PREFILL) {
    base_state.prefill_seq_len++;
  }
}

ForwardInput RecMultiRoundBatchInputBuilder::build_rec_forward_input(
    uint32_t /*num_decoding_tokens*/,
    uint32_t /*min_decoding_batch_size*/) {
  // Rec multi-round mode doesn't use num_decoding_tokens and
  // min_decoding_batch_size parameters, so we ignore them and call the internal
  // build_forward_input method.
  return build_forward_input();
}

ForwardInput RecMultiRoundBatchInputBuilder::build_forward_input() {
  // Reset Rec multi-round state for this build.
  rec_multi_round_state_.total_steps = get_rec_multi_round_decode_rounds();

  is_mtp_decode_ = false;
  // Single-threaded processing for now; can be extended to use thread_pool_
  for (int32_t i = 0; i < static_cast<int32_t>(sequences_.size()); ++i) {
    process_single_sequence(i, &rec_multi_round_state_.base_state, nullptr);
  }
  return state_to_forward_input();
}

void RecMultiRoundBatchInputBuilder::extract_tokens_and_positions(
    Sequence* sequence,
    uint32_t n_kv_cache_tokens,
    uint32_t seq_len,
    RecMultiRoundBuilderState* state_ptr) {
  // First build the "base" view that matches the single-round builder
  BuilderState& base_state = state_ptr->base_state;

  const auto& token_ids = sequence->tokens();
  const uint32_t n_tokens = token_ids.size();

  // Prepare adjusted token counts for sampling
  std::unordered_map<int32_t, int32_t> adjusted_token_to_count_map;
  for (uint32_t j = n_kv_cache_tokens; j < seq_len; ++j) {
    // skip prompt tokens except the last one
    if (j + 1 < n_tokens) continue;
    ++adjusted_token_to_count_map[token_ids[j]];
  }

  // Process each token
  for (uint32_t j = n_kv_cache_tokens; j < seq_len; ++j) {
    base_state.flatten_tokens_vec.push_back(token_ids[j]);

    if (!use_mrope_) {
      base_state.flatten_positions_vec.push_back(static_cast<int32_t>(j));
    }

    // Handle sampling for last tokens
    if (j + 1 < n_tokens) continue;

    // Inlined sampling/unique-token logic on base_state.
    BuilderState& state = base_state;

    const auto token_id = sequence->tokens()[j];
    // Adjust token count
    --adjusted_token_to_count_map[token_id];

    // Select token for sampling
    state.selected_token_idxes.push_back(
        static_cast<int32_t>(state.flatten_tokens_vec.size() - 1));
    state.sampling_params.push_back(sequence->sampling_param());

    // Process unique tokens
    const auto& seq_token_counts = sequence->token_to_count_map();
    auto& ids = state.unique_token_ids_vec.emplace_back();
    auto& counts = state.unique_token_counts_vec.emplace_back();

    ids.reserve(seq_token_counts.size());
    counts.reserve(seq_token_counts.size());

    for (const auto& [tok_id, count] : seq_token_counts) {
      const auto it = adjusted_token_to_count_map.find(tok_id);
      const auto adjust_count =
          (it != adjusted_token_to_count_map.end()) ? it->second : 0;

      if (count > adjust_count) {
        ids.push_back(tok_id);
        counts.push_back(count - adjust_count);
      }
    }

    state.unique_token_lens_vec.push_back(static_cast<int32_t>(ids.size()));

    // Mark sample token if it's the last token
    if (j == seq_len - 1) {
      state.sample_idxes.push_back(
          static_cast<int32_t>(state.selected_token_idxes.size() - 1));
    }
  }

  // `linear_state_ids` is consumed per sequence, so preserve one entry per
  // logical batch row even when this slice is not the last prefill chunk. The
  // slot is drawn from the dedicated LinearStateBlockManager and is -1 when
  // linear state is disabled.
  base_state.linear_state_ids.push_back(sequence->get_linear_state_slot_id());

  // Add extra token id
  if (n_tokens == seq_len) {
    // last chunk of prefill and decode
    // add -1 as extra token id
    base_state.extra_token_ids.push_back(-1);
    base_state.embedding_ids.push_back(sequence->get_single_block_id());
  } else {
    base_state.extra_token_ids.push_back(token_ids[seq_len]);
  }

  // begin process decode data
  seq_len = n_kv_cache_tokens + 1;
  uint32_t prompt_len = sequence->num_prompt_tokens();
  state_ptr->decode_positions_vec.push_back(static_cast<int32_t>(prompt_len));

  int32_t bw =
      std::max(1, ::xllm::BeamSearchConfig::get_instance().beam_width());
  const int32_t sel_start =
      static_cast<int32_t>(state_ptr->decode_selected_token_idxes.size());
  state_ptr->decode_selected_token_idxes.reserve(sel_start + bw);
  state_ptr->decode_sample_idxes.reserve(state_ptr->decode_sample_idxes.size() +
                                         bw);
  state_ptr->decode_unique_token_ids_vec.resize(
      state_ptr->decode_unique_token_ids_vec.size() + bw);
  state_ptr->decode_unique_token_counts_vec.resize(
      state_ptr->decode_unique_token_counts_vec.size() + bw);
  state_ptr->decode_unique_token_lens_vec.insert(
      state_ptr->decode_unique_token_lens_vec.end(), bw, 0);
  state_ptr->decode_sampling_params.reserve(
      state_ptr->decode_sampling_params.size() + bw);
  for (int32_t i = 0; i < bw; ++i) {
    const int32_t idx = sel_start + i;
    state_ptr->decode_selected_token_idxes.push_back(idx);
    state_ptr->decode_sample_idxes.push_back(idx);
    state_ptr->decode_sampling_params.push_back(sequence->sampling_param());
  }
}

void RecMultiRoundBatchInputBuilder::setup_kv_cache_info(
    Sequence* sequence,
    uint32_t n_kv_cache_tokens,
    uint32_t seq_len,
    uint32_t q_seq_len,
    BuilderState* state_ptr,
    std::unordered_set<int32_t>* write_block_ids_ptr) {
#if defined(USE_NPU)
  (void)write_block_ids_ptr;
  BuilderState& state = *state_ptr;
  const auto blocks = sequence->kv_state().blocks(BlockType::KV);
  std::vector<int32_t> block_ids;
  block_ids.reserve(blocks.size());
  for (const auto& block : blocks) {
    block_ids.push_back(block.id());
  }
  state.block_tables_vec.emplace_back(std::move(block_ids));
#elif defined(USE_MLU) || defined(USE_CUDA) || defined(USE_ILU) || \
    defined(USE_MUSA) || defined(USE_DCU)
  // TODO: refactor this branch when NPU multi-round xattention lands.
  RecMultiRoundBuilderState& state = rec_multi_round_state_;
  BuilderState& base_state = state.base_state;

  // fill paged_kv_last_page_len for batch_size calculation.
  base_state.paged_kv_last_page_len.push_back(seq_len);
  base_state.block_tables_vec.emplace_back(std::vector<int32_t>{});
#endif
}

ForwardInput RecMultiRoundBatchInputBuilder::state_to_forward_input() {
  BuilderState& state = rec_multi_round_state_.base_state;
  if (state.flatten_tokens_vec.empty()) {
    return {};
  }

  ForwardInput forward_input;

  // Create tensors (same as BatchInputBuilder)
  forward_input.token_ids =
      torch::tensor(state.flatten_tokens_vec, torch::kInt);
  forward_input.token_ids_host = forward_input.token_ids;

  if (!use_mrope_) {
    forward_input.positions =
        torch::tensor(state.flatten_positions_vec, torch::kInt);
  } else {
    forward_input.positions = torch::cat(state.mrope_positions_vec, 1);
  }
  forward_input.positions_host = forward_input.positions;

  auto& input_params = forward_input.input_params;
  input_params.meta.batch_forward_type = state.batch_forward_type;
  input_params.meta.num_sequences = state.block_tables_vec.size();
  input_params.meta.kv_max_seq_len = state.max_seq_len;
  input_params.meta.q_max_seq_len = state.q_max_seq_len;
  input_params.attention.device.kv_seq_lens =
      torch::tensor(state.seq_lens, torch::kInt);
  input_params.attention.device.q_seq_lens =
      torch::tensor(state.q_seq_lens, torch::kInt);
  std::vector<int32_t> q_cu_seq_lens =
      build_q_cu_seq_lens_vec(state.q_seq_lens);
  input_params.attention.device.q_cu_seq_lens =
      torch::tensor(q_cu_seq_lens, torch::kInt);
  input_params.attention.host.kv_seq_lens = std::move(state.seq_lens);
  input_params.attention.host.q_cu_seq_lens = std::move(q_cu_seq_lens);
  input_params.attention.host.q_seq_lens = std::move(state.q_seq_lens);
  input_params.attention.device.new_cache_slots =
      torch::tensor(state.new_token_slot_ids, torch::kInt);

  // for flashinfer
  input_params.attention.device.paged_kv_indptr =
      torch::tensor(state.paged_kv_indptr, torch::kInt);
  input_params.attention.device.paged_kv_indices =
      torch::tensor(state.paged_kv_indices, torch::kInt);
  input_params.attention.device.paged_kv_last_page_len =
      torch::tensor(state.paged_kv_last_page_len, torch::kInt);

  // Setup multimodal data
  input_params.multimodal.mm_data.batch(mm_data_vec_);

  // Setup block tables
  util::pad_2d_vector(state.block_tables_vec, /*pad_value=*/0);
  input_params.attention.device.block_tables =
      create_2d_tensor(state.block_tables_vec, torch::kInt);
  input_params.attention.host.block_tables =
      input_params.attention.device.block_tables;

  if (input_embeddings_vec_.size() != 0) {
    input_params.embedding.input_embedding = torch::cat(input_embeddings_vec_);
  }
  input_params.embedding.embedding_ids = std::move(state.embedding_ids);
  input_params.embedding.linear_state_ids = std::move(state.linear_state_ids);
  if (!input_params.embedding.linear_state_ids.empty()) {
    input_params.embedding.linear_state_indices =
        torch::tensor(input_params.embedding.linear_state_ids, torch::kInt);
  }
  input_params.embedding.extra_token_ids = std::move(state.extra_token_ids);

  if (swap_block_transfer_infos_ != nullptr &&
      swap_block_transfer_infos_->size() > 0) {
    input_params.block_copy.swap_blocks.insert(
        input_params.block_copy.swap_blocks.end(),
        swap_block_transfer_infos_->begin(),
        swap_block_transfer_infos_->end());
  }

  CHECK_EQ(state.sampling_params.size(), state.selected_token_idxes.size());
  // Setup sampling parameters
  if (!state.selected_token_idxes.empty()) {
    util::pad_2d_vector<int64_t>(state.unique_token_ids_vec, /*pad_value=*/0);
    util::pad_2d_vector(state.unique_token_counts_vec, /*pad_value=*/0);

    forward_input.sampling_params.init(state.sampling_params,
                                       state.selected_token_idxes,
                                       state.sample_idxes,
                                       state.unique_token_ids_vec,
                                       state.unique_token_counts_vec,
                                       state.unique_token_lens_vec);
  }

  // Rec multi-round specific metadata.
  rec_multi_round_state_.total_steps = get_rec_multi_round_decode_rounds();
  const int32_t beam_width =
      ::xllm::BeamSearchConfig::get_instance().beam_width();
  const int32_t total_round = rec_multi_round_state_.total_steps;
  const int32_t current_round = 0;
  std::vector<int64_t> full_kv_shape;
  std::vector<int32_t> decode_positions_vec;

  // Setup decoder sampling parameters for Rec multi-round decode.
  if (!rec_multi_round_state_.decode_selected_token_idxes.empty()) {
    CHECK_EQ(rec_multi_round_state_.decode_sampling_params.size(),
             rec_multi_round_state_.decode_selected_token_idxes.size());
    util::pad_2d_vector<int64_t>(
        rec_multi_round_state_.decode_unique_token_ids_vec,
        /*pad_value=*/0);
    util::pad_2d_vector(rec_multi_round_state_.decode_unique_token_counts_vec,
                        /*pad_value=*/0);

    forward_input.decoder_sampling_params.init(
        rec_multi_round_state_.decode_sampling_params,
        rec_multi_round_state_.decode_selected_token_idxes,
        rec_multi_round_state_.decode_sample_idxes,
        rec_multi_round_state_.decode_unique_token_ids_vec,
        rec_multi_round_state_.decode_unique_token_counts_vec,
        rec_multi_round_state_.decode_unique_token_lens_vec);
  }

  // Set full_kv_shape if we have Rec multi-round decode data.
  if (is_rec_multi_round_mode() && !sequences_.empty()) {
    int64_t batch_size = static_cast<int64_t>(sequences_.size());
    int64_t n_kv_heads =
        args_ ? args_->n_kv_heads().value_or(args_->n_heads()) : 0;
    int64_t head_dim = args_ ? args_->head_dim() : 0;

    int32_t decode_rounds = get_rec_multi_round_decode_rounds();
    full_kv_shape = {
        ::xllm::SchedulerConfig::get_instance().max_tokens_per_batch() +
            ::xllm::SchedulerConfig::get_instance().max_seqs_per_batch() *
                ::xllm::BeamSearchConfig::get_instance().beam_width() *
                std::max(0, decode_rounds - 1),
        n_kv_heads,
        head_dim};
  }

  // Decode positions
  if (!rec_multi_round_state_.decode_positions_vec.empty()) {
    decode_positions_vec = rec_multi_round_state_.decode_positions_vec;
  }

  if (is_rec_multi_round_mode()) {
    StepDecodeMeta step_meta;
    step_meta.batch_size = static_cast<int32_t>(sequences_.size());
    step_meta.beam_width = beam_width;
    step_meta.current_round = current_round;
    step_meta.total_round = total_round;
    step_meta.full_kv_shape = std::move(full_kv_shape);
    step_meta.decode_positions_vec = std::move(decode_positions_vec);
    forward_input.step_decode = std::move(step_meta);
  }

  // Batch ID
  input_params.meta.batch_id = batch_id_;

  return forward_input;
}

}  // namespace xllm
