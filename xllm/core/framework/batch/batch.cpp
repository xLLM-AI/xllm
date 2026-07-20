/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "batch.h"

#include <c10/core/DeviceType.h>
#include <torch/torch.h>

#include <algorithm>
#include <vector>

#include "batch_input_builder.h"
#include "common/global_flags.h"
#include "common/metrics.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/util/rec_model_utils.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/request/sequence.h"
#include "framework/sampling/sampling_params.h"
#include "rec_batch_input_builder.h"
#include "runtime/params_utils.h"
#include "util/slice.h"
#include "util/tensor_helper.h"
#include "util/utils.h"

namespace xllm {
namespace {

uint32_t get_sample_source_position(const SampleSlot& sample_slot) {
  if (sample_slot.token_position == 0) {
    return 0;
  }
  return static_cast<uint32_t>(sample_slot.token_position - 1);
}

Token make_token(const RawToken& raw_token) {
  Token token(raw_token.id);
  if (raw_token.logprob.has_value()) {
    token.logprob = raw_token.logprob.value();
  }
  token.top_tokens = raw_token.top_tokens;
  token.top_logprobs = raw_token.top_logprobs;
  return token;
}

Token make_empty_logprob_placeholder(const Sequence& seq) {
  const auto prompt_tokens = seq.tokens();
  const int64_t placeholder_token_id =
      prompt_tokens.empty() ? 0 : prompt_tokens[0];
  return Token(placeholder_token_id);
}

}  // namespace

Batch::Batch(Sequence* sequence) { add(sequence); }
Batch::Batch(const std::vector<Sequence*>& sequences) { add(sequences); }

void Batch::add(Sequence* sequence, uint32_t allowed_max_token) {
  CHECK(sequence != nullptr);
  CHECK(!sequence->finished());
  CHECK_GT(allowed_max_token, 0);

  sequences_.push_back(sequence);
  allowed_max_tokens_.push_back(allowed_max_token);

  const auto& input_embedding = sequence->get_input_embedding();
  if (input_embedding.defined())
    input_embeddings_vec_.emplace_back(input_embedding);

  update_forward_type(sequence);
}

void Batch::update_forward_type(Sequence* sequence) {
  auto stage = sequence->stage();
  switch (batch_forward_type_.value()) {
    case BatchForwardType::PREFILL:
      if (stage == SequenceStage::CHUNKED_PREFILL) {
        batch_forward_type_ = BatchForwardType::CHUNKED_PREFILL;
      } else if (stage == SequenceStage::DECODE) {
        batch_forward_type_ = BatchForwardType::MIXED;
      }
      break;
    case BatchForwardType::CHUNKED_PREFILL:
      if (stage == SequenceStage::DECODE) {
        batch_forward_type_ = BatchForwardType::MIXED;
      }
      break;
    case BatchForwardType::DECODE:
      if (stage != SequenceStage::DECODE) {
        batch_forward_type_ = BatchForwardType::MIXED;
      }
      break;
    case BatchForwardType::MIXED:
      break;
    case BatchForwardType::EMPTY:
      batch_forward_type_ = BatchForwardType(static_cast<int32_t>(stage));
      break;
  }
}

void Batch::refresh_forward_type() {
  batch_forward_type_ = BatchForwardType();
  const auto sequences = get_sequences();
  for (auto* sequence : sequences) {
    update_forward_type(sequence);
  }
}

void Batch::add(const std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    add(sequence);
  }
}

ForwardInput Batch::prepare_forward_input(uint32_t num_decoding_tokens,
                                          uint32_t min_decoding_batch_size,
                                          const ModelArgs& args,
                                          int32_t cp_size) {
  if (sequences_.empty() && !sequence_groups_.empty()) {
    output_targets_.clear();
    return prepare_rec_forward_input(
        num_decoding_tokens, min_decoding_batch_size, args);
  }
  refresh_output_targets();
  BatchInputBuilder builder(sequences_,
                            allowed_max_tokens_,
                            input_embeddings_vec_,
                            mm_data_vec_,
                            &swap_block_transfer_infos_,
                            batch_id_,
                            &args,
                            batch_forward_type_,
                            cp_size);
  return builder.build_forward_input(num_decoding_tokens,
                                     min_decoding_batch_size);
}

ForwardInput Batch::prepare_rec_forward_input(uint32_t num_decoding_tokens,
                                              uint32_t min_decoding_batch_size,
                                              const ModelArgs& args,
                                              MPMCThreadPool* thread_pool) {
  RecType rec_type = RecType::kNone;
  if (!sequence_groups_.empty() && !sequence_groups_[0]->sequences().empty()) {
    rec_type = sequence_groups_[0]->sequences()[0]->rec_type();
  }
  output_targets_.clear();
  if (rec_type == RecType::kOneRec) {
    if (!sequence_groups_.empty()) {
      // OneRec REC batches are tracked via sequence_groups_, while output
      // target generation still walks sequences_. Refresh the flattened
      // sequence view on every step so token writeback stays aligned after
      // beam search expands or replaces the group-owned Sequence instances.
      refresh_sequences_from_groups();
    }
    if (use_legacy_onerec_prefill_only_contract()) {
      refresh_onerec_prefill_output_targets();
    } else {
      refresh_output_targets();
    }
  }

  auto builder = RecBatchInputBuilder::create(rec_type,
                                              sequence_groups_,
                                              allowed_max_tokens_,
                                              input_embeddings_vec_,
                                              mm_data_vec_,
                                              &swap_block_transfer_infos_,
                                              batch_id_,
                                              &args,
                                              batch_forward_type_,
                                              thread_pool);
  return builder->build_rec_forward_input(num_decoding_tokens,
                                          min_decoding_batch_size);
}

std::vector<Sequence*> Batch::get_sequences() {
  if (!sequences_.empty()) {
    return sequences_;
  }

  std::vector<Sequence*> result;
  for (auto* seq_group : sequence_groups_) {
    const auto& sequences = seq_group->sequences();
    for (const auto& seq_ptr : sequences) {
      result.push_back(seq_ptr.get());
    }
  }
  return result;
}

bool Batch::has_partial_finished_beam_group() const {
  if (sequence_groups_.empty()) {
    return false;
  }

  for (auto* seq_group : sequence_groups_) {
    if (!seq_group->check_beam_search()) {
      continue;
    }

    const auto& sequences = seq_group->sequences();
    if (sequences.empty()) {
      continue;
    }

    const size_t finished_cnt = static_cast<size_t>(
        std::count_if(sequences.begin(), sequences.end(), [](const auto& seq) {
          return seq->finished();
        }));
    if (finished_cnt > 0 && finished_cnt < sequences.size()) {
      return true;
    }
  }
  return false;
}

void Batch::refresh_sequences_from_groups() {
  if (sequence_groups_.empty()) {
    return;
  }
  sequences_.clear();
  allowed_max_tokens_.clear();
  for (auto* seq_group : sequence_groups_) {
    const auto& sequences = seq_group->sequences();
    for (const auto& seq_ptr : sequences) {
      sequences_.push_back(seq_ptr.get());
      // Use max value as default budget for beam search sequences
      allowed_max_tokens_.push_back(std::numeric_limits<uint32_t>::max());
    }
  }
}

void Batch::dp_balance_shuffle_seqs() {
#if defined(USE_NPU)
  // this shuffle operation is mainly used for npu with 24 cores
  // and specific mla op implementation
  const auto num_npu_cores = 24;  // npu cube core num
  if (::xllm::KernelConfig::get_instance().enable_customize_mla_kernel() &&
      ::xllm::ParallelConfig::get_instance().enable_dp_balance() &&
      sequences_.size() > num_npu_cores) {
    std::vector<uint32_t> kv_cache_tokens_num;
    kv_cache_tokens_num.reserve(sequences_.size());
    for (auto& seq : sequences_) {
      kv_cache_tokens_num.push_back(seq->kv_state().kv_cache_tokens_num());
    }
    auto seq_index_shift = cal_seq_exchange_index(kv_cache_tokens_num);

    std::vector<Sequence*> balanced_sequences;
    std::vector<uint32_t> balanced_allowed_max_tokens;
    balanced_sequences.resize(sequences_.size());
    balanced_allowed_max_tokens.resize(allowed_max_tokens_.size());
    for (auto& ele : seq_index_shift) {
      balanced_sequences[ele.second] = sequences_[ele.first];
      balanced_allowed_max_tokens[ele.second] = allowed_max_tokens_[ele.first];
    }

    CHECK_EQ(sequences_.size(), balanced_sequences.size());
    sequences_ = std::move(balanced_sequences);
    allowed_max_tokens_ = std::move(balanced_allowed_max_tokens);
  }
#else
  // TODO: implement dp_balance_shuffle_seqs for non-npu devices
  static bool warning = true;
  if (warning) {
    LOG(WARNING)
        << "dp_balance_shuffle_seqs is not implemented for current device";
    warning = false;
  }
#endif
}

std::unordered_map<uint32_t, uint32_t> Batch::cal_seq_exchange_index(
    std::vector<uint32_t>& kv_cache_tokens_num) {
  const auto num_npu_cores = 24;  // npu cube core num
  const auto num_seqs = kv_cache_tokens_num.size();
  const auto base_per_core = num_seqs / num_npu_cores;
  const auto remainder = num_seqs % num_npu_cores;

  // find the indices of the remainder biggest elements
  std::vector<uint32_t> indices(num_seqs);
  std::iota(indices.begin(), indices.end(), 0);
  if (remainder > 0) {
    std::nth_element(indices.begin(),
                     indices.end() - remainder,
                     indices.end(),
                     [&kv_cache_tokens_num](uint32_t a, uint32_t b) {
                       return kv_cache_tokens_num[a] < kv_cache_tokens_num[b];
                     });
  }

  std::vector<uint32_t> base_indices(indices.begin(),
                                     indices.end() - remainder);
  std::vector<uint32_t> remainder_indices(indices.end() - remainder,
                                          indices.end());

  // sort base_indices in descending order
  std::sort(base_indices.begin(),
            base_indices.end(),
            [&kv_cache_tokens_num](uint32_t a, uint32_t b) {
              return kv_cache_tokens_num[a] > kv_cache_tokens_num[b];
            });

  // allocate a long and a short request to each core, to ensuring
  // load balance among all cores
  std::vector<std::vector<uint32_t>> base_assignment(
      num_npu_cores, std::vector<uint32_t>(base_per_core));
  for (auto i = 0; i < base_indices.size(); ++i) {
    auto col = i / num_npu_cores;
    auto row = (col % 2 == 0) ? (i % num_npu_cores)
                              : (num_npu_cores - 1 - (i % num_npu_cores));
    base_assignment[row][col] = base_indices[i];
  }

  // record the index map, first one is original index,
  // second one is the target index to be exchanged to
  std::unordered_map<uint32_t, uint32_t> index_shift;
  // add base part data
  for (auto i = 0; i < num_npu_cores; ++i) {
    for (auto j = 0; j < base_per_core; ++j) {
      auto idx = base_assignment[i][j];
      index_shift[idx] = i + j * num_npu_cores;
    }
  }
  // add remainder part data
  for (auto i = 0; i < remainder; ++i) {
    index_shift[remainder_indices[i]] = i + num_npu_cores * base_per_core;
  }

  return index_shift;
}

ForwardInput Batch::prepare_forward_input(const ModelArgs& args,
                                          ThreadPool* thread_pool,
                                          int32_t cp_size) {
  dp_balance_shuffle_seqs();
  refresh_output_targets();
  BatchInputBuilder builder(sequences_,
                            allowed_max_tokens_,
                            input_embeddings_vec_,
                            mm_data_vec_,
                            &swap_block_transfer_infos_,
                            batch_id_,
                            &args,
                            batch_forward_type_,
                            cp_size,
                            thread_pool);
  ForwardInput forward_input =
      builder.build_forward_input(/*num_decoding_tokens=*/0,
                                  /*min_decoding_batch_size=*/0);
  if (has_partial_finished_beam_group()) {
    // Beam-search kernel assumes fixed beam width per group. When only part of
    // a group is active, fall back to software beam merge.
    forward_input.sampling_params.acc_logprob = torch::Tensor();
  }
  return forward_input;
}

void Batch::refresh_output_targets() {
  output_targets_.clear();
  if (sequences_.empty()) {
    return;
  }

  for (size_t seq_index = 0; seq_index < sequences_.size(); ++seq_index) {
    auto* sequence = sequences_[seq_index];
    if (sequence == nullptr) {
      continue;
    }

    const auto token_ids = sequence->tokens();
    const uint32_t n_tokens = token_ids.size();
    const uint32_t n_kv_cache_tokens =
        sequence->kv_state().kv_cache_tokens_num();
    if (n_tokens <= n_kv_cache_tokens) {
      continue;
    }

    CHECK(allowed_max_tokens_[seq_index] > 0);
    const uint32_t q_seq_len =
        std::min(n_tokens - n_kv_cache_tokens, allowed_max_tokens_[seq_index]);
    const uint32_t seq_len = q_seq_len + n_kv_cache_tokens;
    const auto& sample_slots = sequence->sample_slots();

    if (sample_slots.empty()) {
      if (seq_len == n_tokens) {
        output_targets_.push_back({sequence, /*sample_id=*/0, false});
      }
      continue;
    }

    for (const auto& sample_slot : sample_slots) {
      const uint32_t sample_source_position =
          get_sample_source_position(sample_slot);
      if (sample_source_position < n_kv_cache_tokens ||
          sample_source_position >= seq_len) {
        continue;
      }
      output_targets_.push_back(
          {sequence, sample_slot.sample_id, /*from_sample_slot=*/true});
    }
  }
}

void Batch::refresh_onerec_prefill_output_targets() {
  output_targets_.clear();
  if (sequences_.empty()) {
    return;
  }

  for (size_t seq_index = 0; seq_index < sequences_.size(); ++seq_index) {
    auto* sequence = sequences_[seq_index];
    if (sequence == nullptr) {
      continue;
    }

    const auto token_ids = sequence->tokens();
    const uint32_t n_tokens = token_ids.size();
    const uint32_t n_kv_cache_tokens =
        sequence->kv_state().kv_cache_tokens_num();
    const bool needs_context_target = sequence->is_onerec_model() &&
                                      n_tokens == 0 && n_kv_cache_tokens == 0 &&
                                      sequence->num_decoder_embeddings() > 0;
    if (needs_context_target) {
      output_targets_.push_back({sequence, /*sample_id=*/0, false});
      continue;
    }
    if (n_tokens <= n_kv_cache_tokens) {
      continue;
    }

    CHECK(allowed_max_tokens_[seq_index] > 0);
    const uint32_t q_seq_len =
        std::min(n_tokens - n_kv_cache_tokens, allowed_max_tokens_[seq_index]);
    const uint32_t seq_len = q_seq_len + n_kv_cache_tokens;
    const auto& sample_slots = sequence->sample_slots();

    if (sample_slots.empty()) {
      if (seq_len == n_tokens) {
        output_targets_.push_back({sequence, /*sample_id=*/0, false});
      }
      continue;
    }

    for (const auto& sample_slot : sample_slots) {
      const uint32_t sample_source_position =
          get_sample_source_position(sample_slot);
      if (sample_source_position < n_kv_cache_tokens ||
          sample_source_position >= seq_len) {
        continue;
      }
      output_targets_.push_back(
          {sequence, sample_slot.sample_id, /*from_sample_slot=*/true});
    }
  }
}

void Batch::process_sample_output(const RawForwardOutput& raw_output,
                                  bool replace_fake_token) {
  for (size_t output_idx = 0; output_idx < output_targets_.size();
       ++output_idx) {
    const auto& target = output_targets_[output_idx];
    auto* seq = target.sequence;
    CHECK(seq != nullptr);

    if (output_idx < raw_output.outputs.size()) {
      const auto& seq_mm_embeddings =
          raw_output.outputs[output_idx].mm_embeddings;
      if (!seq_mm_embeddings.empty()) {
        seq->update_mm_embeddings(seq_mm_embeddings);
      }
    }

    if (!target.from_sample_slot) {
      if (seq->finished()) {
        continue;
      }
      if (update_sequence_state(seq, replace_fake_token)) {
        continue;
      }
    }

    const bool missing_output = output_idx >= raw_output.outputs.size();
    const bool empty_output =
        !missing_output && raw_output.outputs[output_idx].tokens.empty();
    if (missing_output || empty_output) {
      if (target.from_sample_slot) {
        append_token_for_sequence(
            seq, make_empty_logprob_placeholder(*seq), 0, replace_fake_token);
      }
      continue;
    }

    const auto& raw_sample_output = raw_output.outputs[output_idx];
    for (size_t token_idx = 0; token_idx < raw_sample_output.tokens.size();
         ++token_idx) {
      const auto& raw_token = raw_sample_output.tokens[token_idx];
      append_token_for_sequence(
          seq, make_token(raw_token), token_idx, replace_fake_token);

      if (!raw_token.embeddings.empty()) {
        torch::Tensor embeddings = torch::tensor(raw_token.embeddings);
        seq->update_embeddings(embeddings);
        seq->update_mtp_bootstrap_embedding(embeddings);
      }
      // Speculative decoding may append an EOS token at the beginning,
      // followed by bonus tokens, causing the sequence stopping check to fail.
      if (!target.from_sample_slot && seq->finished()) {
        break;
      }
    }
  }
  if (replace_fake_token) {
    output_targets_.clear();
  }

  if (!::xllm::SchedulerConfig::get_instance().enable_schedule_overlap() ||
      replace_fake_token) {
    process_beam_search();
  }
}

void Batch::process_beam_sequence_group(const ForwardOutput& output) {
  if (!output.beam_sequence_group.defined() ||
      output.beam_sequence_group.numel() == 0) {
    return;
  }

  // Get sequences from either sequences_ or sequence_groups_
  auto sequences = get_sequences();
  if (sequences.empty()) {
    return;
  }

  const int32_t beam_width = sequences[0]->sampling_param()->beam_width;
  if (beam_width <= 1) {
    return;
  }
  const int32_t result_width =
      output.beam_sequence_group.defined()
          ? static_cast<int32_t>(output.beam_sequence_group.size(1))
          : beam_width;
  int32_t total_rounds =
      static_cast<int32_t>(output.beam_sequence_group.size(2));
  size_t num_groups = sequence_groups_.size();
  if (num_groups == 0) {
    // Fallback: treat sequences_ as single group
    num_groups = sequences.size();
  }

  // Tensor should already be on CPU (transferred in get_model_output)
  auto seq_group_accessor = output.beam_sequence_group.accessor<int32_t, 3>();

  // out_logprobs from beam_search_output, shape: [batch * beam_width]
  // Tensor should already be on CPU (transferred in get_model_output)
  bool has_logprobs = output.beam_search_output.out_logprobs.defined() &&
                      output.beam_search_output.out_logprobs.numel() > 0;

  std::vector<std::vector<int32_t>> group_flat2d;
  std::vector<float> last_logprobs;
  group_flat2d.reserve(static_cast<size_t>(result_width));
  last_logprobs.reserve(static_cast<size_t>(result_width));

  for (size_t g = 0; g < num_groups; ++g) {
    group_flat2d.clear();
    last_logprobs.clear();

    for (int b = 0; b < result_width; ++b) {
      std::vector<int32_t> row_tokens;
      row_tokens.reserve(static_cast<size_t>(total_rounds));
      for (int c = 0; c < total_rounds; ++c) {
        // Access [g][b][c]
        row_tokens.push_back(seq_group_accessor[g][b][c]);
      }
      group_flat2d.emplace_back(std::move(row_tokens));
      if (has_logprobs) {
        // logprobs is flattened [batch * result_width] for multi-round widened
        // final output.
        int32_t logprob_idx = static_cast<int32_t>(g) * result_width + b;
        last_logprobs.push_back(
            output.beam_search_output.out_logprobs[logprob_idx].item<float>());
      }
    }
    // Access sequence from sequence_groups_ if available
    Sequence* seq = sequence_groups_.empty()
                        ? sequences[g]
                        : sequence_groups_[g]->sequences()[0].get();
    seq->set_beam_result(
        result_width, total_rounds, group_flat2d, last_logprobs);
  }
}

void Batch::process_sample_output(const SampleOutput& sample_output,
                                  bool replace_fake_token,
                                  bool force_requested_beam_result_size) {
  if (sample_output.embeddings.defined()) {
    const int64_t num_seqs = sample_output.embeddings.size(0);
    int64_t output_idx = 0;
    const auto sequences = get_sequences();
    for (auto* seq : sequences) {
      CHECK_LT(output_idx, num_seqs);
      auto cur_seq_embed =
          safe_to(sample_output.embeddings[output_idx++], torch::kFloat32);
      seq->update_embeddings(cur_seq_embed);
      seq->update_mtp_bootstrap_embedding(cur_seq_embed);
    }
  }

  // if sample_output.next_tokens not defined,
  // sample_output.next_tokens.size(0) value is 0,
  // this means all sequences are in prefill stage status.
  const int64_t num_outputs = sample_output.next_tokens.size(0);
  for (size_t output_idx = 0; output_idx < output_targets_.size();
       ++output_idx) {
    const auto& target = output_targets_[output_idx];
    auto* seq = target.sequence;
    CHECK(seq != nullptr);

    if (!target.from_sample_slot) {
      if (seq->finished()) {
        continue;
      }
      if (update_sequence_state(seq, replace_fake_token)) {
        continue;
      }
    }

    if (output_idx >= static_cast<size_t>(num_outputs)) {
      if (target.from_sample_slot) {
        append_token_for_sequence(
            seq, make_empty_logprob_placeholder(*seq), 0, replace_fake_token);
      }
      continue;
    }

    const auto token = build_token(output_idx,
                                   sample_output.next_tokens,
                                   sample_output.logprobs,
                                   sample_output.top_tokens,
                                   sample_output.top_logprobs);

    // always append a token, maybe true or fake token
    append_token_for_sequence(seq, token, 0, replace_fake_token);
  }
  if (replace_fake_token) {
    output_targets_.clear();
  }

  if (!::xllm::SchedulerConfig::get_instance().enable_schedule_overlap() ||
      replace_fake_token) {
    process_beam_search(force_requested_beam_result_size);
  }
}

bool Batch::update_sequence_state(Sequence* seq, bool replace_fake_token) {
  // In chunked prefill case, if enable_schedule_overlap, we need the
  // prefill-or-not state of last stage, otherwise, we need the state
  // of current stage.
  if (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()) {
    if (!replace_fake_token && seq->is_chunked_prefill_stage()) {
      seq->pre_scheduled_step_prefill_queue().push(true);
      // if not replace_fake_token, pop out here to avoid endless growth
      if (seq->pre_scheduled_step_prefill_queue().size() > 2) {
        seq->pre_scheduled_step_prefill_queue().pop();
      }
      return true;
    } else if (replace_fake_token &&
               seq->pre_scheduled_step_prefill_queue().front()) {
      seq->pre_scheduled_step_prefill_queue().pop();
      return true;
    }
  }
  return false;
}

void Batch::append_token_for_sequence(Sequence* seq,
                                      const Token& token,
                                      int token_idx,
                                      bool replace_fake_token) {
  // always append a token, maybe true or fake token
  if (!replace_fake_token) {
    seq->append_token(token);
    if (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()) {
      seq->pre_scheduled_step_prefill_queue().push(false);
      // if not replace_fake_token, pop out here to avoid endless growth
      if (seq->pre_scheduled_step_prefill_queue().size() > 2) {
        seq->pre_scheduled_step_prefill_queue().pop();
      }
    }
  } else if (!seq->cancelled()) {
    // truely update the real token if replace_fake_token
    seq->update_last_step_token(token, token_idx);
    if (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() &&
        token_idx == 0) {
      seq->pre_scheduled_step_prefill_queue().pop();
    }
  }
}

void Batch::process_beam_search(bool force_requested_result_size) {
  for (auto* sequence_group : sequence_groups_) {
    sequence_group->process_beam_search(force_requested_result_size);
  }
}

void Batch::process_beam_search_output(const RawForwardOutput& raw_output,
                                       bool replace_fake_token) {
  const int32_t beam_width = sequences_[0]->sampling_param()->beam_width;
  if (beam_width <= 1) {
    return;
  }

  CHECK_EQ(raw_output.src_seq_idxes.size(), sequences_.size());
  CHECK_EQ(raw_output.out_tokens.size(), sequences_.size());
  CHECK_EQ(raw_output.out_logprobs.size(), sequences_.size());

  auto update_for_sequence_group = [&](size_t sequence_group_id) {
    std::unordered_set<int32_t> seq_idx_set;
    std::vector<float> src_acc_logprob_vec;
    std::vector<std::vector<int32_t>> src_token_ids;
    std::vector<std::vector<std::optional<float>>> src_logprobs;
    src_acc_logprob_vec.resize(beam_width);
    src_token_ids.resize(beam_width);
    src_logprobs.resize(beam_width);

    for (size_t i = 0; i < beam_width; i++) {
      size_t task_id = sequence_group_id * beam_width + i;
      int32_t src_seq_idx = raw_output.src_seq_idxes[task_id];
      CHECK_LE(src_seq_idx, sequences_.size());
      auto src_seq = sequences_[src_seq_idx];
      src_acc_logprob_vec[i] = src_seq->get_acc_logprob();
      src_token_ids[i] = std::vector<int32_t>(src_seq->tokens());
      src_logprobs[i] = src_seq->logprob_state()->get_logprobs();
    }

    for (size_t i = 0; i < beam_width; i++) {
      size_t task_id = sequence_group_id * beam_width + i;
      int32_t src_seq_idx = raw_output.src_seq_idxes[task_id];
      CHECK_LE(src_seq_idx, sequences_.size());
      auto& base_seq = sequences_[task_id];
      auto& src_seq = sequences_[src_seq_idx];

      for (size_t token_idx = base_seq->num_prompt_tokens();
           token_idx < base_seq->num_tokens();
           token_idx++) {
        Token new_token(src_token_ids[i][token_idx]);
        new_token.logprob = src_logprobs[i][token_idx];
        base_seq->update_token(token_idx, new_token);
      }

      Token new_token(raw_output.out_tokens[task_id]);
      new_token.logprob =
          raw_output.out_logprobs[task_id] - src_acc_logprob_vec[i];
      append_token_for_sequence(base_seq, new_token, 0, replace_fake_token);

      base_seq->logprob_state()->set_acc_logprob(
          raw_output.out_logprobs[task_id]);
      base_seq->logprob_state()->set_last_acc_token_idx(base_seq->num_tokens());

      bool need_swap = false;
      if (seq_idx_set.find(src_seq_idx) != seq_idx_set.end()) {
        need_swap = true;
      } else {
        seq_idx_set.insert(src_seq_idx);
      }

      auto src_blocks = src_seq->kv_state().blocks(BlockType::KV);
      base_seq->kv_state().set_src_blocks(src_blocks, need_swap);
    }
  };

  for (size_t sequence_group_id = 0;
       sequence_group_id < sequence_groups_.size();
       sequence_group_id++) {
    update_for_sequence_group(sequence_group_id);
  }
}

void Batch::finish() {
  for (auto* sequence_group : sequence_groups_) {
    sequence_group->finish();
  }

  const auto sequences = get_sequences();
  for (auto* sequence : sequences) {
    sequence->finish();
  }
}
}  // namespace xllm
