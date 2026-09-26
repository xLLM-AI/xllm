/* Copyright 2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "core/framework/batch/batch_output_handler.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/framework/config/scheduler_config.h"
#include "core/runtime/params_utils.h"
#include "core/util/tensor_helper.h"

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

void update_sequence_embedding(Sequence* seq, const torch::Tensor& embedding) {
  if (!embedding.defined()) {
    return;
  }
  torch::Tensor cur_seq_embed = safe_to(embedding, torch::kFloat32);
  seq->update_embeddings(cur_seq_embed);
  seq->update_mtp_bootstrap_embedding(cur_seq_embed);
}

std::unordered_set<std::string> fail_json_object_requests(
    const std::vector<Sequence*>& sequences,
    const std::vector<JsonObjectOutputError>& errors) {
  std::unordered_set<std::string> failed_request_ids;
  if (errors.empty()) {
    return failed_request_ids;
  }

  std::unordered_map<std::string, Sequence*> sequences_by_sample_id;
  sequences_by_sample_id.reserve(sequences.size());
  for (Sequence* sequence : sequences) {
    CHECK(sequence != nullptr);
    const bool inserted =
        sequences_by_sample_id.emplace(sequence->sample_sequence_id(), sequence)
            .second;
    CHECK(inserted) << "duplicate sampled sequence id in batch: "
                    << sequence->sample_sequence_id();
  }

  std::unordered_map<std::string, Status> request_errors;
  request_errors.reserve(errors.size());
  for (const JsonObjectOutputError& error : errors) {
    CHECK(!error.sample_sequence_id.empty())
        << "json_object output error has empty sampled sequence id";
    const auto sequence_iter =
        sequences_by_sample_id.find(error.sample_sequence_id);
    CHECK(sequence_iter != sequences_by_sample_id.end())
        << "json_object output error references unknown sampled sequence id: "
        << error.sample_sequence_id;

    const std::string& request_id = sequence_iter->second->request_id();
    request_errors.try_emplace(request_id,
                               StatusCode::UNKNOWN,
                               "json_object constrained decoding failed for " +
                                   error.sample_sequence_id + ": " +
                                   error.message);
    failed_request_ids.emplace(request_id);
  }

  for (Sequence* sequence : sequences) {
    const auto error_iter = request_errors.find(sequence->request_id());
    if (error_iter != request_errors.end()) {
      sequence->fail(error_iter->second);
    }
  }
  return failed_request_ids;
}

}  // namespace

void BatchOutputHandler::prepare(const BatchInputData& data) {
  clear();
  reserve(data.sequences.size());
  for (size_t seq_index = 0; seq_index < data.sequences.size(); ++seq_index) {
    add_sequence_targets(data.sequences[seq_index],
                         data.allowed_max_tokens[seq_index]);
  }
}

void BatchOutputHandler::add_sequence_target(Sequence* sequence) {
  CHECK(sequence != nullptr);
  output_targets_.emplace_back(OutputTarget{sequence, /*sample_id=*/0, false});
}

void BatchOutputHandler::add_sequence_targets(Sequence* sequence,
                                              uint32_t token_budget) {
  if (sequence == nullptr) {
    return;
  }

  const uint32_t n_tokens = static_cast<uint32_t>(sequence->tokens().size());
  const uint32_t n_kv_cache_tokens = sequence->kv_state().kv_cache_tokens_num();
  if (n_tokens <= n_kv_cache_tokens) {
    return;
  }

  CHECK_GT(token_budget, 0);
  const uint32_t q_seq_len =
      std::min(n_tokens - n_kv_cache_tokens, token_budget);
  const uint32_t seq_len = q_seq_len + n_kv_cache_tokens;
  const auto& sample_slots = sequence->sample_slots();

  if (sample_slots.empty()) {
    if (seq_len == n_tokens) {
      add_sequence_target(sequence);
    }
    return;
  }

  for (const auto& sample_slot : sample_slots) {
    const uint32_t sample_source_position =
        get_sample_source_position(sample_slot);
    if (sample_source_position < n_kv_cache_tokens ||
        sample_source_position >= seq_len) {
      continue;
    }
    output_targets_.emplace_back(OutputTarget{
        sequence, sample_slot.sample_id, /*from_sample_slot=*/true});
  }
}

void BatchOutputHandler::process_sample_output(
    const BatchOutputData& data,
    const RawForwardOutput& raw_output,
    bool replace_fake_token) {
  const auto& sequences = data.sequences;
  const std::unordered_set<std::string> failed_request_ids =
      fail_json_object_requests(sequences, raw_output.json_object_errors);

  for (size_t output_idx = 0; output_idx < output_targets_.size();
       ++output_idx) {
    const auto& target = output_targets_[output_idx];
    auto* seq = target.sequence;
    CHECK(seq != nullptr);

    if (failed_request_ids.contains(seq->request_id()) ||
        seq->error_status().has_value()) {
      continue;
    }

    if (output_idx < raw_output.outputs.size()) {
      seq->record_speculative_token_stats(
          raw_output.outputs[output_idx].speculative_token_stats);
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
      if (seq->error_status().has_value()) {
        break;
      }

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
    process_beam_search(data);
  }
}

void BatchOutputHandler::process_sample_output(
    const BatchOutputData& data,
    const SampleOutput& sample_output,
    bool replace_fake_token,
    bool force_requested_beam_result_size) {
  const bool has_next_tokens = sample_output.next_tokens.defined() &&
                               sample_output.next_tokens.numel() > 0;
  const bool next_tokens_are_spec_width =
      has_next_tokens && sample_output.next_tokens.dim() == 2;
  if (sample_output.embeddings.defined() && !next_tokens_are_spec_width) {
    const int64_t num_seqs = sample_output.embeddings.size(0);
    int64_t output_idx = 0;
    const auto& sequences = data.sequences;
    for (auto* seq : sequences) {
      CHECK_LT(output_idx, num_seqs);
      update_sequence_embedding(seq, sample_output.embeddings[output_idx++]);
    }
  }
  if (sample_output.embeddings.defined() && next_tokens_are_spec_width) {
    const int64_t num_embedding_rows = sample_output.embeddings.size(0);
    const int64_t num_token_rows = sample_output.next_tokens.size(0);
    int64_t output_idx = 0;
    const auto& sequences = data.sequences;
    for (auto* seq : sequences) {
      CHECK_LT(output_idx, num_token_rows);
      CHECK_LT(output_idx, num_embedding_rows);
      const auto curr_next_tokens = sample_output.next_tokens[output_idx];

      int64_t last_token_idx = -1;
      const int64_t num_tokens = curr_next_tokens.size(0);
      for (int64_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
        if (curr_next_tokens[token_idx].item<int64_t>() < 0) {
          break;
        }
        last_token_idx = token_idx;
      }

      if (last_token_idx >= 0) {
        torch::Tensor token_embeddings = sample_output.embeddings[output_idx];
        if (token_embeddings.dim() > 1) {
          CHECK_LT(last_token_idx, token_embeddings.size(0))
              << "speculative embedding token index out of range";
          token_embeddings = token_embeddings[last_token_idx];
        }
        update_sequence_embedding(seq, token_embeddings);
      }
      ++output_idx;
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
    if (seq->error_status().has_value()) {
      continue;
    }

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

    if (next_tokens_are_spec_width) {
      const auto curr_next_tokens = sample_output.next_tokens[output_idx];
      const auto curr_logprobs = sample_output.logprobs.defined()
                                     ? sample_output.logprobs[output_idx]
                                     : sample_output.logprobs;
      const auto curr_top_tokens = sample_output.top_tokens.defined()
                                       ? sample_output.top_tokens[output_idx]
                                       : sample_output.top_tokens;
      const auto curr_top_logprobs =
          sample_output.top_logprobs.defined()
              ? sample_output.top_logprobs[output_idx]
              : sample_output.top_logprobs;

      const int64_t num_tokens = curr_next_tokens.size(0);
      bool appended_token = false;
      for (int64_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
        const auto token = build_token(token_idx,
                                       curr_next_tokens,
                                       curr_logprobs,
                                       curr_top_tokens,
                                       curr_top_logprobs);
        if (token.id < 0) {
          break;
        }

        append_token_for_sequence(
            seq, token, static_cast<int32_t>(token_idx), replace_fake_token);
        appended_token = true;
        if (!target.from_sample_slot && seq->finished()) {
          break;
        }
      }

      if (!appended_token && target.from_sample_slot) {
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
    process_beam_search(data, force_requested_beam_result_size);
  }
}

bool BatchOutputHandler::update_sequence_state(Sequence* seq,
                                               bool replace_fake_token) {
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

void BatchOutputHandler::append_token_for_sequence(Sequence* seq,
                                                   const Token& token,
                                                   int32_t token_idx,
                                                   bool replace_fake_token) {
  // always append a token, maybe true or fake token
  if (!replace_fake_token) {
    seq->append_token(token);
    if (seq->error_status().has_value()) {
      return;
    }
    if (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()) {
      seq->pre_scheduled_step_prefill_queue().push(false);
      // if not replace_fake_token, pop out here to avoid endless growth
      if (seq->pre_scheduled_step_prefill_queue().size() > 2) {
        seq->pre_scheduled_step_prefill_queue().pop();
      }
    }
  } else if (!seq->cancelled()) {
    // truly update the real token if replace_fake_token
    seq->update_last_step_token(token, token_idx);
    if (seq->error_status().has_value()) {
      return;
    }
    if (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() &&
        token_idx == 0) {
      seq->pre_scheduled_step_prefill_queue().pop();
    }
  }
}

void BatchOutputHandler::process_beam_search(const BatchOutputData& data,
                                             bool force_requested_result_size) {
  for (auto* sequence_group : data.sequence_groups) {
    sequence_group->process_beam_search(force_requested_result_size);
  }
}

void BatchOutputHandler::process_beam_search_output(
    const BatchOutputData& data,
    const RawForwardOutput& raw_output,
    bool replace_fake_token) {
  const auto& sequences = data.sequences;
  const std::unordered_set<std::string> failed_request_ids =
      fail_json_object_requests(sequences, raw_output.json_object_errors);

  const int32_t beam_width = data.sequences[0]->sampling_param()->beam_width;
  if (beam_width <= 1) {
    return;
  }

  CHECK_EQ(raw_output.src_seq_idxes.size(), data.sequences.size());
  CHECK_EQ(raw_output.out_tokens.size(), data.sequences.size());
  CHECK_EQ(raw_output.out_logprobs.size(), data.sequences.size());

  auto update_for_sequence_group = [&](size_t sequence_group_id) {
    CHECK_LT(sequence_group_id, data.sequence_groups.size());
    const auto& group_sequences =
        data.sequence_groups[sequence_group_id]->sequences();
    CHECK(!group_sequences.empty());
    if (failed_request_ids.contains(group_sequences[0]->request_id())) {
      return;
    }

    std::unordered_set<int32_t> seq_idx_set;
    std::vector<float> src_acc_logprob_vec;
    std::vector<std::vector<int32_t>> src_token_ids;
    std::vector<std::vector<std::optional<float>>> src_logprobs;
    const bool restore_json_states =
        group_sequences[0]->json_object_state() != nullptr;
    std::vector<JsonObjectGrammarSnapshot> src_json_states;
    src_acc_logprob_vec.resize(beam_width);
    src_token_ids.resize(beam_width);
    src_logprobs.resize(beam_width);
    if (restore_json_states) {
      src_json_states.resize(beam_width);
    }

    for (size_t i = 0; i < beam_width; i++) {
      size_t task_id = sequence_group_id * beam_width + i;
      int32_t src_seq_idx = raw_output.src_seq_idxes[task_id];
      CHECK_GE(src_seq_idx, 0);
      CHECK_LT(static_cast<size_t>(src_seq_idx), data.sequences.size());
      auto src_seq = data.sequences[src_seq_idx];
      src_acc_logprob_vec[i] = src_seq->get_acc_logprob();
      src_token_ids[i] = std::vector<int32_t>(src_seq->tokens());
      src_logprobs[i] = src_seq->logprob_state()->get_logprobs();
      if (restore_json_states) {
        const JsonObjectGrammarState* json_state = src_seq->json_object_state();
        CHECK(json_state != nullptr)
            << "beam sequences in one request must share JSON grammar state";
        src_json_states[i] = json_state->snapshot();
      }
    }

    for (size_t i = 0; i < beam_width; i++) {
      size_t task_id = sequence_group_id * beam_width + i;
      int32_t src_seq_idx = raw_output.src_seq_idxes[task_id];
      CHECK_GE(src_seq_idx, 0);
      CHECK_LT(static_cast<size_t>(src_seq_idx), data.sequences.size());
      auto& base_seq = data.sequences[task_id];
      auto& src_seq = data.sequences[src_seq_idx];

      if (restore_json_states &&
          !base_seq->restore_json_object_state(src_json_states[i])) {
        return;
      }

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
      if (base_seq->error_status().has_value()) {
        return;
      }

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
       sequence_group_id < data.sequence_groups.size();
       sequence_group_id++) {
    update_for_sequence_group(sequence_group_id);
  }
}

}  // namespace xllm
