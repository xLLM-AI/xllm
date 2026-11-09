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
  if (seq->cancelled() || !embedding.defined()) {
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
    if (!sequence->cancelled() && error_iter != request_errors.end()) {
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
    BatchOutputAction action) {
  const bool replace_fake_token = action == BatchOutputAction::COMMIT_PENDING;
  const auto& sequences = data.sequences;
  const std::unordered_set<std::string> failed_request_ids =
      fail_json_object_requests(sequences, raw_output.json_object_errors);

  for (size_t output_idx = 0; output_idx < output_targets_.size();
       ++output_idx) {
    const auto& target = output_targets_[output_idx];
    auto* seq = target.sequence;
    CHECK(seq != nullptr);

    if (seq->cancelled() || failed_request_ids.contains(seq->request_id()) ||
        seq->error_status().has_value()) {
      continue;
    }

    if (output_idx < raw_output.outputs.size()) {
      seq->record_speculative_token_stats(
          raw_output.outputs[output_idx].speculative_token_stats);
      const auto& seq_mm_embeddings =
          raw_output.outputs[output_idx].mm_embeddings;
      if (!seq_mm_embeddings.empty()) {
        seq->update_mm_embeddings(
            seq_mm_embeddings,
            raw_output.outputs[output_idx].mm_embeddings_metadata);
      }
    }

    if (!target.from_sample_slot && seq->finished()) {
      continue;
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
  finish_sample_output(action);
}

void BatchOutputHandler::process_sample_output(
    const BatchOutputData& data,
    const SampleOutput& sample_output,
    BatchOutputAction action) {
  const bool replace_fake_token = action == BatchOutputAction::COMMIT_PENDING;
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
    if (seq->cancelled() || seq->error_status().has_value()) {
      continue;
    }

    if (!target.from_sample_slot && seq->finished()) {
      continue;
    }

    if (output_idx >= static_cast<size_t>(num_outputs)) {
      if (target.from_sample_slot) {
        append_token_for_sequence(
            seq, make_empty_logprob_placeholder(*seq), 0, replace_fake_token);
      }
      continue;
    }

    if (next_tokens_are_spec_width) {
      const int64_t num_tokens = sample_output.next_tokens.size(1);
      bool appended_token = false;
      for (int64_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
        const Token token =
            sample_output.token_at(static_cast<int64_t>(output_idx), token_idx);
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

    const Token token =
        sample_output.token_at(static_cast<int64_t>(output_idx));

    // always append a token, maybe true or fake token
    append_token_for_sequence(seq, token, 0, replace_fake_token);
  }
  finish_sample_output(action);
}

void BatchOutputHandler::finish_sample_output(BatchOutputAction action) {
  if (action == BatchOutputAction::COMMIT_PENDING) {
    output_targets_.clear();
  }
}

void BatchOutputHandler::append_token_for_sequence(Sequence* seq,
                                                   const Token& token,
                                                   int32_t token_idx,
                                                   bool replace_fake_token) {
  // always append a token, maybe true or fake token
  if (!replace_fake_token) {
    seq->append_token(token);
  } else if (!seq->cancelled()) {
    // truly update the real token if replace_fake_token
    seq->update_last_step_token(token, token_idx);
  }
}
}  // namespace xllm
