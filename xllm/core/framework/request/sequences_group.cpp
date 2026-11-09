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

#include "sequences_group.h"

#include <algorithm>
#include <limits>

#include "core/framework/request/sequence_factory.h"
#include "util/blocking_counter.h"

namespace xllm {

SequencesGroup::SequencesGroup(const std::string& prompt,
                               const std::vector<int32_t>& prompt_tokens,
                               const torch::Tensor& input_embedding,
                               const MMData& mm_data,
                               SequenceParams sequence_params)
    : prompt_(prompt),
      prompt_tokens_(prompt_tokens),
      input_embedding_(input_embedding),
      mm_data_(mm_data),
      sequence_params_(std::move(sequence_params)) {
  add();
}

void SequencesGroup::add() {
  const size_t index = sequences_.size();
  IncrementalDecoder decoder(prompt_,
                             prompt_tokens_.size(),
                             sequence_params_.echo,
                             sequence_params_.skip_special_tokens);
  sequences_.emplace_back(create_sequence(index,
                                          prompt_tokens_,
                                          input_embedding_,
                                          mm_data_,
                                          std::move(decoder),
                                          sequence_params_));
}

bool SequencesGroup::finished() const {
  if (sequences_.size() < sequence_params_.best_of) {
    return false;
  }

  return std::all_of(sequences_.begin(),
                     sequences_.end(),
                     [](const auto& sequence) { return sequence->finished(); });
}

bool SequencesGroup::expand_sequences(bool share_prefix) {
  size_t current_seq_count = sequences_.size();
  size_t best_of = sequence_params_.best_of;
  if (current_seq_count == best_of) {
    return false;
  } else if (current_seq_count > best_of) {
    LOG(FATAL) << "Request has more than " << best_of << " sequences.";
    return false;
  }

  CHECK(!sequences_.empty()) << "Request has no sequence.";
  const auto& seq = sequences_[0];
  // prefill is not finished, can not expand
  // FIXME later share_prefix
  if (!share_prefix ||
      seq->kv_state().kv_cache_tokens_num() >= seq->num_prompt_tokens()) {
    while (sequences_.size() < best_of) {
      add();
    }
    return true;
  }

  return false;
}

void SequencesGroup::generate_outputs(std::vector<SequenceOutput>& outputs,
                                      const Tokenizer& tokenizer,
                                      ThreadPool* thread_pool) {
  const bool has_sample_outputs =
      std::any_of(sequences_.begin(), sequences_.end(), [](const auto& seq) {
        return seq != nullptr && !seq->sample_slots().empty();
      });
  if (has_sample_outputs) {
    const size_t previous_size = outputs.size();
    size_t total_outputs = previous_size;
    for (const auto& seq : sequences_) {
      if (seq == nullptr) {
        continue;
      }
      total_outputs +=
          seq->sample_slots().empty() ? 1 : seq->sample_slots().size();
    }
    outputs.reserve(total_outputs);
    for (auto& seq : sequences_) {
      if (seq == nullptr) {
        continue;
      }
      seq->generate_sample_outputs(outputs, tokenizer);
    }
    std::stable_sort(
        outputs.begin() + previous_size,
        outputs.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.index < rhs.index; });
    return;
  }

  if (sequence_params_.streaming) {
    outputs.reserve(sequences_.size());
    for (auto& seq : sequences_) {
      SequenceOutput output = seq->generate_output();
      // A received PD token can finish the request before any Decode step.
      // Drain text and metadata not consumed by earlier stream callbacks.
      auto delta = seq->generate_streaming_output(seq->num_tokens(), tokenizer);
      if (delta.has_value()) {
        output.text = std::move(delta->text);
        output.token_ids = std::move(delta->token_ids);
        output.logprobs = std::move(delta->logprobs);
      }
      outputs.emplace_back(std::move(output));
    }
    return;
  }

  auto n = sequence_params_.n;
  auto num = std::min(sequences_.size(), n);
  outputs.reserve(num);
  if (sequences_.size() > n) {
    std::vector<std::pair<float, size_t>> logprobs_vec;
    logprobs_vec.reserve(sequences_.size());
    for (size_t i = 0; i < sequences_.size(); ++i) {
      const int32_t generated_tokens_num =
          sequences_[i]->num_generated_tokens();
      // Use lowest() (negative) instead of min() (smallest positive) so
      // candidates that produced no tokens do not outrank real samples
      // (real average logprobs are negative).
      const float average_logprob =
          generated_tokens_num > 0
              ? sequences_[i]->get_acc_logprob() / generated_tokens_num
              : std::numeric_limits<float>::lowest();
      logprobs_vec.emplace_back(average_logprob, i);
    }
    std::sort(logprobs_vec.begin(),
              logprobs_vec.end(),
              [](const auto& l, const auto& r) { return l.first > r.first; });
    for (size_t i = 0; i < n; ++i) {
      const auto [logprob, index] = logprobs_vec[i];
      auto seq_output = sequences_[index]->generate_output(tokenizer);
      seq_output.index = i;
      outputs.push_back(std::move(seq_output));
    }
  } else {
    // speed up when using thread pool
    if (thread_pool && sequences_.size() > 1) {
      generate_outputs_parallel(outputs, tokenizer, thread_pool);
      return;
    }
    for (auto& seq : sequences_) {
      outputs.push_back(seq->generate_output(tokenizer));
    }
  }
}

void SequencesGroup::generate_outputs_parallel(
    std::vector<SequenceOutput>& outputs,
    const Tokenizer& tokenizer,
    ThreadPool* thread_pool) {
  size_t seq_size = sequences_.size();
  outputs.reserve(seq_size);
  size_t num_tasks = std::min(thread_pool->size(), seq_size);
  size_t num_tasks_pre =
      (seq_size + thread_pool->size() - 1) / thread_pool->size();
  std::vector<std::vector<SequenceOutput>> local_outputs;
  local_outputs.resize(num_tasks);

  auto gererate_output_local = [&](size_t work_id) {
    local_outputs[work_id].reserve(num_tasks_pre);
    for (size_t i = 0; i < num_tasks_pre; ++i) {
      size_t task_id = work_id * num_tasks_pre + i;
      if (task_id >= seq_size) break;

      const auto& base_seq = sequences_[task_id];
      local_outputs[work_id].emplace_back(base_seq->generate_output(tokenizer));
    }
  };

  BlockingCounter counter(num_tasks);

  {
    for (size_t i = 0; i < num_tasks; ++i) {
      thread_pool->schedule([gererate_output_local, i, &counter]() mutable {
        gererate_output_local(i);
        counter.decrement_count();
      });
    }

    counter.wait();
  }

  for (size_t i = 0; i < num_tasks; ++i) {
    outputs.insert(
        outputs.end(), local_outputs[i].begin(), local_outputs[i].end());
  }
}

void SequencesGroup::finish() {
  for (auto& sequence : sequences_) {
    sequence->finish();
  }
}

}  // namespace xllm
