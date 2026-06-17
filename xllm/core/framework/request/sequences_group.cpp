/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "sequences_group.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

#include "common/global_flags.h"
#include "core/framework/config/rec_config.h"
#include "core/util/rec_model_utils.h"
#include "framework/batch/beam_search.h"
#include "util/blocking_counter.h"
#include "util/slice.h"

namespace xllm {

SequencesGroup::SequencesGroup(const std::string& prompt,
                               const std::vector<int32_t>& prompt_tokens,
                               const torch::Tensor& input_embedding,
                               const MMData& mm_data,
                               const SequenceParams& sequence_params)
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
  sequences_.emplace_back(std::make_unique<Sequence>(index,
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
  // when enable beam search, dont need to expand sequences
  if (check_beam_search()) {
    return false;
  }
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

  // Check for multi-round beam search results
  if (is_rec_multi_round_mode() && check_beam_search() &&
      sequences_.size() == 1) {
    auto* base = sequences_[0].get();
    if (base->has_beam_result()) {
      generate_multi_round_output(outputs, tokenizer, *base);
      return;
    }
  }

  if (sequence_params_.streaming) {
    for (auto& seq : sequences_) {
      outputs.push_back(std::move(seq->generate_output()));
    }
    return;
  }

  auto n = sequence_params_.n;
  auto num = std::min(sequences_.size(), n);
  outputs.reserve(num);
  if (sequences_.size() > n && !check_beam_search()) {
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

void SequencesGroup::process_beam_search(bool force_requested_result_size) {
  if (!check_beam_search()) {
    return;
  }

  if (sequences_.empty()) {
    return;
  }

  const size_t beam_width = sequence_params_.sampling_param->beam_width;
  const int32_t requested_num_return_sequences =
      sequence_params_.sampling_param->num_return_sequences > 0
          ? sequence_params_.sampling_param->num_return_sequences
          : sequence_params_.sampling_param->beam_width;
  const size_t requested_result_size =
      static_cast<size_t>(requested_num_return_sequences);
  const size_t topk =
      std::max<size_t>(1, sequence_params_.sampling_param->top_logprobs);

  std::vector<BeamSourceInfo> source_infos;
  source_infos.reserve(sequences_.size());

  auto build_source_info = [&](Sequence* seq) {
    BeamSourceInfo source_info;
    source_info.suffix_start_idx = seq->num_prompt_tokens();

    const auto token_ids = seq->tokens();
    const auto& log_probs = seq->logprob_state()->get_logprobs();
    const size_t generated_token_count =
        token_ids.size() - source_info.suffix_start_idx;
    source_info.generated_token_ids.reserve(generated_token_count);
    source_info.generated_logprobs.reserve(generated_token_count);
    for (size_t token_idx = source_info.suffix_start_idx;
         token_idx < token_ids.size();
         ++token_idx) {
      source_info.generated_token_ids.push_back(token_ids[token_idx]);
      source_info.generated_logprobs.push_back(log_probs[token_idx]);
    }
    source_info.src_blocks.assign(seq->kv_state().kv_blocks().begin(),
                                  seq->kv_state().kv_blocks().end());

    source_infos.emplace_back(std::move(source_info));
  };

  for (size_t seq_index = 0; seq_index < sequences_.size(); ++seq_index) {
    build_source_info(sequences_[seq_index].get());
  }

  const size_t target_result_size =
      force_requested_result_size ? requested_result_size : beam_width;
  SimpleTopKOptimizerBeamCandidate topk_optimizer(target_result_size);
  auto add_self_candidate = [&](size_t seq_index, Sequence* seq) {
    BeamCandidate candidate;
    candidate.source_index = seq_index;
    candidate.logprob_sum = seq->get_acc_logprob();
    topk_optimizer.insert(std::move(candidate));
  };

  for (size_t i = 0; i < sequences_.size(); ++i) {
    auto* seq = sequences_[i].get();
    const bool updated_this_step = seq->updated_since_last_beam_search();
    const bool seq_finished = seq->finished();

    if (seq_finished && !updated_this_step) {
      // A beam that was already finished before this step should be kept
      // as-is and no longer expanded.
      add_self_candidate(i, seq);
      continue;
    }

    if (seq->num_tokens() == 0) {
      add_self_candidate(i, seq);
      continue;
    }

    const int32_t last_token_idx = seq->num_tokens() - 1;
    const auto& top_logprobs =
        seq->logprob_state()->get_top_logprobs()[last_token_idx];
    const auto& top_tokens =
        seq->logprob_state()->get_top_tokens()[last_token_idx];
    const size_t candidate_topk = std::min<size_t>(
        topk, std::min(top_logprobs.size(), top_tokens.size()));
    if (candidate_topk == 0) {
      add_self_candidate(i, seq);
      continue;
    }

    const float base_logprob = seq->get_base_logprob();
    const size_t source_index = i;
    for (size_t idx = 0; idx < candidate_topk; ++idx) {
      const float new_logprob = base_logprob + top_logprobs[idx];
      if (!topk_optimizer.worthInserting(new_logprob)) {
        break;
      }

      BeamCandidate candidate;
      candidate.source_index = source_index;
      candidate.logprob_sum = new_logprob;
      candidate.override_last_token = true;
      candidate.last_token_id = static_cast<int32_t>(top_tokens[idx]);
      candidate.last_token_logprob = top_logprobs[idx];
      topk_optimizer.insert(std::move(candidate));
    }
  }

  std::vector<BeamCandidate> candidates = topk_optimizer.getTopKSorted();
  for (auto& seq : sequences_) {
    seq->clear_updated_since_last_beam_search();
  }
  if (candidates.empty()) {
    return;
  }

  const size_t result_size = std::min(target_result_size, candidates.size());
  CHECK(!sequences_.empty());

  const size_t existing_size = sequences_.size();
  const size_t existing_result_size = std::min(result_size, existing_size);
  std::vector<std::unique_ptr<Sequence>> replacement_sequences(
      existing_result_size);
  std::vector<std::unique_ptr<Sequence>> tail_sequences;
  if (result_size > existing_size) {
    tail_sequences.reserve(result_size - existing_size);
  }
  for (size_t i = 0; i < result_size; ++i) {
    const BeamCandidate& candidate = candidates[i];
    const BeamSourceInfo& source_info = source_infos[candidate.source_index];
    const bool need_replace =
        i >= existing_size || sequences_[i] == nullptr ||
        sequences_[i]->num_prompt_tokens() != source_info.suffix_start_idx ||
        sequences_[i]->num_tokens() - source_info.suffix_start_idx !=
            source_info.generated_token_ids.size();
    if (!need_replace) {
      continue;
    }

    CHECK_LT(candidate.source_index, sequences_.size());
    CHECK(sequences_[candidate.source_index] != nullptr);
    if (i < existing_size) {
      replacement_sequences[i] =
          std::make_unique<Sequence>(*sequences_[candidate.source_index]);
    } else {
      tail_sequences.emplace_back(
          std::make_unique<Sequence>(*sequences_[candidate.source_index]));
    }
  }

  if (existing_size < result_size) {
    sequences_.reserve(result_size);
    for (auto& tail_sequence : tail_sequences) {
      sequences_.emplace_back(std::move(tail_sequence));
    }
  }

  std::unordered_set<size_t> reused_src;
  for (size_t i = 0; i < result_size; ++i) {
    const BeamCandidate& candidate = candidates[i];
    const BeamSourceInfo& source_info = source_infos[candidate.source_index];
    if (i < replacement_sequences.size() &&
        replacement_sequences[i] != nullptr) {
      sequences_[i] = std::move(replacement_sequences[i]);
    }
    auto& next_seq = sequences_[i];
    CHECK(next_seq != nullptr);

    CHECK_EQ(next_seq->num_prompt_tokens(), source_info.suffix_start_idx);
    CHECK_EQ(next_seq->num_tokens() - source_info.suffix_start_idx,
             source_info.generated_token_ids.size());
    for (size_t token_idx = source_info.suffix_start_idx;
         token_idx < next_seq->num_tokens();
         ++token_idx) {
      const size_t suffix_idx = token_idx - source_info.suffix_start_idx;
      Token new_token(source_info.generated_token_ids[suffix_idx]);
      new_token.logprob =
          source_info.generated_logprobs[suffix_idx].has_value()
              ? source_info.generated_logprobs[suffix_idx].value()
              : 0.0f;
      if (candidate.override_last_token &&
          token_idx + 1 == next_seq->num_tokens()) {
        new_token.id = candidate.last_token_id;
        new_token.logprob = candidate.last_token_logprob.has_value()
                                ? candidate.last_token_logprob.value()
                                : 0.0f;
      }
      next_seq->update_token(token_idx, new_token);
    }
    next_seq->logprob_state()->set_acc_logprob(candidate.logprob_sum);
    next_seq->logprob_state()->set_last_acc_token_idx(next_seq->num_tokens());
    next_seq->reset_finish_state_for_beam_search();

    const bool need_swap = !reused_src.insert(candidate.source_index).second;
    next_seq->kv_state().set_src_blocks(source_info.src_blocks, need_swap);
  }

  if (sequences_.size() > result_size) {
    sequences_.resize(result_size);
  }
}

void SequencesGroup::finish() {
  for (auto& sequence : sequences_) {
    sequence->finish();
  }
}

void SequencesGroup::generate_multi_round_output(
    std::vector<SequenceOutput>& outputs,
    const Tokenizer& tokenizer,
    const Sequence& base) {
  size_t bw = static_cast<size_t>(base.beam_width_cached());
  const auto& last_lps = base.beam_last_logprobs();

  // Rank by logprob
  std::vector<std::pair<float, size_t>> rank;
  rank.reserve(bw);
  for (size_t b = 0; b < bw; ++b) {
    float lp = (b < last_lps.size()) ? last_lps[b] : 0.0f;
    rank.emplace_back(lp, b);
  }
  std::sort(rank.begin(), rank.end(), [](const auto& l, const auto& r) {
    return l.first > r.first;
  });

  const auto& flat2d = base.beam_seq_group_flat();
  size_t rounds = static_cast<size_t>(base.total_rounds_cached());
  outputs.reserve(bw);

  for (size_t i = 0; i < bw; ++i) {
    size_t b = rank[i].second;
    std::vector<int32_t> gen_ids(flat2d[b].begin(), flat2d[b].end());

    SequenceOutput out;
    out.index = i;
    out.text = tokenizer.decode(Slice<int32_t>{gen_ids.data(), gen_ids.size()},
                                sequence_params_.skip_special_tokens);
    out.token_ids = std::move(gen_ids);
    std::vector<LogProb> log_probs;
    log_probs.resize(1);
    log_probs[0].logprob = rank[i].first;
    out.logprobs = log_probs;

    auto fr = base.finish_reason().to_string();
    if (fr.has_value()) {
      out.finish_reason = fr.value();
    }
    outputs.push_back(std::move(out));
  }
}

}  // namespace xllm
