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

#include "core/framework/batch/sampling_input_builder.h"

#include <glog/logging.h>

#include <iterator>
#include <limits>

#include "core/framework/request/sequence.h"
#include "core/util/tensor_helper.h"
#include "core/util/utils.h"

namespace xllm {

void SamplingInputBuilder::reserve(size_t rows) {
  params_.reserve(rows);
  sequences_.reserve(rows);
  selected_token_indices_.reserve(rows);
  token_ids_.reserve(rows);
  token_counts_.reserve(rows);
  token_lengths_.reserve(rows);
}

void SamplingInputBuilder::append(const RequestSamplingParam* params,
                                  int32_t token_index,
                                  const TokenCounts* counts,
                                  const TokenCounts* excluded_counts,
                                  const Sequence* sequence) {
  CHECK(params != nullptr);
  CHECK_GE(token_index, 0);
  CHECK_LT(size(), std::numeric_limits<int32_t>::max());
  params_.emplace_back(params);
  sequences_.emplace_back(sequence);
  selected_token_indices_.emplace_back(token_index);
  auto& ids = token_ids_.emplace_back();
  auto& frequencies = token_counts_.emplace_back();
  if (counts != nullptr) {
    ids.reserve(counts->size());
    frequencies.reserve(counts->size());
    for (const auto& [token, count] : *counts) {
      CHECK_GE(count, 0);
      int32_t excluded = 0;
      if (excluded_counts != nullptr) {
        const auto it = excluded_counts->find(token);
        excluded = it == excluded_counts->end() ? 0 : it->second;
        if (count <= excluded) {
          continue;
        }
      }
      ids.emplace_back(token);
      frequencies.emplace_back(count - excluded);
    }
  }
  token_lengths_.emplace_back(static_cast<int32_t>(ids.size()));
}

void SamplingInputBuilder::merge(SamplingInputBuilder other,
                                 int32_t token_offset) {
  CHECK_GE(token_offset, 0);
  CHECK_LE(size() + other.size(), std::numeric_limits<int32_t>::max());
  reserve(size() + other.size());
  for (int32_t index : other.selected_token_indices_) {
    CHECK_LE(index, std::numeric_limits<int32_t>::max() - token_offset);
    selected_token_indices_.emplace_back(index + token_offset);
  }
  params_.insert(params_.end(), other.params_.begin(), other.params_.end());
  sequences_.insert(
      sequences_.end(), other.sequences_.begin(), other.sequences_.end());
  token_ids_.insert(token_ids_.end(),
                    std::make_move_iterator(other.token_ids_.begin()),
                    std::make_move_iterator(other.token_ids_.end()));
  token_counts_.insert(token_counts_.end(),
                       std::make_move_iterator(other.token_counts_.begin()),
                       std::make_move_iterator(other.token_counts_.end()));
  token_lengths_.insert(token_lengths_.end(),
                        other.token_lengths_.begin(),
                        other.token_lengths_.end());
}

SamplingParameters SamplingInputBuilder::build() {
  SamplingParameters result;
  if (empty()) {
    return result;
  }
  util::pad_2d_vector<int64_t>(token_ids_, /*pad_value=*/0);
  util::pad_2d_vector(token_counts_, /*pad_value=*/0);
  result.init(params_,
              selected_token_indices_,
              token_ids_,
              token_counts_,
              token_lengths_);
  bool need_constraints = false;
  bool need_seed = false;
  bool need_device_min_tokens = false;
  size_t max_words = 0;
  size_t max_word_length = 0;
  int64_t vocab_size = 0;
  for (const auto* params : params_) {
    need_constraints = need_constraints || !params->logit_bias.empty() ||
                       params->allowed_token_ids.has_value() ||
                       (params->min_tokens > 0 && !params->device_min_tokens);
    need_device_min_tokens =
        need_device_min_tokens ||
        (params->device_min_tokens && params->min_tokens > 0);
    need_seed = need_seed || params->seed.has_value();
    max_words = std::max(max_words, params->bad_words_token_ids.size());
    for (const auto& word : params->bad_words_token_ids) {
      max_word_length = std::max(max_word_length, word.size());
    }
    vocab_size = std::max(vocab_size, params->vocab_size);
  }
  if (need_constraints) {
    CHECK_GT(vocab_size, 0);
    result.logits_bias() = torch::zeros(
        {static_cast<int64_t>(size()), vocab_size}, torch::kFloat32);
    auto rows = result.logits_bias().accessor<float, 2>();
    const float blocked = -std::numeric_limits<float>::infinity();
    for (size_t row = 0; row < size(); ++row) {
      const auto& params = *params_[row];
      const Sequence* sequence = sequences_[row];
      if (params.allowed_token_ids.has_value()) {
        std::fill_n(rows[row].data(), vocab_size, blocked);
        for (int32_t token : *params.allowed_token_ids) {
          rows[row][token] = 0;
        }
      }
      for (const auto& [token, bias] : params.logit_bias) {
        rows[row][token] += bias;
      }
      const size_t generated =
          sequence == nullptr ? 0 : sequence->num_generated_tokens();
      if (!params.device_min_tokens && generated < params.min_tokens) {
        for (int32_t token : params.all_stop_token_ids) {
          if (token >= 0 && token < vocab_size) {
            rows[row][token] = blocked;
          }
        }
      }
    }
  }
  if (max_words > 0) {
    result.bad_words() = torch::full({static_cast<int64_t>(size()),
                                      static_cast<int64_t>(max_words),
                                      static_cast<int64_t>(max_word_length)},
                                     -1,
                                     torch::kInt64);
    result.bad_word_suffix() =
        torch::full({static_cast<int64_t>(size()),
                     static_cast<int64_t>(max_word_length - 1)},
                    -1,
                    torch::kInt64);
    auto words = result.bad_words().accessor<int64_t, 3>();
    auto suffixes = result.bad_word_suffix().accessor<int64_t, 2>();
    for (size_t row = 0; row < size(); ++row) {
      const auto& bad_words = params_[row]->bad_words_token_ids;
      for (size_t index = 0; index < bad_words.size(); ++index) {
        const auto& word = bad_words[index];
        std::copy(word.begin(),
                  word.end(),
                  words[row][index].data() + max_word_length - word.size());
      }
      const Sequence* sequence = sequences_[row];
      if (sequence == nullptr) {
        continue;
      }
      const size_t generated = sequence->num_valid_generated_tokens();
      const size_t length = std::min(generated, max_word_length - 1);
      const auto output = sequence->tokens().slice(
          sequence->num_prompt_tokens() + generated - length,
          sequence->num_prompt_tokens() + generated);
      std::copy(output.begin(),
                output.end(),
                suffixes[row].data() + max_word_length - 1 - length);
    }
  }
  if (need_device_min_tokens) {
    CHECK_GT(vocab_size, 0);
    std::vector<int64_t> limits;
    std::vector<int64_t> remaining;
    std::vector<std::vector<int64_t>> stops;
    limits.reserve(size());
    remaining.reserve(size());
    stops.reserve(size());
    for (size_t row = 0; row < size(); ++row) {
      const auto& params = *params_[row];
      const Sequence* sequence = sequences_[row];
      const int64_t minimum = params.device_min_tokens ? params.min_tokens : 0;
      const int64_t generated =
          sequence == nullptr ? 0 : sequence->num_generated_tokens();
      const int64_t prompt_end =
          sequence == nullptr
              ? 0
              : static_cast<int64_t>(sequence->num_prompt_tokens()) - 1;
      limits.emplace_back(minimum == 0 ? 0 : prompt_end + minimum);
      remaining.emplace_back(std::max<int64_t>(0, minimum - generated));
      auto& ids = stops.emplace_back();
      ids.reserve(params.all_stop_token_ids.size() + 1);
      if (minimum > 0) {
        for (int32_t token : params.all_stop_token_ids) {
          if (token >= 0 && token < vocab_size) {
            ids.emplace_back(token);
          }
        }
      }
      // Keep the compact tensor well-formed even without a configured EOS.
      if (ids.empty()) {
        ids.emplace_back(-1);
      }
    }
    util::pad_2d_vector<int64_t>(stops, /*pad_value=*/-1);
    result.min_token_positions() = make_pinned_cpu_tensor(limits);
    result.min_tokens_remaining() = make_pinned_cpu_tensor(remaining);
    result.min_token_ids() = create_2d_tensor(stops, torch::kInt64);
  }
  if (need_seed) {
    std::vector<int64_t> seeds;
    std::vector<int64_t> offsets;
    seeds.reserve(size());
    offsets.reserve(size());
    for (size_t row = 0; row < size(); ++row) {
      seeds.emplace_back(params_[row]->seed.value_or(-1));
      const Sequence* sequence = sequences_[row];
      const uint64_t offset =
          sequence == nullptr
              ? 0
              : (static_cast<uint64_t>(sequence->index()) << 32) +
                    sequence->num_valid_generated_tokens();
      offsets.emplace_back(static_cast<int64_t>(offset));
    }
    result.seeds() = make_pinned_cpu_tensor(seeds);
    result.seed_offsets() = make_pinned_cpu_tensor(offsets);
  }
  return result;
}

}  // namespace xllm
