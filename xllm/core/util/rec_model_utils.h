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

#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "core/common/global_flags.h"
#include "core/common/types.h"

namespace xllm {

enum class RecModelKind : int8_t {
  kNone = 0,
  kOneRec = 1,
  kLlmRec = 2,
};

// Pipeline strategy types (extensible for future strategies)
enum class RecPipelineType : uint8_t {
  kLlmRecDefault = 0,             // LlmRec without mm_data (pure qwen)
  kLlmRecWithMmData = 1,          // LlmRec with mm_data (qwen + embedding)
  kOneRecDefault = 2,             // OneRec
  kLlmRecMultiRoundPipeline = 3,  // LlmRec multi-round pipeline (device loop)
  kOneRecXAttentionPipeline = 4,  // OneRec xattention pipeline (device loop)
};

// Check if Rec multi-round mode is enabled.
// Rec multi-round mode: multi-round decode loop runs on device (worker layer),
// while the engine issues a single step.
inline bool is_rec_multi_round_mode() { return FLAGS_max_decode_rounds > 0; }

// Get the number of decode rounds for Rec multi-round mode.
// Returns 0 if Rec multi-round mode is disabled.
inline int32_t get_rec_multi_round_decode_rounds() {
  return is_rec_multi_round_mode() ? FLAGS_max_decode_rounds : 0;
}

inline bool use_legacy_onerec_prefill_only_contract() {
  return FLAGS_enable_rec_prefill_only;
}

inline bool is_onerec_xattention_mode() {
  return !use_legacy_onerec_prefill_only_contract() &&
         FLAGS_max_decode_rounds > 0;
}

// OneRec single-step mode: run one step (prefill / round 0) only and return the
// sampler's full vocab probability distribution instead of a multi-round beam
// search sequence. Triggered by max_decode_rounds == 1 in the OneRec xattention
// path.
inline bool is_onerec_single_step_mode() {
  return is_onerec_xattention_mode() && FLAGS_max_decode_rounds == 1;
}

inline bool is_onerec_pipeline_type(RecPipelineType type) {
  return type == RecPipelineType::kOneRecDefault ||
         type == RecPipelineType::kOneRecXAttentionPipeline;
}

// Pipeline strategy selector: choose strategy based on RecModelKind
inline RecPipelineType get_rec_pipeline_type(RecModelKind kind) {
  switch (kind) {
    case RecModelKind::kLlmRec:
      if (is_rec_multi_round_mode()) {
        return RecPipelineType::kLlmRecMultiRoundPipeline;
      } else {
        return RecPipelineType::kLlmRecDefault;
      }
    case RecModelKind::kOneRec:
      return is_onerec_xattention_mode()
                 ? RecPipelineType::kOneRecXAttentionPipeline
                 : RecPipelineType::kOneRecDefault;
    default:
      return RecPipelineType::kLlmRecDefault;
  }
}

inline constexpr bool is_onerec_model_type(std::string_view model_type) {
  return model_type == "onerec";
}

inline constexpr bool is_llmrec_model_type(std::string_view model_type) {
  return model_type == "qwen2" || model_type == "qwen3" ||
         model_type == "qwen3_moe";
}

inline constexpr RecModelKind get_rec_model_kind(std::string_view model_type) {
  if (is_onerec_model_type(model_type)) {
    return RecModelKind::kOneRec;
  }
  if (is_llmrec_model_type(model_type)) {
    return RecModelKind::kLlmRec;
  }
  return RecModelKind::kNone;
}

// Sample up to `max_count` items from `src` without copying or shuffling the
// full source vector. `max_count <= 0` copies everything (legacy "no cap").
// Predictor sets each_conversion_threshold=1, which is a single O(1) pick.
template <typename T>
inline void append_sampled_rec_items(const std::vector<T>& src,
                                     int32_t max_count,
                                     size_t sequence_index,
                                     std::vector<T>* out) {
  if (out == nullptr || src.empty()) {
    return;
  }
  if (max_count <= 0 || static_cast<size_t>(max_count) >= src.size()) {
    out->insert(out->end(), src.begin(), src.end());
    return;
  }

  std::mt19937 seeded_rng;
  std::mt19937* gen = nullptr;
  if (FLAGS_random_seed >= 0) {
    seeded_rng = std::mt19937(static_cast<uint32_t>(FLAGS_random_seed) +
                              static_cast<uint32_t>(sequence_index));
    gen = &seeded_rng;
  } else {
    static thread_local std::mt19937 rng{std::random_device{}()};
    gen = &rng;
  }

  const size_t n = src.size();
  const size_t k = static_cast<size_t>(max_count);
  std::uniform_int_distribution<size_t> dist(0, n - 1);
  out->reserve(out->size() + k);
  if (k == 1) {
    out->push_back(src[dist(*gen)]);
    return;
  }

  std::unordered_set<size_t> picked;
  picked.reserve(k * 2);
  while (picked.size() < k) {
    picked.insert(dist(*gen));
  }
  for (const size_t idx : picked) {
    out->push_back(src[idx]);
  }
}

inline std::vector<int64_t> unique_rec_item_ids(
    const std::vector<int64_t>& raw_ids) {
  std::vector<int64_t> item_ids;
  item_ids.reserve(raw_ids.size());
  std::unordered_set<int64_t> seen_item_ids;
  for (const int64_t item_id : raw_ids) {
    if (seen_item_ids.insert(item_id).second) {
      item_ids.emplace_back(item_id);
    }
  }
  return item_ids;
}

inline std::vector<RecItemInfo> unique_rec_item_infos(
    const std::vector<RecItemInfo>& raw_item_infos) {
  std::vector<RecItemInfo> item_infos;
  item_infos.reserve(raw_item_infos.size());
  std::unordered_set<int64_t> seen_item_ids;
  for (const RecItemInfo& item_info : raw_item_infos) {
    if (seen_item_ids.insert(item_info.item_id).second) {
      item_infos.emplace_back(item_info);
    }
  }
  return item_infos;
}

// Dedup + optional truncate to `each_conversion_threshold`.
// Shared between single-round (Sequence::generate_onerec_output) and
// multi-round (SequencesGroup::generate_multi_round_output) item assembly so
// that downstream c_api / consumers see consistent item_ids regardless of the
// decode pipeline. Truncation samples O(k) items instead of shuffling O(n).
inline std::vector<int64_t> normalize_rec_item_ids(
    const std::vector<int64_t>& raw_ids,
    size_t sequence_index) {
  const int32_t each_threshold = FLAGS_each_conversion_threshold;
  if (each_threshold > 0 &&
      static_cast<int32_t>(raw_ids.size()) > each_threshold) {
    std::vector<int64_t> sampled;
    sampled.reserve(static_cast<size_t>(each_threshold));
    append_sampled_rec_items(raw_ids, each_threshold, sequence_index, &sampled);
    return unique_rec_item_ids(sampled);
  }
  return unique_rec_item_ids(raw_ids);
}

inline std::vector<RecItemInfo> normalize_rec_item_infos(
    const std::vector<RecItemInfo>& raw_item_infos,
    size_t sequence_index) {
  const int32_t each_threshold = FLAGS_each_conversion_threshold;
  if (each_threshold > 0 &&
      static_cast<int32_t>(raw_item_infos.size()) > each_threshold) {
    std::vector<RecItemInfo> sampled;
    sampled.reserve(static_cast<size_t>(each_threshold));
    append_sampled_rec_items(
        raw_item_infos, each_threshold, sequence_index, &sampled);
    return unique_rec_item_infos(sampled);
  }
  return unique_rec_item_infos(raw_item_infos);
}

}  // namespace xllm
