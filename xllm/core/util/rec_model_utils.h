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

#include <cstdint>
#include <string_view>

#include "core/common/global_flags.h"
#include "core/framework/config/rec_config.h"

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
inline bool is_rec_multi_round_mode() {
  return ::xllm::RecConfig::get_instance().max_decode_rounds() > 0;
}

// Get the number of decode rounds for Rec multi-round mode.
// Returns 0 if Rec multi-round mode is disabled.
inline int32_t get_rec_multi_round_decode_rounds() {
  return is_rec_multi_round_mode()
             ? ::xllm::RecConfig::get_instance().max_decode_rounds()
             : 0;
}

inline bool is_onerec_xattention_mode() {
  return ::xllm::RecConfig::get_instance().max_decode_rounds() > 0;
}

// This helper intentionally names the legacy prefill-only contract instead of
// mirroring the raw flag. Under the current CLI wiring, xattention mode and
// enable_rec_prefill_only should not be active together, but many call sites
// still need to branch on "should I use the old prefill-only behavior" as a
// semantic boundary.
inline bool use_legacy_onerec_prefill_only_contract() {
  return ::xllm::RecConfig::get_instance().enable_rec_prefill_only() &&
         !is_onerec_xattention_mode();
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

}  // namespace xllm
