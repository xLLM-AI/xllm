/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "core/common/macros.h"
#include "core/framework/config/option_category.h"

namespace xllm {

class JsonReader;

class SchedulerConfig final {
 public:
  SchedulerConfig() = default;
  ~SchedulerConfig() = default;

  static SchedulerConfig& get_instance();

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "SCHEDULER OPTIONS",
        {"max_tokens_per_batch",
         "max_seqs_per_batch",
         "enable_schedule_overlap",
         "prefill_scheduling_memory_usage_threshold",
         "enable_chunked_prefill",
         "max_tokens_per_chunk_for_prefill",
         "chunked_match_frequency",
         "use_zero_evict",
         "max_decode_token_per_sequence",
         "priority_strategy",
         "use_mix_scheduler",
         "enable_online_preempt_offline",
         "aggressive_coeff",
         "starve_threshold",
         "enable_starve_prevent",
         "enable_mix_decode_first",
         "mix_decode_token_budget",
         "mix_max_prefill_chunks_per_step"}};
    return kOptionCategory;
  }

  PROPERTY(int32_t, max_tokens_per_batch) = 10240;

  PROPERTY(int32_t, max_seqs_per_batch) = 1024;

  PROPERTY(bool, enable_schedule_overlap) = false;

  PROPERTY(double, prefill_scheduling_memory_usage_threshold) = 0.95;

  PROPERTY(bool, enable_chunked_prefill) = true;

  PROPERTY(int32_t, max_tokens_per_chunk_for_prefill) = -1;

  PROPERTY(int32_t, chunked_match_frequency) = 2;

  PROPERTY(bool, use_zero_evict) = false;

  PROPERTY(int32_t, max_decode_token_per_sequence) = 256;

  PROPERTY(std::string, priority_strategy) = "fcfs";

  PROPERTY(bool, use_mix_scheduler) = false;

  PROPERTY(bool, enable_online_preempt_offline) = true;

  PROPERTY(double, aggressive_coeff) = 1.0;

  PROPERTY(double, starve_threshold) = 1.0;

  PROPERTY(bool, enable_starve_prevent) = true;

  // MixScheduler decode-first scheduling: when true, prepare_batch processes
  // decode-stage requests first (so they get the full token budget) before
  // admitting prefill chunks. Mitigates the "decode tpot dragged by prefill
  // chunk step time" problem in colocate (MIX) topologies.
  PROPERTY(bool, enable_mix_decode_first) = false;

  // Reserved token budget for decode in MixScheduler. When >0, prefill chunks
  // are restricted to (max_tokens_per_batch - mix_decode_token_budget) tokens
  // per step, giving decode requests guaranteed headroom even when prefill
  // would otherwise consume the entire batch budget. Only takes effect when
  // enable_mix_decode_first is true. 0 disables the cap.
  PROPERTY(int32_t, mix_decode_token_budget) = 0;

  // Path C cap on number of prefill chunks admitted into the same step in
  // MixScheduler. When >0, phase 2 (prefill admission) stops after this many
  // prefill sequences enter running_sequences_, regardless of token budget.
  // Each request has 1 sequence so this directly caps prefill chunks per
  // forward. Bounds how much prefill drags decode tpot. Only takes effect
  // when enable_mix_decode_first is true. 0 disables the cap.
  PROPERTY(int32_t, mix_max_prefill_chunks_per_step) = 0;
};

}  // namespace xllm
