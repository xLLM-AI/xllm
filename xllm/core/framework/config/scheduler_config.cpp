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

#include "core/framework/config/scheduler_config.h"

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"

DEFINE_int32(max_tokens_per_batch, 10240, "Max number of tokens per batch.");

DEFINE_int32(max_seqs_per_batch, 200, "Max number of sequences per batch.");

DEFINE_bool(enable_schedule_overlap,
            false,
            "Whether to enable schedule overlap.");

DEFINE_double(prefill_scheduling_memory_usage_threshold,
              0.95,
              "The memory usage threshold during prefill scheduling.");

DEFINE_bool(enable_chunked_prefill, true, "Whether to enable chunked prefill.");

DEFINE_int32(max_tokens_per_chunk_for_prefill,
             -1,
             "Max number of token per chunk in prefill stage.");

DEFINE_int32(chunked_match_frequency,
             2,
             "Number of sequence prefix cache match frequency.");

DEFINE_bool(enable_mix_batch,
            true,
            "Enable mixed batch (prefill + decode in same batch). "
            "Forced to false when CP or MTP is active.");

DEFINE_bool(use_zero_evict,
            false,
            "Use ZeroEvictionScheduler but ContinuousScheduler.");

DEFINE_int32(
    max_decode_token_per_sequence,
    256,
    "Max decode token per sequence which used for ZeroEvictionScheduler.");

DEFINE_string(priority_strategy,
              "fcfs",
              "Priority strategy for requests(e.g. fcfs, priority, deadline).");

DEFINE_bool(enable_online_preempt_offline,
            true,
            "Whether to enable online preempt offline.");

DEFINE_double(aggressive_coeff,
              1.0,
              "Aggressive coefficient for MixScheduler urgency judgment.");

DEFINE_double(starve_threshold,
              1.0,
              "Starvation threshold coefficient for MixScheduler.");

DEFINE_bool(enable_starve_prevent,
            true,
            "Whether to enable anti-starvation in MixScheduler.");

namespace xllm {

void SchedulerConfig::from_flags() {
  XLLM_CONFIG_ASSIGN_FROM_FLAG(max_tokens_per_batch);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(max_seqs_per_batch);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_schedule_overlap);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(prefill_scheduling_memory_usage_threshold);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_chunked_prefill);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(max_tokens_per_chunk_for_prefill);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(chunked_match_frequency);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(use_zero_evict);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(max_decode_token_per_sequence);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(priority_strategy);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_mix_batch);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_online_preempt_offline);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(aggressive_coeff);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(starve_threshold);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_starve_prevent);
}

void SchedulerConfig::from_json(const JsonReader& json) {
  XLLM_CONFIG_ASSIGN_FROM_JSON(max_tokens_per_batch);
  XLLM_CONFIG_ASSIGN_FROM_JSON(max_seqs_per_batch);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_schedule_overlap);
  XLLM_CONFIG_ASSIGN_FROM_JSON(prefill_scheduling_memory_usage_threshold);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_chunked_prefill);
  XLLM_CONFIG_ASSIGN_FROM_JSON(max_tokens_per_chunk_for_prefill);
  XLLM_CONFIG_ASSIGN_FROM_JSON(chunked_match_frequency);
  XLLM_CONFIG_ASSIGN_FROM_JSON(use_zero_evict);
  XLLM_CONFIG_ASSIGN_FROM_JSON(max_decode_token_per_sequence);
  XLLM_CONFIG_ASSIGN_FROM_JSON(priority_strategy);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_mix_batch);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_online_preempt_offline);
  XLLM_CONFIG_ASSIGN_FROM_JSON(aggressive_coeff);
  XLLM_CONFIG_ASSIGN_FROM_JSON(starve_threshold);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_starve_prevent);
}

void SchedulerConfig::append_config_json(
    nlohmann::ordered_json& config_json) const {
  const SchedulerConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, max_tokens_per_batch);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, max_seqs_per_batch);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_schedule_overlap);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, prefill_scheduling_memory_usage_threshold);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_chunked_prefill);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, max_tokens_per_chunk_for_prefill);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, chunked_match_frequency);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, use_zero_evict);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, max_decode_token_per_sequence);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, priority_strategy);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_mix_batch);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_online_preempt_offline);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, aggressive_coeff);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, starve_threshold);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_starve_prevent);
}

SchedulerConfig& SchedulerConfig::get_instance() {
  static SchedulerConfig config;
  return config;
}

void SchedulerConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
}

}  // namespace xllm
