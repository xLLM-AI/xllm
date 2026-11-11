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

#include "core/framework/config/scheduler_config.h"

#include <tuple>
#include <type_traits>
#include <utility>

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

DEFINE_string(priority_strategy,
              "fcfs",
              "Priority strategy for requests(e.g. fcfs, priority, deadline).");

DEFINE_double(aggressive_coeff,
              1.0,
              "Aggressive coefficient for MixScheduler urgency judgment.");

DEFINE_double(starve_threshold,
              1.0,
              "Starvation threshold coefficient for MixScheduler.");

DEFINE_bool(enable_starve_prevent,
            true,
            "Whether to enable anti-starvation in MixScheduler.");

DEFINE_bool(enable_dp_fair_token_budget,
            true,
            "Fair per-DP-group token budget for prefill scheduling on disagg "
            "PD PREFILL instances. Caps each DP group at "
            "max_tokens_per_batch / dp_size tokens per scheduling round.");

DEFINE_int32(request_queue_size,
             100000,
             "The request queue size of the scheduler");

namespace xllm {
namespace {

template <typename T>
struct SchedulerConfigField {
  using Value = T;
  const char* name;
  T* flag;
  config::ConfigInput<T> SchedulerConfigInput::* input;
  const T& (SchedulerConfig::*get)() const& noexcept;
  SchedulerConfig& (SchedulerConfig::*set)(const T&)&;
};

// One mapping serves the adapters, resolution and serialization. Runtime
// consumers continue to access plain typed values in SchedulerConfig.
const auto kSchedulerFields = std::tuple{
    SchedulerConfigField<int32_t>{"max_tokens_per_batch",
                                  &FLAGS_max_tokens_per_batch,
                                  &SchedulerConfigInput::max_tokens_per_batch,
                                  &SchedulerConfig::max_tokens_per_batch,
                                  &SchedulerConfig::max_tokens_per_batch},
    SchedulerConfigField<int32_t>{"max_seqs_per_batch",
                                  &FLAGS_max_seqs_per_batch,
                                  &SchedulerConfigInput::max_seqs_per_batch,
                                  &SchedulerConfig::max_seqs_per_batch,
                                  &SchedulerConfig::max_seqs_per_batch},
    SchedulerConfigField<bool>{"enable_schedule_overlap",
                               &FLAGS_enable_schedule_overlap,
                               &SchedulerConfigInput::enable_schedule_overlap,
                               &SchedulerConfig::enable_schedule_overlap,
                               &SchedulerConfig::enable_schedule_overlap},
    SchedulerConfigField<double>{
        "prefill_scheduling_memory_usage_threshold",
        &FLAGS_prefill_scheduling_memory_usage_threshold,
        &SchedulerConfigInput::prefill_scheduling_memory_usage_threshold,
        &SchedulerConfig::prefill_scheduling_memory_usage_threshold,
        &SchedulerConfig::prefill_scheduling_memory_usage_threshold},
    SchedulerConfigField<bool>{"enable_chunked_prefill",
                               &FLAGS_enable_chunked_prefill,
                               &SchedulerConfigInput::enable_chunked_prefill,
                               &SchedulerConfig::enable_chunked_prefill,
                               &SchedulerConfig::enable_chunked_prefill},
    SchedulerConfigField<int32_t>{
        "max_tokens_per_chunk_for_prefill",
        &FLAGS_max_tokens_per_chunk_for_prefill,
        &SchedulerConfigInput::max_tokens_per_chunk_for_prefill,
        &SchedulerConfig::max_tokens_per_chunk_for_prefill,
        &SchedulerConfig::max_tokens_per_chunk_for_prefill},
    SchedulerConfigField<int32_t>{
        "chunked_match_frequency",
        &FLAGS_chunked_match_frequency,
        &SchedulerConfigInput::chunked_match_frequency,
        &SchedulerConfig::chunked_match_frequency,
        &SchedulerConfig::chunked_match_frequency},
    SchedulerConfigField<std::string>{"priority_strategy",
                                      &FLAGS_priority_strategy,
                                      &SchedulerConfigInput::priority_strategy,
                                      &SchedulerConfig::priority_strategy,
                                      &SchedulerConfig::priority_strategy},
    SchedulerConfigField<bool>{"enable_mix_batch",
                               &FLAGS_enable_mix_batch,
                               &SchedulerConfigInput::enable_mix_batch,
                               &SchedulerConfig::enable_mix_batch,
                               &SchedulerConfig::enable_mix_batch},
    SchedulerConfigField<double>{"aggressive_coeff",
                                 &FLAGS_aggressive_coeff,
                                 &SchedulerConfigInput::aggressive_coeff,
                                 &SchedulerConfig::aggressive_coeff,
                                 &SchedulerConfig::aggressive_coeff},
    SchedulerConfigField<double>{"starve_threshold",
                                 &FLAGS_starve_threshold,
                                 &SchedulerConfigInput::starve_threshold,
                                 &SchedulerConfig::starve_threshold,
                                 &SchedulerConfig::starve_threshold},
    SchedulerConfigField<bool>{"enable_starve_prevent",
                               &FLAGS_enable_starve_prevent,
                               &SchedulerConfigInput::enable_starve_prevent,
                               &SchedulerConfig::enable_starve_prevent,
                               &SchedulerConfig::enable_starve_prevent},
    SchedulerConfigField<bool>{
        "enable_dp_fair_token_budget",
        &FLAGS_enable_dp_fair_token_budget,
        &SchedulerConfigInput::enable_dp_fair_token_budget,
        &SchedulerConfig::enable_dp_fair_token_budget,
        &SchedulerConfig::enable_dp_fair_token_budget},
    SchedulerConfigField<int32_t>{"request_queue_size",
                                  &FLAGS_request_queue_size,
                                  &SchedulerConfigInput::request_queue_size,
                                  &SchedulerConfig::request_queue_size,
                                  &SchedulerConfig::request_queue_size}};

template <typename Visitor>
void visit_scheduler_fields(Visitor&& visit) {
  std::apply([&](const auto&... field) { (visit(field), ...); },
             kSchedulerFields);
}

}  // namespace

SchedulerConfigInput scheduler_config_input_from_flags(
    const config::FlagInputSnapshot& flags) {
  SchedulerConfigInput input;
  visit_scheduler_fields([&](const auto& field) {
    if (flags.is_specified(field.name)) {
      input.*field.input = {*field.flag, config::ConfigInputSource::GFLAGS};
    }
  });
  return input;
}

void merge_scheduler_config_json(SchedulerConfigInput& input,
                                 const JsonReader& json) {
  visit_scheduler_fields([&](const auto& field) {
    auto& requested = input.*field.input;
    if (requested.value.has_value()) {
      return;
    }
    using T = typename std::decay_t<decltype(field)>::Value;
    if (auto value = json.value<T>(field.name)) {
      requested = {std::move(value), config::ConfigInputSource::JSON};
    }
  });
}

SchedulerConfig resolve_scheduler_config(const SchedulerConfigInput& input) {
  SchedulerConfig resolved;
  visit_scheduler_fields([&](const auto& field) {
    const auto& requested = input.*field.input;
    if (requested.value.has_value()) {
      (resolved.*field.set)(*requested.value);
    }
  });
  return resolved;
}

void SchedulerConfig::from_flags() {
  visit_scheduler_fields(
      [&](const auto& field) { (this->*field.set)(*field.flag); });
}

void SchedulerConfig::from_json(const JsonReader& json) {
  // Compatibility adapter for callers that still use from_flags/from_json.
  // Startup captures requested values before any JSON writeback instead.
  const auto flags = config::FlagInputSnapshot::capture();
  visit_scheduler_fields([&](const auto& field) {
    if (flags.is_specified(field.name)) {
      return;
    }
    using T = typename std::decay_t<decltype(field)>::Value;
    if (auto value = json.value<T>(field.name)) {
      (this->*field.set)(*value);
    }
    *field.flag = (this->*field.get)();
  });
}

void SchedulerConfig::append_config_json(
    nlohmann::ordered_json& config_json) const {
  const SchedulerConfig default_config;
  visit_scheduler_fields([&](const auto& field) {
    const auto& value = (this->*field.get)();
    if (value != (default_config.*field.get)()) {
      config_json[field.name] = value;
    }
  });
}

SchedulerConfig& SchedulerConfig::get_instance() {
  static SchedulerConfig config;
  return config;
}

void SchedulerConfig::initialize() {
  auto input =
      scheduler_config_input_from_flags(config::FlagInputSnapshot::capture());
  if (const auto& json_config = config::get_parsed_json_config()) {
    merge_scheduler_config_json(input, *json_config);
  }
  initialize(input);
}

void SchedulerConfig::initialize(const SchedulerConfigInput& input) {
  *this = resolve_scheduler_config(input);
  // Legacy consumers still read FLAGS_; keep this one-way compatibility
  // writeback at application time, outside input parsing and resolution.
  visit_scheduler_fields(
      [&](const auto& field) { *field.flag = (this->*field.get)(); });
}

}  // namespace xllm
