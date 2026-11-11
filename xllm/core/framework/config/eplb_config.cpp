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

#include "core/framework/config/eplb_config.h"

#include <tuple>
#include <type_traits>
#include <utility>

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"

DEFINE_bool(enable_eplb, false, "Whether to use expert parallel load balance.");

DEFINE_int32(redundant_experts_num,
             1,
             "Number of redundant experts on per device.");

DEFINE_int64(eplb_update_interval, 1000, "EPLB update rate.");

DEFINE_double(eplb_min_peak_load_improvement,
              0.05,
              "Minimum peak-rank load reduction required before any EPLB "
              "policy publishes a new placement. Must "
              "be in [0, 1].");

DEFINE_string(
    eplb_policy_kind,
    "balanced",
    "EPLB rebalance policy kind: 'balanced' (default) or 'greedy'. "
    "Historical policy names remain accepted as compatibility aliases. "
    "Unknown values silently fall back to 'greedy'.");

DEFINE_bool(eplb_use_decode_only_load,
            false,
            "When true, only real decode tokens contribute to expert_load; "
            "prefill rows in mixed batches and graph padding are filtered. "
            "Off by default so behavior is unchanged unless operators opt in.");

DEFINE_int32(eplb_prepare_timeout_seconds,
             30,
             "Manager-thread backstop timeout in seconds when devices fail to "
             "report a layer prepared (worker crash mid-prepare, weight "
             "provider unwired). After this many seconds waiting on the same "
             "layer, the manager warns and skips to the next layer so the "
             "rebalance loop keeps progressing.");

DEFINE_int32(expert_parallel_degree, 0, "Expert parallel degree.");

DEFINE_string(rank_tablefile, "", "HCCL rank table file.");

namespace xllm {
namespace {

template <typename T>
struct EplbConfigField {
  using Value = T;
  const char* name;
  T* flag;
  config::ConfigInput<T> EplbConfigInput::* input;
  const T& (EPLBConfig::*get)() const& noexcept;
  EPLBConfig& (EPLBConfig::*set)(const T&)&;
};

// Preserve the existing adapter and dump order in one typed mapping.
const auto kEplbFields = std::tuple{
    EplbConfigField<bool>{"enable_eplb",
                          &FLAGS_enable_eplb,
                          &EplbConfigInput::enable_eplb,
                          &EPLBConfig::enable_eplb,
                          &EPLBConfig::enable_eplb},
    EplbConfigField<int32_t>{"redundant_experts_num",
                             &FLAGS_redundant_experts_num,
                             &EplbConfigInput::redundant_experts_num,
                             &EPLBConfig::redundant_experts_num,
                             &EPLBConfig::redundant_experts_num},
    EplbConfigField<int64_t>{"eplb_update_interval",
                             &FLAGS_eplb_update_interval,
                             &EplbConfigInput::eplb_update_interval,
                             &EPLBConfig::eplb_update_interval,
                             &EPLBConfig::eplb_update_interval},
    EplbConfigField<double>{"eplb_min_peak_load_improvement",
                            &FLAGS_eplb_min_peak_load_improvement,
                            &EplbConfigInput::eplb_min_peak_load_improvement,
                            &EPLBConfig::eplb_min_peak_load_improvement,
                            &EPLBConfig::eplb_min_peak_load_improvement},
    EplbConfigField<std::string>{"eplb_policy_kind",
                                 &FLAGS_eplb_policy_kind,
                                 &EplbConfigInput::eplb_policy_kind,
                                 &EPLBConfig::eplb_policy_kind,
                                 &EPLBConfig::eplb_policy_kind},
    EplbConfigField<bool>{"eplb_use_decode_only_load",
                          &FLAGS_eplb_use_decode_only_load,
                          &EplbConfigInput::eplb_use_decode_only_load,
                          &EPLBConfig::eplb_use_decode_only_load,
                          &EPLBConfig::eplb_use_decode_only_load},
    EplbConfigField<int32_t>{"eplb_prepare_timeout_seconds",
                             &FLAGS_eplb_prepare_timeout_seconds,
                             &EplbConfigInput::eplb_prepare_timeout_seconds,
                             &EPLBConfig::eplb_prepare_timeout_seconds,
                             &EPLBConfig::eplb_prepare_timeout_seconds},
    EplbConfigField<int32_t>{"expert_parallel_degree",
                             &FLAGS_expert_parallel_degree,
                             &EplbConfigInput::expert_parallel_degree,
                             &EPLBConfig::expert_parallel_degree,
                             &EPLBConfig::expert_parallel_degree},
    EplbConfigField<std::string>{"rank_tablefile",
                                 &FLAGS_rank_tablefile,
                                 &EplbConfigInput::rank_tablefile,
                                 &EPLBConfig::rank_tablefile,
                                 &EPLBConfig::rank_tablefile}};

template <typename Visitor>
void visit_eplb_fields(Visitor&& visit) {
  std::apply([&](const auto&... field) { (visit(field), ...); }, kEplbFields);
}

}  // namespace

EplbConfigInput eplb_config_input_from_flags(
    const config::FlagInputSnapshot& flags) {
  EplbConfigInput input;
  visit_eplb_fields([&](const auto& field) {
    if (flags.is_specified(field.name)) {
      input.*field.input = {*field.flag, config::ConfigInputSource::GFLAGS};
    }
  });
  return input;
}

void merge_eplb_config_json(EplbConfigInput& input, const JsonReader& json) {
  visit_eplb_fields([&](const auto& field) {
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

EPLBConfig resolve_eplb_config(const EplbConfigInput& input) {
  EPLBConfig resolved;
  visit_eplb_fields([&](const auto& field) {
    const auto& requested = input.*field.input;
    if (requested.value.has_value()) {
      (resolved.*field.set)(*requested.value);
    }
  });
  return resolved;
}

void EPLBConfig::from_flags() {
  visit_eplb_fields(
      [&](const auto& field) { (this->*field.set)(*field.flag); });
}

void EPLBConfig::from_json(const JsonReader& json) {
  // Legacy callers retain the from_flags/from_json contract. Startup instead
  // copies typed requested values before any configuration writes flags back.
  const auto flags = config::FlagInputSnapshot::capture();
  visit_eplb_fields([&](const auto& field) {
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

void EPLBConfig::append_config_json(nlohmann::ordered_json& config_json) const {
  const EPLBConfig default_config;
  visit_eplb_fields([&](const auto& field) {
    const auto& value = (this->*field.get)();
    if (value != (default_config.*field.get)()) {
      config_json[field.name] = value;
    }
  });
}

EPLBConfig& EPLBConfig::get_instance() {
  static EPLBConfig config;
  return config;
}

void EPLBConfig::initialize() {
  auto input =
      eplb_config_input_from_flags(config::FlagInputSnapshot::capture());
  if (const auto& json_config = config::get_parsed_json_config()) {
    merge_eplb_config_json(input, *json_config);
  }
  initialize(input);
}

void EPLBConfig::initialize(const EplbConfigInput& input) {
  *this = resolve_eplb_config(input);
  // Preserve process-global consumers only at the explicit application
  // boundary.
  visit_eplb_fields(
      [&](const auto& field) { *field.flag = (this->*field.get)(); });
}

}  // namespace xllm
