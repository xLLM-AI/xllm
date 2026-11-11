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

#pragma once

#include <cstdint>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "core/common/macros.h"
#include "core/framework/config/config_input.h"
#include "core/framework/config/option_category.h"

namespace xllm {

class JsonReader;

// Requested values retain omission and source until legacy value resolution.
struct EplbConfigInput {
  config::ConfigInput<bool> enable_eplb;
  config::ConfigInput<int32_t> redundant_experts_num;
  config::ConfigInput<int64_t> eplb_update_interval;
  config::ConfigInput<double> eplb_min_peak_load_improvement;
  config::ConfigInput<std::string> eplb_policy_kind;
  config::ConfigInput<bool> eplb_use_decode_only_load;
  config::ConfigInput<int32_t> eplb_prepare_timeout_seconds;
  config::ConfigInput<int32_t> expert_parallel_degree;
  config::ConfigInput<std::string> rank_tablefile;
};

EplbConfigInput eplb_config_input_from_flags(
    const config::FlagInputSnapshot& flags);

// Explicit GFlags/API values take precedence over JSON.
void merge_eplb_config_json(EplbConfigInput& input, const JsonReader& json);

class EPLBConfig final {
 public:
  EPLBConfig() = default;
  ~EPLBConfig() = default;

  static EPLBConfig& get_instance();

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();
  void initialize(const EplbConfigInput& input);

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "EP LOAD BALANCE OPTIONS",
        {"enable_eplb",
         "redundant_experts_num",
         "eplb_update_interval",
         "eplb_min_peak_load_improvement",
         "eplb_policy_kind",
         "eplb_use_decode_only_load",
         "eplb_prepare_timeout_seconds",
         "expert_parallel_degree",
         "rank_tablefile"}};
    return kOptionCategory;
  }

  PROPERTY(bool, enable_eplb) = false;

  PROPERTY(int32_t, redundant_experts_num) = 1;

  PROPERTY(int64_t, eplb_update_interval) = 1000;

  // Minimum reduction in the measured peak rank load required before a policy
  // publishes a new placement. This gate is shared by all EPLB policies.
  PROPERTY(double, eplb_min_peak_load_improvement) = 0.05;

  // Selects the concrete EPLB rebalance strategy MakeEplbPolicy instantiates.
  // Accepted (case-insensitive) values: "balanced" (default,
  // max-load-reduction replica selection + strict equal-cardinality LPT
  // packing across the HCCS super-node) and "greedy" (historical xLLM
  // replica selection and LPT packing). Historical names remain aliases.
  // Unknown values silently fall back to "greedy" so a bad operator flag does
  // not knock the rebalance loop offline.
  PROPERTY(std::string, eplb_policy_kind) = "balanced";

  // When true, only real decode tokens contribute to expert_load. Pure
  // prefill, prefill rows inside mixed batches, and synthetic graph padding
  // are filtered by the per-token decode mask. Off by default so rebalance
  // behaviour is unchanged unless operators opt in.
  PROPERTY(bool, eplb_use_decode_only_load) = false;

  // Backstop for the manager thread when devices never finish preparing the
  // current EPLB layer (worker crash mid-prepare, weight_provider unwired,
  // etc). After this many seconds waiting on the same layer, the manager
  // logs a warning and skips to the next layer so the rebalance loop keeps
  // progressing. Slow weight_provider deployments may need to raise this;
  // the default matches the pre-config-flag hard-coded 30s.
  PROPERTY(int32_t, eplb_prepare_timeout_seconds) = 30;

  PROPERTY(int32_t, expert_parallel_degree) = 0;

  PROPERTY(std::string, rank_tablefile);
};

// Resolves values and defaults without global access or eager policy
// validation.
EPLBConfig resolve_eplb_config(const EplbConfigInput& input);

}  // namespace xllm
