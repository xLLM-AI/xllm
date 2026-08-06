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

#include "core/framework/config/eplb_config.h"

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

DEFINE_string(rank_tablefile, "", "ATB HCCL rank table file.");

namespace xllm {

void EPLBConfig::from_flags() {
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_eplb);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(redundant_experts_num);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(eplb_update_interval);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(eplb_min_peak_load_improvement);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(eplb_policy_kind);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(eplb_use_decode_only_load);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(eplb_prepare_timeout_seconds);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(expert_parallel_degree);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(rank_tablefile);
}

void EPLBConfig::from_json(const JsonReader& json) {
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_eplb);
  XLLM_CONFIG_ASSIGN_FROM_JSON(redundant_experts_num);
  XLLM_CONFIG_ASSIGN_FROM_JSON(eplb_update_interval);
  XLLM_CONFIG_ASSIGN_FROM_JSON(eplb_min_peak_load_improvement);
  XLLM_CONFIG_ASSIGN_FROM_JSON(eplb_policy_kind);
  XLLM_CONFIG_ASSIGN_FROM_JSON(eplb_use_decode_only_load);
  XLLM_CONFIG_ASSIGN_FROM_JSON(eplb_prepare_timeout_seconds);
  XLLM_CONFIG_ASSIGN_FROM_JSON(expert_parallel_degree);
  XLLM_CONFIG_ASSIGN_FROM_JSON(rank_tablefile);
}

void EPLBConfig::append_config_json(nlohmann::ordered_json& config_json) const {
  const EPLBConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_eplb);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, redundant_experts_num);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, eplb_update_interval);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, eplb_min_peak_load_improvement);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, eplb_policy_kind);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, eplb_use_decode_only_load);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, eplb_prepare_timeout_seconds);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, expert_parallel_degree);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, rank_tablefile);
}

EPLBConfig& EPLBConfig::get_instance() {
  static EPLBConfig config;
  return config;
}

void EPLBConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
}

}  // namespace xllm
