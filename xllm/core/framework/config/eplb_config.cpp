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

#include "core/framework/config/eplb_config.h"

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"

DEFINE_bool(enable_eplb, false, "Whether to use expert parallel load balance.");

DEFINE_int32(redundant_experts_num,
             1,
             "Number of redundant experts on per device.");

DEFINE_int64(eplb_update_interval, 1000, "EPLB update rate.");

DEFINE_double(eplb_update_threshold, 0.8, "EPLB update threshold.");

DEFINE_int32(expert_parallel_degree, 0, "Expert parallel degree.");

DEFINE_string(rank_tablefile, "", "ATB HCCL rank table file.");

namespace xllm {

void EPLBConfig::from_flags() {
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_eplb);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(redundant_experts_num);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(eplb_update_interval);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(eplb_update_threshold);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(expert_parallel_degree);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(rank_tablefile);
}

void EPLBConfig::from_json(const JsonReader& json) {
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_eplb);
  XLLM_CONFIG_ASSIGN_FROM_JSON(redundant_experts_num);
  XLLM_CONFIG_ASSIGN_FROM_JSON(eplb_update_interval);
  XLLM_CONFIG_ASSIGN_FROM_JSON(eplb_update_threshold);
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
      config_json, default_config, eplb_update_threshold);
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
