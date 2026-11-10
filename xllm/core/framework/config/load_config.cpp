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

#include "core/framework/config/load_config.h"

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"

DEFINE_bool(enable_vmm_weights,
            false,
            "Whether to enable virtual memory for model weights "
            "with the physical page pool. NPU only.");

DEFINE_bool(enable_rolling_load,
            false,
            "Enable rolling DiT weight load: keep only N layer weight "
            "slots in HBM and stream-load each layer just-in-time. "
            "NPU only.");

DEFINE_int32(rolling_load_num_cached_layers,
             2,
             "Number of decoder layer weight slots to keep in HBM when "
             "enable_rolling_load=true.");

DEFINE_int32(rolling_load_num_rolling_slots,
             -1,
             "Number of rolling slots used by decoder rolling load. "
             "Fixed slots are computed as "
             "rolling_load_num_cached_layers - rolling_load_num_rolling_slots."
             " -1 means auto (min(2, preload_count)). "
             "Must be in [-1, rolling_load_num_cached_layers].");

namespace xllm {

void LoadConfig::from_flags() {
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_vmm_weights);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_rolling_load);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(rolling_load_num_cached_layers);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(rolling_load_num_rolling_slots);
}

void LoadConfig::from_json(const JsonReader& json) {
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_vmm_weights);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_rolling_load);
  XLLM_CONFIG_ASSIGN_FROM_JSON(rolling_load_num_cached_layers);
  XLLM_CONFIG_ASSIGN_FROM_JSON(rolling_load_num_rolling_slots);
}

void LoadConfig::append_config_json(nlohmann::ordered_json& config_json) const {
  const LoadConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_vmm_weights);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_rolling_load);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, rolling_load_num_cached_layers);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, rolling_load_num_rolling_slots);
}

LoadConfig& LoadConfig::get_instance() {
  static LoadConfig config;
  return config;
}

void LoadConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
}

}  // namespace xllm
