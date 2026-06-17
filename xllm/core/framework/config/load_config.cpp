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

#include "core/framework/config/load_config.h"

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"

DEFINE_bool(enable_manual_loader,
            false,
            "Pin decoder layer weights to host memory and use async H2D "
            "transfer. Required by enable_rolling_load; also implied by "
            "enable_xtensor.");

DEFINE_bool(enable_rolling_load,
            false,
            "Enable rolling weight load: keep only N decoder layer weight "
            "slots in HBM and stream-load each layer just-in-time. "
            "Requires enable_manual_loader=true. NPU only.");

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

DEFINE_bool(
    enable_prefetch_weight,
    false,
    "Whether to enable prefetch weight,only applicable to Qwen3-dense model."
    "The default prefetching ratio for gateup weight is 40%."
    "If adjustments are needed, e.g. export PREFETCH_COEFFOCIENT=0.5");

namespace xllm {

void LoadConfig::from_flags() {
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_manual_loader);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_rolling_load);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(rolling_load_num_cached_layers);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(rolling_load_num_rolling_slots);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_prefetch_weight);
}

void LoadConfig::from_json(const JsonReader& json) {
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_manual_loader);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_rolling_load);
  XLLM_CONFIG_ASSIGN_FROM_JSON(rolling_load_num_cached_layers);
  XLLM_CONFIG_ASSIGN_FROM_JSON(rolling_load_num_rolling_slots);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_prefetch_weight);
}

void LoadConfig::append_config_json(nlohmann::ordered_json& config_json) const {
  const LoadConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_manual_loader);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_rolling_load);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, rolling_load_num_cached_layers);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, rolling_load_num_rolling_slots);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_prefetch_weight);
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
