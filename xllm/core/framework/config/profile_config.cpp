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

#include "core/framework/config/profile_config.h"

#include <limits>

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"

DEFINE_bool(enable_profile_step_time,
            false,
            "Whether to enable profile step time.");

DEFINE_bool(enable_profile_token_budget,
            false,
            "Whether to enable profile token budget.");

DEFINE_bool(enable_latency_aware_schedule,
            false,
            "use predicted latency for latency aware schedule.");

DEFINE_int32(profile_max_prompt_length,
             2048,
             "The max prompt length for profile.");

DEFINE_int32(max_global_ttft_ms,
             std::numeric_limits<int32_t>::max(),
             "all requests use single global ttft");

DEFINE_int32(max_global_tpot_ms,
             std::numeric_limits<int32_t>::max(),
             "all requests use single global ttft");

DEFINE_bool(enable_profile_kv_blocks,
            true,
            "true if generate kv cache for profile");

DEFINE_bool(disable_ttft_profiling,
            false,
            "Whether to disable TTFT profiling.");

DEFINE_bool(enable_forward_interruption,
            false,
            "Whether to enable forward interruption.");

DEFINE_bool(enable_online_profile,
            false,
            "Whether to enable the online timeline profiling endpoints "
            "(/start_profile and /stop_profile). CUDA only for now; pair with "
            "launching the server under nsys "
            "--capture-range=cudaProfilerApi.");

DEFINE_string(
    profile_backend,
    "torch",
    "Online profiling backend: 1: 'torch' (default) records CPU+CUDA "
    "activities in-process and writes a Chrome trace on "
    "/stop_profile, no external profiler needed; 2: 'cuda' only toggles "
    "the CUDA profiler capture range and requires launching under "
    "nsys --capture-range=cudaProfilerApi.");

DEFINE_string(profile_dir,
              "",
              "Directory the 'torch' online profiling backend writes timeline "
              "traces to. Empty means the current working directory.");

namespace xllm {

void ProfileConfig::from_flags() {
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_profile_step_time);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_profile_token_budget);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_latency_aware_schedule);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(profile_max_prompt_length);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(max_global_ttft_ms);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(max_global_tpot_ms);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_profile_kv_blocks);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(disable_ttft_profiling);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_forward_interruption);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_online_profile);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(profile_backend);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(profile_dir);
}

void ProfileConfig::from_json(const JsonReader& json) {
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_profile_step_time);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_profile_token_budget);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_latency_aware_schedule);
  XLLM_CONFIG_ASSIGN_FROM_JSON(profile_max_prompt_length);
  XLLM_CONFIG_ASSIGN_FROM_JSON(max_global_ttft_ms);
  XLLM_CONFIG_ASSIGN_FROM_JSON(max_global_tpot_ms);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_profile_kv_blocks);
  XLLM_CONFIG_ASSIGN_FROM_JSON(disable_ttft_profiling);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_forward_interruption);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_online_profile);
  XLLM_CONFIG_ASSIGN_FROM_JSON(profile_backend);
  XLLM_CONFIG_ASSIGN_FROM_JSON(profile_dir);
}

void ProfileConfig::append_config_json(
    nlohmann::ordered_json& config_json) const {
  const ProfileConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_profile_step_time);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_profile_token_budget);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_latency_aware_schedule);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, profile_max_prompt_length);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, max_global_ttft_ms);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, max_global_tpot_ms);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_profile_kv_blocks);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, disable_ttft_profiling);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_forward_interruption);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_online_profile);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, profile_backend);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, profile_dir);
}

ProfileConfig& ProfileConfig::get_instance() {
  static ProfileConfig config;
  return config;
}

void ProfileConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
}

}  // namespace xllm
