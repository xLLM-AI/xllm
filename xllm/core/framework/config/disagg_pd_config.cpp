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

#include "core/framework/config/disagg_pd_config.h"

#include <glog/logging.h>

#include <string>

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/scheduler_config.h"

DEFINE_bool(enable_disagg_pd,
            false,
            "Whether to enable disaggregated prefill and decode execution.");

DEFINE_bool(
    enable_pd_ooc,
    false,
    "Whether to enable online-offline co-location in disaggregated PD mode.");

DEFINE_int32(disagg_pd_port, 7777, "Port for brpc disagg pd server.");

DEFINE_string(instance_role,
              "DEFAULT",
              "The role of instance(e.g. DEFAULT, PREFILL, DECODE, MIX).");

DEFINE_string(
    kv_cache_transfer_type,
    "LlmDataDist",
    "The type of kv cache transfer(e.g. LlmDataDist, Mooncake, HCCL).");

DEFINE_string(kv_cache_transfer_mode,
              "PUSH",
              "The mode of kv cache transfer(e.g. PUSH, PULL).");

DEFINE_int32(transfer_listen_port, 26000, "The KVCacheTranfer listen port.");

DEFINE_bool(kv_push_dst_rotate,
            false,
            "Rotate the dst-worker traversal order in push_kv_blocks per "
            "KV-split rank to spread incast across D workers.");

namespace xllm {
namespace {

bool supports_prefix_cache(const std::string& instance_role) {
  return instance_role == "PREFILL" || instance_role == "MIX";
}

}  // namespace

void DisaggPDConfig::from_flags() {
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_disagg_pd);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_pd_ooc);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(disagg_pd_port);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(instance_role);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(kv_cache_transfer_type);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(kv_cache_transfer_mode);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(transfer_listen_port);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(kv_push_dst_rotate);
}

void DisaggPDConfig::from_json(const JsonReader& json) {
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_disagg_pd);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_pd_ooc);
  XLLM_CONFIG_ASSIGN_FROM_JSON(disagg_pd_port);
  XLLM_CONFIG_ASSIGN_FROM_JSON(instance_role);
  XLLM_CONFIG_ASSIGN_FROM_JSON(kv_cache_transfer_type);
  XLLM_CONFIG_ASSIGN_FROM_JSON(kv_cache_transfer_mode);
  XLLM_CONFIG_ASSIGN_FROM_JSON(transfer_listen_port);
}

void DisaggPDConfig::append_config_json(
    nlohmann::ordered_json& config_json) const {
  const DisaggPDConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_disagg_pd);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_pd_ooc);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, disagg_pd_port);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, instance_role);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, kv_cache_transfer_type);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, kv_cache_transfer_mode);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, transfer_listen_port);
}

DisaggPDConfig& DisaggPDConfig::get_instance() {
  static DisaggPDConfig config;
  return config;
}

void DisaggPDConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
}

void DisaggPDConfig::normalize_mlu(KVCacheConfig& kv_cache_config,
                                   SchedulerConfig& scheduler_config) {
  if (kv_cache_transfer_type() != "Mooncake") {
    LOG(WARNING) << "MLU disaggregated PD requires "
                 << "kv_cache_transfer_type=Mooncake; forcing from "
                 << kv_cache_transfer_type() << " to Mooncake.";
    kv_cache_transfer_type("Mooncake");
  }
  if (kv_cache_transfer_mode() != "PUSH") {
    LOG(WARNING) << "MLU disaggregated PD requires "
                 << "kv_cache_transfer_mode=PUSH; forcing from "
                 << kv_cache_transfer_mode() << " to PUSH.";
    kv_cache_transfer_mode("PUSH");
  }
  if (kv_cache_config.kv_cache_dtype() != "auto") {
    LOG(WARNING) << "MLU disaggregated PD requires kv_cache_dtype=auto; "
                 << "forcing from " << kv_cache_config.kv_cache_dtype()
                 << " to auto.";
    kv_cache_config.kv_cache_dtype("auto");
  }
  if (scheduler_config.enable_schedule_overlap()) {
    LOG(WARNING) << "MLU disaggregated PD does not support schedule overlap; "
                 << "forcing enable_schedule_overlap=false.";
    scheduler_config.enable_schedule_overlap(false);
  }
  if (kv_cache_config.enable_prefix_cache() &&
      !supports_prefix_cache(instance_role())) {
    LOG(WARNING) << "MLU disaggregated PD role " << instance_role()
                 << " does not support prefix cache; "
                 << "forcing enable_prefix_cache=false.";
    kv_cache_config.enable_prefix_cache(false);
  }
  if (enable_pd_ooc()) {
    LOG(WARNING) << "MLU disaggregated PD does not support pd_ooc; "
                 << "forcing enable_pd_ooc=false.";
    enable_pd_ooc(false);
  }
}

}  // namespace xllm
