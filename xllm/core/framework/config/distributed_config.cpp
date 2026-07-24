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

#include "core/framework/config/distributed_config.h"

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"
#include "core/util/env_var.h"

DEFINE_string(master_node_addr,
              "",
              "The master address for multi-node distributed serving(e.g. "
              "10.18.1.1:9999). Required for multi-process/multi-node serving; "
              "leave empty only when XLLM_ENABLE_SINGLE_PROCESS is set.");

DEFINE_string(
    xtensor_master_node_addr,
    "127.0.0.1:19889",
    "The master address for XTensor distributed service(e.g. 10.18.1.1:9999).");

DEFINE_int32(nnodes, 1, "The number of multi-nodes.");

DEFINE_int32(node_rank, 0, "The node rank.");

DEFINE_string(etcd_addr, "", "Etcd adderss for save instance meta info.");

DEFINE_string(etcd_namespace,
              "",
              "Optional etcd namespace prefix for all xllm keys, e.g. prod-a.");

DEFINE_bool(enable_service_routing,
            false,
            "Whether to use xllm service routing.");

DEFINE_double(heart_beat_interval, 0.5, "Heart beat interval.");

DEFINE_int32(etcd_ttl, 3, "Time to live for etcd.");

namespace xllm {

void DistributedConfig::from_flags() {
  XLLM_CONFIG_ASSIGN_FROM_FLAG(master_node_addr);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(xtensor_master_node_addr);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(nnodes);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(node_rank);
  // Single-process mode is opt-in via environment variable rather than a
  // gflag, so it can be set from the launch environment without threading a
  // CLI flag through every entrypoint.
  enable_single_process(util::get_bool_env("XLLM_ENABLE_SINGLE_PROCESS",
                                           enable_single_process()));
  XLLM_CONFIG_ASSIGN_FROM_FLAG(etcd_addr);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(etcd_namespace);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_service_routing);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(heart_beat_interval);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(etcd_ttl);
}

void DistributedConfig::from_json(const JsonReader& json) {
  XLLM_CONFIG_ASSIGN_FROM_JSON(master_node_addr);
  XLLM_CONFIG_ASSIGN_FROM_JSON(xtensor_master_node_addr);
  XLLM_CONFIG_ASSIGN_FROM_JSON(nnodes);
  // don't read rank-related config
  // XLLM_CONFIG_ASSIGN_FROM_JSON(node_rank);
  // enable_single_process is sourced from the environment, not JSON.
  XLLM_CONFIG_ASSIGN_FROM_JSON(etcd_addr);
  XLLM_CONFIG_ASSIGN_FROM_JSON(etcd_namespace);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_service_routing);
  XLLM_CONFIG_ASSIGN_FROM_JSON(heart_beat_interval);
  XLLM_CONFIG_ASSIGN_FROM_JSON(etcd_ttl);
}

void DistributedConfig::append_config_json(
    nlohmann::ordered_json& config_json) const {
  const DistributedConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, master_node_addr);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, xtensor_master_node_addr);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(config_json, default_config, nnodes);
  // don't dump rank-related config
  //   APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
  //       config_json, default_config, node_rank);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_single_process);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, etcd_addr);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, etcd_namespace);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_service_routing);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, heart_beat_interval);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, etcd_ttl);
}

DistributedConfig& DistributedConfig::get_instance() {
  static DistributedConfig config;
  return config;
}

void DistributedConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
}

}  // namespace xllm
