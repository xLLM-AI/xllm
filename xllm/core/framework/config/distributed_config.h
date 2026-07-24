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

#pragma once

#include <cstdint>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "core/common/macros.h"
#include "core/framework/config/option_category.h"

namespace xllm {

class JsonReader;

class DistributedConfig final {
 public:
  DistributedConfig() = default;
  ~DistributedConfig() = default;

  static DistributedConfig& get_instance();

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {"DISTRIBUTED OPTIONS",
                                                   {"master_node_addr",
                                                    "xtensor_master_node_addr",
                                                    "nnodes",
                                                    "node_rank",
                                                    "etcd_addr",
                                                    "etcd_namespace",
                                                    "enable_service_routing",
                                                    "heart_beat_interval",
                                                    "etcd_ttl"}};
    return kOptionCategory;
  }

  PROPERTY(std::string, master_node_addr) = "";

  PROPERTY(std::string, xtensor_master_node_addr) = "127.0.0.1:19889";

  PROPERTY(int32_t, nnodes) = 1;

  PROPERTY(int32_t, node_rank) = 0;

  // Run all local devices inside a single OS process (one worker thread per
  // device, sharing one HcclCommInitAll world) instead of one process per
  // device. Independent of master_node_addr: single-node single-process needs
  // no master_node_addr, while a future multi-node single-process deployment
  // would still use master_node_addr for inter-node rendezvous.
  PROPERTY(bool, enable_single_process) = false;

  PROPERTY(std::string, etcd_addr);

  PROPERTY(std::string, etcd_namespace);

  PROPERTY(bool, enable_service_routing) = false;

  PROPERTY(double, heart_beat_interval) = 0.5;

  PROPERTY(int32_t, etcd_ttl) = 3;
};

}  // namespace xllm
