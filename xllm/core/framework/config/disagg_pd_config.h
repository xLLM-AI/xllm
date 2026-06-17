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

#pragma once

#include <cstdint>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "core/common/macros.h"
#include "core/framework/config/option_category.h"

namespace xllm {

class JsonReader;
class KVCacheConfig;
class SchedulerConfig;

class DisaggPDConfig final {
 public:
  DisaggPDConfig() = default;
  ~DisaggPDConfig() = default;

  static DisaggPDConfig& get_instance();

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();
  void normalize_mlu(KVCacheConfig& kv_cache_config,
                     SchedulerConfig& scheduler_config);

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "DISAGGREGATED PREFILL-DECODE OPTIONS",
        {"enable_disagg_pd",
         "enable_pd_ooc",
         "disagg_pd_port",
         "instance_role",
         "kv_cache_transfer_type",
         "kv_cache_transfer_mode",
         "transfer_listen_port"}};
    return kOptionCategory;
  }

  PROPERTY(bool, enable_disagg_pd) = false;

  PROPERTY(bool, enable_pd_ooc) = false;

  PROPERTY(int32_t, disagg_pd_port) = 7777;

  PROPERTY(std::string, instance_role) = "DEFAULT";

  PROPERTY(std::string, kv_cache_transfer_type) = "LlmDataDist";

  PROPERTY(std::string, kv_cache_transfer_mode) = "PUSH";

  PROPERTY(int32_t, transfer_listen_port) = 26000;

  PROPERTY(bool, kv_push_dst_rotate) = false;
};

}  // namespace xllm
