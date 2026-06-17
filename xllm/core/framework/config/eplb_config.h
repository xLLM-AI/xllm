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

class EPLBConfig final {
 public:
  EPLBConfig() = default;
  ~EPLBConfig() = default;

  static EPLBConfig& get_instance();

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {"EP LOAD BALANCE OPTIONS",
                                                   {"enable_eplb",
                                                    "redundant_experts_num",
                                                    "eplb_update_interval",
                                                    "eplb_update_threshold",
                                                    "expert_parallel_degree",
                                                    "rank_tablefile"}};
    return kOptionCategory;
  }

  PROPERTY(bool, enable_eplb) = false;

  PROPERTY(int32_t, redundant_experts_num) = 1;

  PROPERTY(int64_t, eplb_update_interval) = 1000;

  PROPERTY(double, eplb_update_threshold) = 0.8;

  PROPERTY(int32_t, expert_parallel_degree) = 0;

  PROPERTY(std::string, rank_tablefile);
};

}  // namespace xllm
