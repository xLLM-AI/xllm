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

#include "core/common/macros.h"
#include "core/framework/config/option_category.h"

namespace xllm {

class JsonReader;

class LoadConfig final {
 public:
  LoadConfig() = default;
  ~LoadConfig() = default;

  static LoadConfig& get_instance();

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "LOAD OPTIONS",
        {"enable_manual_loader",
         "enable_rolling_load",
         "rolling_load_num_cached_layers",
         "rolling_load_num_rolling_slots",
         "enable_prefetch_weight"}};
    return kOptionCategory;
  }

  PROPERTY(bool, enable_manual_loader) = false;

  PROPERTY(bool, enable_rolling_load) = false;

  PROPERTY(int32_t, rolling_load_num_cached_layers) = 2;

  PROPERTY(int32_t, rolling_load_num_rolling_slots) = -1;

  PROPERTY(bool, enable_prefetch_weight) = false;
};

}  // namespace xllm
