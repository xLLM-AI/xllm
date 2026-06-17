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

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

#include "core/util/json_reader.h"

namespace xllm::config {

JsonReader load_json_file(const std::string& config_path);

JsonReader parse_json_string(std::string_view config_json);

const std::optional<JsonReader>& get_parsed_json_config();

bool is_flag_specified(const char* flag_name);

void dump_startup_config();

}  // namespace xllm::config

#define APPEND_JSON_VALUE_IF_NOT_DEFAULT(       \
    config_json, key, value, default_value)     \
  do {                                          \
    const auto& config_json_value = (value);    \
    if (config_json_value != (default_value)) { \
      (config_json)[key] = config_json_value;   \
    }                                           \
  } while (false)

#define APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT( \
    config_json, default_config, property)       \
  APPEND_JSON_VALUE_IF_NOT_DEFAULT(              \
      config_json, #property, property(), (default_config).property())

#define XLLM_CONFIG_ASSIGN_FROM_FLAG(property) \
  do {                                         \
    property(FLAGS_##property);                \
  } while (false)

#define XLLM_CONFIG_ASSIGN_FROM_JSON(property)                               \
  do {                                                                       \
    property(json.value_or<std::decay_t<decltype(property())>>(#property,    \
                                                               property())); \
  } while (false)
