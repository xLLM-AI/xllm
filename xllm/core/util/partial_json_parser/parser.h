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

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/util/partial_json_parser/options.h"

namespace partial_json_parser {

class MalformedJSONException final : public std::runtime_error {
 public:
  explicit MalformedJSONException(const std::string& message)
      : std::runtime_error(message) {}
};

class JsonCompletion final {
 public:
  int32_t index;
  std::string string;

  JsonCompletion(int32_t idx = 0, std::string str = "")
      : index(idx), string(std::move(str)) {}
};

std::string parse_malformed_string(const std::string& malformed,
                                   TypeOptions options,
                                   bool format = false);

std::string parse_json(const std::string& json_string, TypeOptions allowed);
JsonCompletion complete_any(const std::string& json_string,
                            TypeOptions allowed,
                            bool top_level);
JsonCompletion complete_string(const std::string& json_string,
                               TypeOptions allowed);
JsonCompletion complete_array(const std::string& json_string,
                              TypeOptions allowed);
JsonCompletion complete_object(const std::string& json_string,
                               TypeOptions allowed);
JsonCompletion complete_number(const std::string& json_string,
                               TypeOptions allowed,
                               bool top_level);

int32_t skip_blank(const std::string& text, int32_t index);
std::string format_json(const std::string& json_string);

}  // namespace partial_json_parser
