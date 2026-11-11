/* Copyright 2026 The xLLM Authors.

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
#include <optional>
#include <string>
#include <unordered_set>

namespace xllm::config {

enum class ConfigInputSource : uint8_t { UNSPECIFIED, GFLAGS, JSON, API };

template <typename T>
struct ConfigInput {
  std::optional<T> value;
  ConfigInputSource source = ConfigInputSource::UNSPECIFIED;
};

class FlagInputSnapshot final {
 public:
  // Capture after gflags parsing and before resolved values are written back
  // to FLAGS_. Later gflags queries can classify those writes as user input.
  [[nodiscard]] static FlagInputSnapshot capture();

  [[nodiscard]] bool is_specified(const std::string& flag_name) const;

 private:
  FlagInputSnapshot() = default;

  std::unordered_set<std::string> specified_flags_;
};

}  // namespace xllm::config
