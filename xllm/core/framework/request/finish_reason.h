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
#include <optional>
#include <string>

namespace xllm {
class FinishReason {
 public:
  enum Value : uint8_t { NONE = 0, STOP = 1, LENGTH, FUNCTION_CALL };

  FinishReason() = default;
  FinishReason(Value v) : value(v) {}
  operator Value() const { return value; }
  explicit operator bool() const = delete;

  bool operator==(FinishReason rhs) const { return value == rhs.value; }
  bool operator!=(FinishReason rhs) const { return value != rhs.value; }

  bool operator==(Value v) const { return value == v; }
  bool operator!=(Value v) const { return value != v; }

  std::optional<std::string> to_string();

 private:
  Value value;
};
}  // namespace xllm
