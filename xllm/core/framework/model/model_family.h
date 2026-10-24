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
#include <string_view>

namespace xllm {

class ModelFamily final {
 public:
  enum Value : uint8_t {
    UNKNOWN = 0,
    LLM = 1,
    VLM = 2,
    REC = 3,
    DIT = 4,
  };

  constexpr ModelFamily() noexcept = default;

  // Preserve enum-style initialization, as with EngineType and MMType.
  constexpr ModelFamily(Value value) noexcept : value_(value) {}

  explicit constexpr ModelFamily(std::string_view family) noexcept {
    if (family == "llm") {
      value_ = LLM;
    } else if (family == "vlm") {
      value_ = VLM;
    } else if (family == "rec") {
      value_ = REC;
    } else if (family == "dit") {
      value_ = DIT;
    }
  }

  constexpr operator Value() const noexcept { return value_; }
  explicit operator bool() const = delete;

  constexpr bool operator==(ModelFamily rhs) const noexcept {
    return value_ == rhs.value_;
  }
  constexpr bool operator!=(ModelFamily rhs) const noexcept {
    return value_ != rhs.value_;
  }
  constexpr bool operator==(Value rhs) const noexcept { return value_ == rhs; }
  constexpr bool operator!=(Value rhs) const noexcept { return value_ != rhs; }

  constexpr const char* to_string() const noexcept {
    switch (value_) {
      case LLM:
        return "llm";
      case VLM:
        return "vlm";
      case REC:
        return "rec";
      case DIT:
        return "dit";
      case UNKNOWN:
        return "unknown";
    }
    return "unknown";
  }

 private:
  Value value_ = UNKNOWN;
};

}  // namespace xllm
