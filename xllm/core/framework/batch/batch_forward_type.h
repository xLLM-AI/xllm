/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

namespace xllm {

class BatchForwardType {
 public:
  enum Value : int32_t {
    // Prefill without using kv cache.
    PREFILL = 0,
    // Chunked prefill using kv cache.
    // No decode sequence in this type.
    CHUNKED_PREFILL = 1,
    // Decode one token.
    // No prefill sequence in this type.
    DECODE = 2,
    // Mixed prefill and decode in one batch when doing chunked prefill.
    MIXED = 3,
    // No sequence to forward.
    EMPTY = 4,
  };

  BatchForwardType() : value_(EMPTY) {}

  BatchForwardType(int32_t v) : value_(static_cast<Value>(v)) {}

  constexpr BatchForwardType(Value v) : value_(v) {}

  BatchForwardType& operator=(Value v) {
    value_ = v;
    return *this;
  }

  constexpr int32_t value() const { return value_; }

  constexpr bool is_prefill() const { return (value_ == PREFILL); }

  constexpr bool is_chunked_prefill() const {
    return (value_ == CHUNKED_PREFILL);
  }

  constexpr bool no_decode() const {
    return (value_ == PREFILL || value_ == CHUNKED_PREFILL);
  }

  constexpr bool has_decode() const {
    return (value_ == DECODE || value_ == MIXED);
  }

  constexpr bool is_decode() const { return (value_ == DECODE); }

  constexpr bool is_mixed() const { return (value_ == MIXED); }

  constexpr bool is_empty() const { return (value_ == EMPTY); }

  std::string to_string() const {
    switch (value_) {
      case PREFILL:
        return "PREFILL";
      case CHUNKED_PREFILL:
        return "CHUNKED_PREFILL";
      case DECODE:
        return "DECODE";
      case MIXED:
        return "MIXED";
      case EMPTY:
        return "EMPTY";
      default:
        return "UNKNOWN";
    }
  }

 private:
  Value value_;
};
}  // namespace xllm
