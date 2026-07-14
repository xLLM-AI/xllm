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
#include <string>

namespace xllm {

class KVCacheTensorRole {
 public:
  enum Value : int8_t {
    KEY = 0,
    VALUE = 1,
    INDEX = 2,
    CONV = 3,
    SSM = 4,
    INDEX_SCALE = 5,
    INVALID = -1,
  };

  constexpr KVCacheTensorRole(Value v) : value_(v) {}
  KVCacheTensorRole(const std::string& str) {
    if (str == "KEY" || str == "key") {
      value_ = KEY;
    } else if (str == "VALUE" || str == "value") {
      value_ = VALUE;
    } else if (str == "INDEX" || str == "index") {
      value_ = INDEX;
    } else if (str == "INDEX_SCALE" || str == "index_scale") {
      value_ = INDEX_SCALE;
    } else if (str == "CONV" || str == "conv") {
      value_ = CONV;
    } else if (str == "SSM" || str == "ssm") {
      value_ = SSM;
    } else {
      value_ = INVALID;
    }
  }

  KVCacheTensorRole() = delete;

  constexpr operator Value() const { return value_; }
  explicit operator bool() = delete;

  bool operator==(KVCacheTensorRole rhs) const { return value_ == rhs.value_; }
  bool operator!=(KVCacheTensorRole rhs) const { return value_ != rhs.value_; }
  bool operator==(Value rhs) const { return value_ == rhs; }
  bool operator!=(Value rhs) const { return value_ != rhs; }

  constexpr const char* to_string() const {
    if (this->value_ == KEY) {
      return "key";
    } else if (this->value_ == VALUE) {
      return "value";
    } else if (this->value_ == INDEX) {
      return "index";
    } else if (this->value_ == INDEX_SCALE) {
      return "index_scale";
    } else if (this->value_ == CONV) {
      return "conv";
    } else if (this->value_ == SSM) {
      return "ssm";
    } else {
      return "invalid";
    }
  }

 private:
  Value value_;
};

}  // namespace xllm
