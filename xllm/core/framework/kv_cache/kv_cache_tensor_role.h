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
    KEY_SCALE = 6,
    VALUE_SCALE = 7,
    WINDOW = 8,
    CACHE_SCALE = 9,
    KV_STATE = 10,
    SCORE_STATE = 11,
    INDEX_KV_STATE = 12,
    INDEX_SCORE_STATE = 13,
    SWA = 14,
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
    } else if (str == "KEY_SCALE" || str == "key_scale") {
      value_ = KEY_SCALE;
    } else if (str == "VALUE_SCALE" || str == "value_scale") {
      value_ = VALUE_SCALE;
    } else if (str == "WINDOW" || str == "window") {
      value_ = WINDOW;
    } else if (str == "CACHE_SCALE" || str == "cache_scale") {
      value_ = CACHE_SCALE;
    } else if (str == "KV_STATE" || str == "kv_state") {
      value_ = KV_STATE;
    } else if (str == "SCORE_STATE" || str == "score_state") {
      value_ = SCORE_STATE;
    } else if (str == "INDEX_KV_STATE" || str == "index_kv_state") {
      value_ = INDEX_KV_STATE;
    } else if (str == "INDEX_SCORE_STATE" || str == "index_score_state") {
      value_ = INDEX_SCORE_STATE;
    } else if (str == "SWA" || str == "swa") {
      value_ = SWA;
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
    } else if (this->value_ == KEY_SCALE) {
      return "key_scale";
    } else if (this->value_ == VALUE_SCALE) {
      return "value_scale";
    } else if (this->value_ == WINDOW) {
      return "window";
    } else if (this->value_ == CACHE_SCALE) {
      return "cache_scale";
    } else if (this->value_ == KV_STATE) {
      return "kv_state";
    } else if (this->value_ == SCORE_STATE) {
      return "score_state";
    } else if (this->value_ == INDEX_KV_STATE) {
      return "index_kv_state";
    } else if (this->value_ == INDEX_SCORE_STATE) {
      return "index_score_state";
    } else if (this->value_ == SWA) {
      return "swa";
    } else {
      return "invalid";
    }
  }

 private:
  Value value_;
};

}  // namespace xllm
