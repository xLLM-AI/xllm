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
#include <functional>
#include <string>
#include <string_view>

namespace xllm {

class InstanceName {
 public:
  static InstanceName* name() {
    static InstanceName n;
    return &n;
  }

  void set_name(const std::string& name) {
    name_ = name;
    name_hash_value_ = static_cast<uint64_t>(std::hash<std::string>{}(name_));
    name_hash_ = std::to_string(name_hash_value_);
  }

  std::string get_name() const { return name_; }

  std::string get_name_hash() const { return name_hash_; }

  uint64_t get_name_hash_value() const { return name_hash_value_; }

 private:
  InstanceName() {}
  InstanceName(const InstanceName&) = delete;
  InstanceName& operator=(const InstanceName&) = delete;

 private:
  std::string name_;
  std::string name_hash_;
  uint64_t name_hash_value_ = 0;
};

// Unique per-process request id. High 16 bits mix the instance hash so
// concurrent instances are unlikely to collide; low 48 bits are a counter.
int64_t next_request_id();

// `{prefix}{id}`, e.g. "cmpl-123". Formats `next_request_id()` once for
// OpenAI-compatible / HTTP string fields.
std::string generate_request_id(std::string_view prefix);

}  // namespace xllm
