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

#include <string>

namespace xllm {

class InstanceName {
 public:
  static InstanceName* name() {
    static InstanceName n;
    return &n;
  }

  void set_name(const std::string& name) {
    name_ = name;
    name_hash_ = std::to_string(std::hash<std::string>{}(name_));
  }

  std::string get_name() const { return name_; }

  std::string get_name_hash() const { return name_hash_; }

 private:
  InstanceName() {}
  InstanceName(const InstanceName&) = delete;
  InstanceName& operator=(const InstanceName&) = delete;

 private:
  std::string name_;
  std::string name_hash_;
};

}  // namespace xllm
