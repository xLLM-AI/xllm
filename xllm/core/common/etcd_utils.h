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

inline std::string normalize_etcd_namespace(const std::string& etcd_namespace) {
  if (etcd_namespace.empty()) {
    return "";
  }

  size_t start = 0;
  size_t end = etcd_namespace.size();
  while (start < end && etcd_namespace[start] == '/') {
    ++start;
  }
  while (end > start && etcd_namespace[end - 1] == '/') {
    --end;
  }

  if (start >= end) {
    return "";
  }
  return "/" + etcd_namespace.substr(start, end - start) + "/";
}

}  // namespace xllm
