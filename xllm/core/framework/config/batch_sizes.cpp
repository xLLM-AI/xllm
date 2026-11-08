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

#include "core/framework/config/batch_sizes.h"

#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>
#include <glog/logging.h>

#include <unordered_set>

namespace xllm::util {

std::optional<std::vector<uint32_t>> parse_batch_sizes(
    const std::string& batch_sizes_str) {
  if (batch_sizes_str.empty() || batch_sizes_str == "auto") {
    return std::nullopt;
  }

  // Parse the batch sizes and remove duplicates.
  const auto size_strs = absl::StrSplit(batch_sizes_str, ',');
  std::unordered_set<uint32_t> sizes_set;
  for (const auto& size_str : size_strs) {
    uint32_t batch_size = 0;
    if (!absl::SimpleAtoi(size_str, &batch_size)) {
      LOG(ERROR) << "Failed to parse batch size: " << size_str;
      continue;
    }
    sizes_set.emplace(batch_size);
  }
  if (sizes_set.empty()) {
    return std::nullopt;
  }
  return std::vector<uint32_t>{sizes_set.begin(), sizes_set.end()};
}

}  // namespace xllm::util
