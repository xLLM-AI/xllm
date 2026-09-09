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

#include "pretty_print.h"

#include <array>
#include <iomanip>
#include <sstream>

namespace xllm {

std::string readable_size(size_t bytes) {
  static const std::array<const char*, 5> suffixes = {
      "B", "KB", "MB", "GB", "TB"};
  const size_t bytes_in_kb = 1024;
  double size = static_cast<double>(bytes);
  size_t suffix_index = 0;
  while (size >= bytes_in_kb && suffix_index < suffixes.size() - 1) {
    size /= bytes_in_kb;
    ++suffix_index;
  }
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << size << " "
         << suffixes.at(suffix_index);
  return stream.str();
}

}  // namespace xllm
