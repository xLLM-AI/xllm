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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace xllm {

// Pure geometry shared by request validation and the visual processor. Callers
// validate positive dimensions/multiple and the supported aspect ratio first.
inline std::pair<int64_t, int64_t> minimax_h3_reference_image_size(
    int64_t source_height,
    int64_t source_width,
    int64_t short_edge,
    int64_t multiple) {
  const double scale =
      static_cast<double>(short_edge) /
      static_cast<double>(std::min(source_height, source_width));
  const auto round = [multiple](double value) {
    return std::max<int64_t>(
        multiple,
        static_cast<int64_t>(std::nearbyint(value / multiple)) * multiple);
  };
  return {round(source_height * scale), round(source_width * scale)};
}

}  // namespace xllm
