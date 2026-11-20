/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace xllm {

inline constexpr char kH3RainfusionPolicyEnv[] = "XLLM_H3_RAINFUSION_POLICY";

struct H3RainfusionPolicy {
  std::string_view name;
  int64_t step_begin;
  int64_t step_end;
  int64_t layer_begin;
  int64_t layer_end;

  constexpr bool enabled() const { return step_begin < step_end; }

  constexpr bool includes_step(int64_t step) const {
    return step >= step_begin && step < step_end;
  }

  constexpr bool includes_layer(int64_t layer) const {
    return layer >= layer_begin && layer < layer_end;
  }

  constexpr int64_t layer_count(int64_t begin, int64_t end) const {
    return std::max<int64_t>(
        0, std::min(end, layer_end) - std::max(begin, layer_begin));
  }
};

inline constexpr H3RainfusionPolicy kH3SparseNone{"sparse-none", 0, 0, 0, 0};
inline constexpr H3RainfusionPolicy kH3SparseMiddleSteps{"sparse-middle-steps",
                                                         8,
                                                         36,
                                                         0,
                                                         50};
inline constexpr H3RainfusionPolicy kH3SparseMiddleLayers{
    "sparse-middle-layers",
    0,
    49,
    2,
    40};

inline H3RainfusionPolicy resolve_h3_rainfusion_policy(bool enabled,
                                                       const char* value) {
  if (!enabled) {
    return kH3SparseNone;
  }
  if (value == nullptr || value == kH3SparseMiddleSteps.name) {
    return kH3SparseMiddleSteps;
  }
  if (value == kH3SparseMiddleLayers.name) {
    return kH3SparseMiddleLayers;
  }
  throw std::invalid_argument(
      "XLLM_H3_RAINFUSION_POLICY must be sparse-middle-steps or "
      "sparse-middle-layers");
}

}  // namespace xllm
