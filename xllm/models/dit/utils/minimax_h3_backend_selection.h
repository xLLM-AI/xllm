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

#include <cstdint>
#include <stdexcept>
#include <string_view>

#include "core/framework/request/minimax_h3_conditioning.h"

namespace xllm {

enum class H3BackendConstruction : int8_t { NATIVE, REFERENCE_TEST, A3_16CARD };

// Called before model construction, including the explicit qualified factory.
inline H3BackendConstruction select_h3_production_construction(
    std::string_view profile,
    bool enable_rolling_load) {
  if (profile == "native") {
    return H3BackendConstruction::NATIVE;
  }
  if (profile != kMiniMaxH3ExecutionProfile) {
    throw std::invalid_argument("Unknown MiniMax-H3 execution profile");
  }
  if (enable_rolling_load) {
    throw std::invalid_argument(
        "Qualified MiniMax-H3 requires resident weights; rolling load is "
        "unsupported");
  }
  return H3BackendConstruction::A3_16CARD;
}

}  // namespace xllm
