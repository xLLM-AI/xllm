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

#include "core/common/types.h"

namespace xllm {

// Service-layer abstraction of the active serving pipeline.
// Unlike EngineType (which includes engine-internal variants such as SSM),
// ServingMode only exposes the distinctions that matter to the API surface.
enum class ServingMode : int8_t {
  LLM = 0,
  VLM = 1,
  DIT = 2,
  REC = 3,
};

// Maps an engine-layer EngineType to its corresponding ServingMode.
// SSM (speculative decoding) serves the same API as LLM. VLMSSM is the
// speculative variant of VLM and keeps the VLM API surface.
inline ServingMode to_serving_mode(EngineType engine_type) {
  switch (static_cast<EngineType::Value>(engine_type)) {
    case EngineType::VLM:
    case EngineType::VLMSSM:
      return ServingMode::VLM;
    case EngineType::DIT:
      return ServingMode::DIT;
    case EngineType::REC:
      return ServingMode::REC;
    default:
      return ServingMode::LLM;
  }
}

}  // namespace xllm
