/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <torch/torch.h>

#include <string>

namespace xllm {

// Structured identifier for a hardcoded rho table.
// This is designed for future extensibility across:
// - different model families
// - different CFG scales
// - different inference step counts
struct RhoTableSpec {
  std::string model_name;
  float cfg_scale = 0.0f;
  int64_t infer_steps = 0;

  bool operator==(const RhoTableSpec& other) const {
    return model_name == other.model_name && cfg_scale == other.cfg_scale &&
           infer_steps == other.infer_steps;
  }
};

// Hash support for unordered_map.
struct RhoTableSpecHash {
  std::size_t operator()(const RhoTableSpec& spec) const;
};

// Return a hardcoded rho table if the exact spec is registered.
// Return an undefined tensor if not found.
torch::Tensor get_hardcoded_rho_table(const RhoTableSpec& spec);

// Return whether an exact hardcoded rho table exists for the given spec.
bool has_hardcoded_rho_table(const RhoTableSpec& spec);

// Convert spec to a readable string for logging/debugging.
std::string to_string(const RhoTableSpec& spec);

}  // namespace xllm