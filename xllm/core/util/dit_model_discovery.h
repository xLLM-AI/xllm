/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace xllm {
namespace util {

struct DitDiscoveredComponent final {
  std::string name;
  std::filesystem::path path;
  std::string component_type;
};

struct DitModelLayout final {
  std::vector<DitDiscoveredComponent> components;
  // Registered DiT pipeline type (e.g. cola_dlm), resolved via ModelRegistry.
  std::string pipeline_type;
};

// Scan model_path for component subdirectories containing config.json and
// safetensors weights. Returns nullopt when nothing is found.
std::optional<std::vector<DitDiscoveredComponent>> discover_dit_components(
    const std::filesystem::path& model_path);

// Map discovered component types to a registered DiT pipeline type.
// Implemented in models/model_registry.cpp.
std::string resolve_dit_pipeline_type(
    const std::vector<DitDiscoveredComponent>& components);

inline std::optional<DitModelLayout> discover_dit_model_layout(
    const std::filesystem::path& model_path) {
  std::optional<std::vector<DitDiscoveredComponent>> components =
      discover_dit_components(model_path);
  if (!components.has_value()) {
    return std::nullopt;
  }
  DitModelLayout layout;
  layout.components = std::move(components.value());
  layout.pipeline_type = resolve_dit_pipeline_type(layout.components);
  if (layout.pipeline_type.empty()) {
    return std::nullopt;
  }
  return layout;
}

}  // namespace util
}  // namespace xllm
