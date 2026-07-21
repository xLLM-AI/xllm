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

#include "core/util/dit_model_discovery.h"

#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include "core/util/json_reader.h"

namespace xllm {
namespace util {
namespace {

bool is_hidden_dir_name(const std::string& dir_name) {
  return dir_name.empty() || dir_name[0] == '.';
}

bool has_safetensors_weights(const std::filesystem::path& dir_path) {
  std::error_code ec;
  for (const auto& file : std::filesystem::directory_iterator(dir_path, ec)) {
    if (ec) {
      return false;
    }
    if (file.path().extension() == ".safetensors") {
      return true;
    }
  }
  return false;
}

bool try_discover_component(const std::filesystem::path& dir_path,
                            const std::string& comp_name,
                            DitDiscoveredComponent* component) {
  const std::filesystem::path config_path = dir_path / "config.json";
  if (!std::filesystem::exists(config_path) ||
      !has_safetensors_weights(dir_path)) {
    return false;
  }

  JsonReader reader;
  if (!reader.parse(config_path.string())) {
    return false;
  }

  std::string component_type = comp_name;
  if (auto model_type = reader.value<std::string>("model_type")) {
    component_type = model_type.value();
  }

  component->name = comp_name;
  component->path = dir_path;
  component->component_type = component_type;
  return true;
}

void try_discover_in_directory(
    const std::filesystem::path& parent_path,
    std::vector<DitDiscoveredComponent>* components) {
  std::error_code ec;
  for (const auto& entry :
       std::filesystem::directory_iterator(parent_path, ec)) {
    if (ec) {
      return;
    }
    if (!entry.is_directory()) {
      continue;
    }
    const std::string dir_name = entry.path().filename().string();
    if (is_hidden_dir_name(dir_name)) {
      continue;
    }
    DitDiscoveredComponent component;
    if (try_discover_component(entry.path(), dir_name, &component)) {
      components->push_back(std::move(component));
    }
  }
}

}  // namespace

std::optional<std::vector<DitDiscoveredComponent>> discover_dit_components(
    const std::filesystem::path& model_path) {
  if (!std::filesystem::exists(model_path) ||
      !std::filesystem::is_directory(model_path)) {
    return std::nullopt;
  }

  std::vector<DitDiscoveredComponent> components;
  try_discover_in_directory(model_path, &components);

  if (components.empty()) {
    std::error_code ec;
    for (const auto& entry :
         std::filesystem::directory_iterator(model_path, ec)) {
      if (ec) {
        break;
      }
      if (!entry.is_directory()) {
        continue;
      }
      const std::string dir_name = entry.path().filename().string();
      if (is_hidden_dir_name(dir_name)) {
        continue;
      }
      try_discover_in_directory(entry.path(), &components);
    }
  }

  if (components.empty()) {
    return std::nullopt;
  }
  return components;
}

}  // namespace util
}  // namespace xllm
