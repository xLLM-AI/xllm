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

#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>

namespace xllm::util {

inline std::string get_model_name(
    const std::filesystem::path& normalized_model_path) {
  std::string model_name;

  if (normalized_model_path.has_filename()) {
    model_name = normalized_model_path.filename().string();
  } else {
    model_name = normalized_model_path.parent_path().filename().string();
  }

  if (model_name.empty()) {
    LOG(FATAL) << "Cannot extract model name from path, as it appears to be a "
                  "root directory: "
               << normalized_model_path.string();
    return "";
  }

  return model_name;
}

inline std::string get_model_repository_name(
    const std::filesystem::path& normalized_model_path) {
  std::filesystem::path version_path = normalized_model_path;
  if (!version_path.has_filename()) {
    version_path = version_path.parent_path();
  }

  const std::string model_version = version_path.filename().string();
  const bool is_numeric_version =
      !model_version.empty() &&
      std::all_of(
          model_version.begin(), model_version.end(), [](char character) {
            return std::isdigit(static_cast<unsigned char>(character)) != 0;
          });
  if (!is_numeric_version) {
    return get_model_name(normalized_model_path);
  }

  const std::string repository_name =
      version_path.parent_path().filename().string();
  if (repository_name.empty()) {
    return get_model_name(normalized_model_path);
  }
  return repository_name;
}

}  // namespace xllm::util
