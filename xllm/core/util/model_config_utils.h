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

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace xllm {

class JsonReader;

namespace util {

// Returns zero-based post-layer capture indices from a block draft checkpoint.
std::vector<int32_t> read_capture_layer_ids(
    const std::string& model_weights_path);

std::string get_model_type(const JsonReader& reader,
                           const std::filesystem::path& model_path,
                           std::optional<std::string> backend = std::nullopt);

std::string get_model_type(const std::filesystem::path& model_path,
                           std::optional<std::string> backend = std::nullopt);

}  // namespace util
}  // namespace xllm
