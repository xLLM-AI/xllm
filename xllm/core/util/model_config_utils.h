/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

namespace xllm {

class JsonReader;

namespace util {

std::string get_model_type(const JsonReader& reader,
                           const std::filesystem::path& model_path,
                           std::optional<std::string> backend = std::nullopt);

std::string get_model_type(const std::filesystem::path& model_path,
                           std::optional<std::string> backend = std::nullopt);

}  // namespace util
}  // namespace xllm
