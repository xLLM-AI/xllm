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

#include <filesystem>
#include <nlohmann/json_fwd.hpp>
#include <string>

namespace xllm {

class ModelArgs;
class QuantArgs;
class JsonReader;

// Normalize speculators-format Eagle3 draft configs for the qwen3_eagle3
// loader.
void normalize_speculators_config(nlohmann::json* config);

void normalize_config_torch_dtype(JsonReader& reader);

bool load_quant_cfg(const JsonReader& reader, QuantArgs& quant_args);

bool load_hf_model_args(const std::filesystem::path& model_weights_path,
                        const std::string& backend,
                        ModelArgs* args);

bool load_hf_audio_preprocessor_args(
    const std::filesystem::path& model_weights_path,
    ModelArgs& args);

bool load_hf_image_preprocessor_args(
    const std::filesystem::path& model_weights_path,
    ModelArgs& args);

bool load_hf_video_preprocessor_args(
    const std::filesystem::path& model_weights_path,
    ModelArgs& args);

}  // namespace xllm
