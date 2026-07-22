/* Copyright 2025-2026 The xLLM Authors.
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

#include "core/util/model_config_utils.h"

#include <glog/logging.h>

#include <filesystem>

#include "core/util/dit_model_discovery.h"
#include "core/util/json_reader.h"

namespace xllm::util {

std::string get_model_type(const JsonReader& reader,
                           const std::filesystem::path& model_path,
                           std::optional<std::string> backend) {
  // Prefer model_type (e.g. LLM/VLM); fall back to model_name for configs
  // that only have model_name (e.g. LongCat-Image: {"model_name":
  // "LongCat-Image"}).
  std::optional<std::string> model_type =
      reader.value<std::string>("model_type");
  if (!model_type.has_value()) {
    model_type = reader.value<std::string>("model_name");
  }
  if (!model_type.has_value()) {
    LOG(FATAL) << "Please check config.json file in model path: " << model_path
               << ", it should contain model_type or model_name key.";
  }

  const bool is_qwen35_native_model_type =
      *model_type == "qwen3_5" || *model_type == "qwen3_5_moe";
  const bool use_vlm_model_type = backend.has_value() && *backend == "vlm";
  if (!is_qwen35_native_model_type || use_vlm_model_type) {
    return *model_type;
  }

  const std::optional<std::string> text_model_type =
      reader.value<std::string>("text_config.model_type");
  if (text_model_type.has_value()) {
    return *text_model_type;
  }

  if (*model_type == "qwen3_5_moe") {
    return "qwen3_5_moe_text";
  }
  return "qwen3_5_text";
}

std::string get_model_type(const std::filesystem::path& model_path,
                           std::optional<std::string> backend) {
  JsonReader reader;
  const std::filesystem::path model_index_path =
      model_path / "model_index.json";
  if (std::filesystem::exists(model_index_path)) {
    if (reader.parse(model_index_path.string())) {
      if (auto value = reader.value<std::string>("_class_name")) {
        return value.value();
      }
    }
  }

  const std::filesystem::path config_json_path = model_path / "config.json";
  if (!std::filesystem::exists(config_json_path)) {
    if (auto layout = discover_dit_model_layout(model_path)) {
      return layout->pipeline_type;
    }
    LOG(FATAL) << "Please check config.json or model_index.json file, one of "
                  "them should exist in the model path: "
               << model_path;
  }
  if (!reader.parse(config_json_path.string())) {
    LOG(FATAL) << "Failed to parse config.json file in model path: "
               << model_path;
  }

  return get_model_type(reader, model_path, std::move(backend));
}

}  // namespace xllm::util
