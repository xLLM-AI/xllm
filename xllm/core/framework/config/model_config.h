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
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <string_view>

#include "core/common/macros.h"
#include "core/framework/config/option_category.h"

namespace xllm {

class JsonReader;

class ModelConfig final {
 public:
  ModelConfig() = default;
  ~ModelConfig() = default;

  static ModelConfig& get_instance();

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();
  void normalize_cpp_chat_template(const std::string& model_type);

  [[nodiscard]] static bool is_python_model_impl(std::string_view model_impl);

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "MODEL OPTIONS",
        {"model_id",
         "model",
         "model_impl",
         "backend",
         "task",
         "python_model_path",
         "limit_image_per_prompt",
         "max_encoder_cache_size",
         "reasoning_parser",
         "tool_call_parser",
         "enable_qwen3_reranker",
         "enable_return_mm_full_embeddings",
         "mm_download_headers",
         "flashinfer_workspace_buffer_size",
         "use_audio_in_video",
         "use_cpp_chat_template"}};
    return kOptionCategory;
  }

  PROPERTY(std::string, model_id);

  PROPERTY(std::string, model);

  PROPERTY(std::string, model_impl);

  PROPERTY(std::string, python_model_path);

  PROPERTY(std::string, backend);

  PROPERTY(std::string, task) = "generate";

  PROPERTY(int32_t, limit_image_per_prompt) = 8;

  PROPERTY(int64_t, max_encoder_cache_size) = 0;

  PROPERTY(std::string, reasoning_parser);

  PROPERTY(std::string, tool_call_parser);

  PROPERTY(bool, enable_qwen3_reranker) = false;

  PROPERTY(bool, enable_return_mm_full_embeddings) = false;

  PROPERTY(std::string, mm_download_headers) = "";

  PROPERTY(int32_t, flashinfer_workspace_buffer_size) = 128 * 1024 * 1024;

  PROPERTY(bool, use_audio_in_video) = false;

  PROPERTY(bool, use_cpp_chat_template) = true;
};

}  // namespace xllm
