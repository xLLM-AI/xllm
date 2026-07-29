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

#include "core/framework/config/model_config.h"

#include <glog/logging.h>

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"

DEFINE_string(model_id, "", "hf model name.");

DEFINE_string(model, "", "Name or path of the huggingface model to use.");

DEFINE_string(python_model_path,
              "",
              "Filesystem directory that contains the 'xllm' package, "
              "prepended to sys.path so the embedded interpreter can import "
              "the 'xllm.python' model executor subpackage. Falls back to the "
              "XLLM_PYTHON_MODEL_PATH env var when empty.");

DEFINE_string(
    backend,
    "",
    "Choose the backend model type. 'llm' for text-only, "
    "'vlm' for multimodal (text and images), 'dit' for diffusion models.");

DEFINE_string(model_impl,
              "",
              "Model executor implementation. Empty/'native' uses the built-in "
              "C++ model; 'python' runs the graph via the embedded Python "
              "interpreter ('python' model package).");

DEFINE_string(task,
              "generate",
              "The task to use the model for(e.g. generate, embed, mm_embed).");

DEFINE_int32(limit_image_per_prompt,
             8,
             "Maximum number of image per prompt. Only applicable for "
             "multimodal models.");

DEFINE_int64(max_encoder_cache_size,
             0,
             "Max gpu/npu memory size in MB for encoder cache per worker. "
             "Default is 0, which disables encoder cache.");

DEFINE_string(reasoning_parser,
              "",
              "Specify the reasoning parser for handling reasoning "
              "interactions(e.g. auto, glm45, glm47, glm5, qwen3, qwen35, "
              "deepseek-r1).");

DEFINE_string(tool_call_parser,
              "",
              "Specify the parser for handling tool-call interactions(e.g. "
              "auto, qwen25, qwen3, qwen35, qwen3_coder, kimi_k2, "
              "deepseekv3, deepseekv32, deepseekv4, glm45, glm47, glm5).");

DEFINE_bool(enable_qwen3_reranker, false, "Whether to enable qwen3 reranker.");

DEFINE_int32(flashinfer_workspace_buffer_size,
             128 * 1024 * 1024,
             "The user reserved workspace buffer used to store intermediate "
             "attention results in split-k algorithm for flashinfer.");

DEFINE_bool(enable_return_mm_full_embeddings,
            false,
            "return vit and sequence embeddings for vlm models");

DEFINE_string(mm_download_headers,
              "",
              "Service-level default HTTP headers for multimodal downloads, "
              "as a JSON object. Per-request headers take precedence. "
              "Example: '{\"Authorization\":\"Bearer xxx\"}'");

DEFINE_bool(
    use_audio_in_video,
    false,
    "Whether to decode both audio and video when the input is a video.");

// NOTE: This is an experimental flag,
//       it needs to be removed after the function is stable.
DEFINE_bool(use_cpp_chat_template,
            true,
            "Use native C++ chat template for supported models "
            "(e.g. deepseek_v32, deepseek_v4) instead of Jinja. "
            "Set to false to fallback to Jinja for debugging.");

namespace xllm {
namespace {

bool is_cpp_chat_template_supported_model(const std::string& model_type) {
  return model_type == "deepseek_v32" || model_type == "deepseek_v4";
}

}  // namespace

void ModelConfig::from_flags() {
  XLLM_CONFIG_ASSIGN_FROM_FLAG(model_id);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(model);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(model_impl);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(python_model_path);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(backend);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(task);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(limit_image_per_prompt);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(max_encoder_cache_size);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(reasoning_parser);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(tool_call_parser);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_qwen3_reranker);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_return_mm_full_embeddings);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(mm_download_headers);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(flashinfer_workspace_buffer_size);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(use_audio_in_video);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(use_cpp_chat_template);
}

bool ModelConfig::is_python_model_impl(std::string_view model_impl) {
  // Single place that tolerates the "py" alias for "python". All callers route
  // their model_impl comparison through here, so the raw config value is never
  // normalized: no separate canonicalization step is needed.
  return model_impl == "python" || model_impl == "py";
}

void ModelConfig::normalize_cpp_chat_template(const std::string& model_type) {
  if (!use_cpp_chat_template()) {
    return;
  }

  if (is_cpp_chat_template_supported_model(model_type)) {
    return;
  }

  use_cpp_chat_template(false);
  LOG(WARNING) << "use_cpp_chat_template is not supported for model_type="
               << model_type << ", forcing use_cpp_chat_template=false.";
}

void ModelConfig::from_json(const JsonReader& json) {
  XLLM_CONFIG_ASSIGN_FROM_JSON(model_id);
  XLLM_CONFIG_ASSIGN_FROM_JSON(model);
  XLLM_CONFIG_ASSIGN_FROM_JSON(model_impl);
  XLLM_CONFIG_ASSIGN_FROM_JSON(python_model_path);
  XLLM_CONFIG_ASSIGN_FROM_JSON(backend);
  XLLM_CONFIG_ASSIGN_FROM_JSON(task);
  XLLM_CONFIG_ASSIGN_FROM_JSON(limit_image_per_prompt);
  XLLM_CONFIG_ASSIGN_FROM_JSON(max_encoder_cache_size);
  XLLM_CONFIG_ASSIGN_FROM_JSON(reasoning_parser);
  XLLM_CONFIG_ASSIGN_FROM_JSON(tool_call_parser);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_qwen3_reranker);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_return_mm_full_embeddings);
  XLLM_CONFIG_ASSIGN_FROM_JSON(mm_download_headers);
  XLLM_CONFIG_ASSIGN_FROM_JSON(flashinfer_workspace_buffer_size);
  XLLM_CONFIG_ASSIGN_FROM_JSON(use_audio_in_video);
  XLLM_CONFIG_ASSIGN_FROM_JSON(use_cpp_chat_template);
}

void ModelConfig::append_config_json(
    nlohmann::ordered_json& config_json) const {
  const ModelConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, model_id);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(config_json, default_config, model);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, model_impl);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, python_model_path);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(config_json, default_config, backend);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(config_json, default_config, task);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, limit_image_per_prompt);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, max_encoder_cache_size);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, reasoning_parser);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, tool_call_parser);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_qwen3_reranker);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_return_mm_full_embeddings);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, mm_download_headers);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, flashinfer_workspace_buffer_size);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, use_audio_in_video);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, use_cpp_chat_template);
}

ModelConfig& ModelConfig::get_instance() {
  static ModelConfig config;
  return config;
}

void ModelConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
}

}  // namespace xllm
