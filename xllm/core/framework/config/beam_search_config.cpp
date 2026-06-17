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

#include "core/framework/config/beam_search_config.h"

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"

DEFINE_bool(enable_beam_search_kernel,
            false,
            "Whether to enable beam search kernel.");

DEFINE_int32(beam_width, 1, "Beam width for beam search.");

#if defined(USE_NPU) || defined(USE_CUDA) || defined(USE_DCU)
DEFINE_bool(enable_block_copy_kernel,
            true,
            "Whether to use block copy kernel on supported backends.");
#else
DEFINE_bool(enable_block_copy_kernel,
            false,
            "Whether to use block copy kernel on supported backends.");
#endif

DEFINE_bool(enable_topk_sorted,
            true,
            "Whether to enable sorted output for topk.");

namespace xllm {

void BeamSearchConfig::from_flags() {
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_beam_search_kernel);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(beam_width);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_block_copy_kernel);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_topk_sorted);
}

void BeamSearchConfig::from_json(const JsonReader& json) {
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_beam_search_kernel);
  XLLM_CONFIG_ASSIGN_FROM_JSON(beam_width);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_block_copy_kernel);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_topk_sorted);
}

void BeamSearchConfig::append_config_json(
    nlohmann::ordered_json& config_json) const {
  const BeamSearchConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_beam_search_kernel);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, beam_width);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_block_copy_kernel);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_topk_sorted);
}

BeamSearchConfig& BeamSearchConfig::get_instance() {
  static BeamSearchConfig config;
  return config;
}

void BeamSearchConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
}

}  // namespace xllm
