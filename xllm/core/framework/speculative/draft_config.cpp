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

#include "core/framework/speculative/draft_config.h"

#include <glog/logging.h>

#include <string>
#include <utility>
#include <vector>

#include "core/util/json_reader.h"

namespace xllm {

std::vector<int32_t> boundary_to_post_layer_ids(
    std::vector<int32_t> boundary_ids) {
  for (int32_t& boundary_id : boundary_ids) {
    CHECK_GE(boundary_id, 1)
        << "Embedding capture (eagle_aux_hidden_state_layer_ids boundary "
           "index 0) is not supported; use a decoder-layer boundary index "
           "in [1, num_layers]";
    --boundary_id;
  }
  return boundary_ids;
}

std::vector<int32_t> read_capture_layer_ids(
    const std::string& model_weights_path,
    bool required) {
  JsonReader reader;
  const std::string config_path = model_weights_path + "/config.json";
  if (!reader.parse(config_path)) {
    CHECK(!required) << "Failed to parse draft config: " << config_path;
    return {};
  }

  // Legacy xLLM/vLLM draft configs already use 0-based post-layer output
  // indices, which match ModelArgs::layers_to_capture directly.
  std::vector<int32_t> capture_layer_ids =
      reader.value_or<std::vector<int32_t>>(
          std::vector<std::string>{"dspark_target_layer_ids",
                                   "target_layer_ids",
                                   "dflash_config.target_layer_ids"},
          std::vector<int32_t>{});
  if (!capture_layer_ids.empty()) {
    return capture_layer_ids;
  }

  // Speculators-format keys are hidden-state boundary indices (0=embedding
  // output, v=output of layer v-1); shift them to the post-layer contract.
  capture_layer_ids = reader.value_or<std::vector<int32_t>>(
      std::vector<std::string>{"aux_hidden_state_layer_ids",
                               "eagle_aux_hidden_state_layer_ids"},
      std::vector<int32_t>{});
  capture_layer_ids = boundary_to_post_layer_ids(std::move(capture_layer_ids));
  CHECK(!required || !capture_layer_ids.empty())
      << "Block-diffusion draft config requires dspark_target_layer_ids, "
         "target_layer_ids, dflash_config.target_layer_ids, or "
         "aux_hidden_state_layer_ids: "
      << config_path;
  return capture_layer_ids;
}

}  // namespace xllm
