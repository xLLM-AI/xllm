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

#include "core/framework/config/speculative_config.h"
#include "core/framework/model/aux_hidden_capture.h"
#include "core/runtime/options.h"
#include "core/util/json_reader.h"

namespace xllm {

namespace {

void apply_block_diffusion_target_capture(ModelArgs& args,
                                          const std::string& draft_model_path) {
  CHECK(!draft_model_path.empty())
      << "Block-diffusion target requires a draft checkpoint.";
  args.layers_to_capture(read_capture_layer_ids(draft_model_path));
}

}  // namespace

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

void configure_speculative_model_args(ModelArgs& args,
                                      const runtime::Options& options) {
  if (!options.enable_speculative_decode()) {
    return;
  }
  const std::string& speculative_algorithm = options.speculative_algorithm();
  const bool is_block_diffusion =
      SpeculativeConfig::is_block_diffusion_algorithm(speculative_algorithm);

  if (is_block_diffusion) {
    apply_block_diffusion_target_capture(
        args, options.draft_model_path().value_or(""));
  }
#if defined(USE_NPU)
  if (!options.is_draft_model() &&
      SpeculativeConfig::requires_aux_hidden_capture(speculative_algorithm) &&
      args.layers_to_capture().empty()) {
    std::vector<int32_t> capture_layer_ids;
    if (SpeculativeConfig::is_eagle3_algorithm(speculative_algorithm) &&
        options.draft_model_path().has_value()) {
      // Eagle3.1 pins capture layers in the draft config; legacy configs
      // fall back to the default below.
      capture_layer_ids = read_capture_layer_ids(*options.draft_model_path(),
                                                 /*required=*/false);
    }
    if (capture_layer_ids.empty()) {
      const int32_t num_layers = static_cast<int32_t>(args.n_layers());
      capture_layer_ids =
          AuxHiddenCapture::legacy_default_capture_layer_ids(num_layers);
    }
    args.layers_to_capture(std::move(capture_layer_ids));
  }
#endif
}

}  // namespace xllm
