/* Copyright 2026 The xLLM Authors.

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

#include "core/runtime/block_diffusion_model_config.h"

#include <glog/logging.h>

#include <optional>
#include <unordered_set>

#include "core/framework/config/kernel_config.h"
#include "core/framework/model/model_args.h"
#include "core/platform/platform.h"
#include "core/runtime/options.h"
#include "core/util/utils.h"
#include "util/json_reader.h"

namespace xllm::block_diffusion {
namespace {

struct CheckpointConfig {
  std::string path;
  std::vector<int32_t> target_layer_ids;
};

CheckpointConfig read_checkpoint_config(const runtime::Options& options,
                                        const std::string& model_weights_path) {
  CheckpointConfig config;
  if (options.is_draft_engine()) {
    config.path = model_weights_path;
  } else {
    CHECK(options.draft_model_path().has_value())
        << "block-diffusion speculative decoding requires --draft_model.";
    config.path = options.draft_model_path().value();
  }

  JsonReader reader;
  const std::string config_path = config.path + "/config.json";
  CHECK(reader.parse(config_path))
      << "Failed to parse block-diffusion config: " << config_path;
  config.target_layer_ids = reader.value_or<std::vector<int32_t>>(
      std::vector<std::string>{"dspark_target_layer_ids",
                               "target_layer_ids",
                               "dflash_config.target_layer_ids"},
      std::vector<int32_t>{});
  CHECK(!config.target_layer_ids.empty())
      << "Block-diffusion config requires dspark_target_layer_ids, "
         "target_layer_ids, or dflash_config.target_layer_ids.";

  std::unordered_set<int32_t> unique_layer_ids;
  for (int32_t layer_id : config.target_layer_ids) {
    CHECK_GE(layer_id, 0)
        << "Block-diffusion target layer IDs must be non-negative.";
    CHECK(unique_layer_ids.emplace(layer_id).second)
        << "Block-diffusion target layer IDs must be unique.";
  }
  return config;
}

std::optional<std::string> platform_draft_model_type(
    const ModelArgs& args,
    const runtime::Options& options) {
#if defined(USE_NPU)
  if (options.speculative_algorithm() == "DFlash") {
    return "DFlashDraftModel";
  }
  if (util::is_deepseek_v4_model_type(args.model_type())) {
    return "deepseek_v4_dspark";
  }
  return "DSparkDraftModel";
#else
  (void)args;
  (void)options;
  // Other backends keep the model type declared by their draft checkpoint.
  return std::nullopt;
#endif
}

void configure_deepseek_v4_dspark_args(ModelArgs& args,
                                       const runtime::Options& options) {
  CHECK_GT(args.dspark_num_layers(), 0)
      << "DeepSeek-V4 DSpark requires at least one draft layer.";
  args.n_layers(args.dspark_num_layers());
  args.n_hash_layers(0);
  args.dspark_block_size(options.num_speculative_tokens());
  // DSpark stages are all standard SWA layers. Their stage ids are not target
  // model layer ids, so target compress_ratios[0..N) must not be reused.
  args.compress_ratios(
      std::vector<int32_t>(static_cast<size_t>(args.dspark_num_layers()), 1));

#if defined(USE_NPU)
  args.dspark_use_native_sas(
      KernelConfig::get_instance().enable_dspark_native_sas());
#else
  LOG(FATAL) << "DeepSeek-V4 DSpark is not supported on "
             << Platform::type_str() << ".";
#endif
}

}  // namespace

bool is_algorithm(std::string_view algorithm) {
  return algorithm == "DFlash" || algorithm == "DSpark";
}

std::vector<int32_t> map_target_layer_ids_to_capture_points(
    const std::vector<int32_t>& target_layer_ids) {
  std::vector<int32_t> capture_points;
  capture_points.reserve(target_layer_ids.size());
  // NPU model hooks capture the input of layer i, so target layer L's output
  // is observed at L+1. Backends whose hooks run after a layer use L directly.
  const int32_t capture_offset =
      Platform::block_diffusion_capture_layer_offset();
  for (int32_t layer_id : target_layer_ids) {
    capture_points.emplace_back(layer_id + capture_offset);
  }
  return capture_points;
}

void configure_model_args(ModelArgs& args,
                          const runtime::Options& options,
                          const std::string& model_weights_path) {
  CHECK(is_algorithm(options.speculative_algorithm()));
  const CheckpointConfig checkpoint =
      read_checkpoint_config(options, model_weights_path);
  args.layers_to_capture(
      map_target_layer_ids_to_capture_points(checkpoint.target_layer_ids));

  if (!options.is_draft_engine()) {
    return;
  }

  const bool is_deepseek_v4 =
      util::is_deepseek_v4_model_type(args.model_type());
  if (const std::optional<std::string> draft_model_type =
          platform_draft_model_type(args, options)) {
    LOG(INFO) << "Overriding draft model_type from " << args.model_type()
              << " to " << *draft_model_type
              << " for block-diffusion speculative decoding";
    args.model_type(*draft_model_type);
  }

  if (options.speculative_algorithm() == "DSpark" && is_deepseek_v4) {
    configure_deepseek_v4_dspark_args(args, options);
  }
}

}  // namespace xllm::block_diffusion
