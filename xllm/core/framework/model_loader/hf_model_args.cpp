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

#include "core/framework/model_loader/hf_model_args.h"

#include <glog/logging.h>

#include "core/framework/model/model_args.h"
#include "core/util/json_reader.h"
#include "core/util/model_config_utils.h"
#include "core/util/utils.h"
#include "models/model_registry.h"

namespace xllm {

namespace {

void lift_speculators_config(nlohmann::json& config) {
  if (!config.contains("speculators_model_type") ||
      !config.contains("transformer_layer_config")) {
    return;
  }
  const auto& tlc = config["transformer_layer_config"];
  if (tlc.is_object()) {
    for (const auto& [key, value] : tlc.items()) {
      if (!value.is_null() && !config.contains(key)) {
        config[key] = value;
      }
    }
  }
  if (config.contains("rope_parameters") &&
      config["rope_parameters"].is_object()) {
    const auto& rope = config["rope_parameters"];
    if (rope.contains("rope_theta") && !config.contains("rope_theta")) {
      config["rope_theta"] = rope["rope_theta"];
    }
    if (rope.contains("factor") && !config.contains("rope_scaling")) {
      config["rope_scaling"] = rope;
    }
  }
}

}  // namespace

void normalize_config_torch_dtype(JsonReader& reader) {
  nlohmann::json& config = reader.mutable_data();
  if (!config.contains("torch_dtype") && config.contains("dtype")) {
    config["torch_dtype"] = config["dtype"];
  }
  lift_speculators_config(config);
}

void normalize_speculators_config(nlohmann::json* config) {
  const auto spec_type_it = config->find("speculators_model_type");
  if (spec_type_it == config->end() || !spec_type_it->is_string() ||
      spec_type_it->get<std::string>() != "eagle3") {
    return;
  }
  const auto layer_it = config->find("transformer_layer_config");
  CHECK(layer_it != config->end() && layer_it->is_object())
      << "speculators eagle3 draft config requires a "
         "transformer_layer_config object";
  const std::string layer_model_type = layer_it->value("model_type", "");
  const bool layer_use_qk_norm =
      layer_it->value("use_qk_norm", layer_model_type == "qwen3");
  (*config)["model_type"] = "qwen3_eagle3";
  if (!config->contains("use_qk_norm")) {
    (*config)["use_qk_norm"] = layer_use_qk_norm;
  }
}

bool load_hf_model_args(const std::filesystem::path& model_weights_path,
                        const std::string& backend,
                        ModelArgs* args) {
  JsonReader reader;
  const std::filesystem::path args_file_path =
      model_weights_path / "config.json";
  if (!reader.parse(args_file_path.string())) {
    LOG(ERROR) << "Failed to parse model args file: " << args_file_path;
    return false;
  }

  normalize_speculators_config(&reader.mutable_data());

  const std::string model_type =
      util::get_model_type(reader, model_weights_path, backend);

  std::string resolved_model_type;
  std::string error_message;
  if (!resolve_model_registration_name(
          model_type, &resolved_model_type, &error_message)) {
    LOG(ERROR) << error_message;
    return false;
  }

  const auto model_args_loader =
      ModelRegistry::get_model_args_loader(resolved_model_type);
  if (model_args_loader == nullptr) {
    LOG(ERROR) << "Failed to find model args loader for model type "
               << resolved_model_type;
    return false;
  }
  normalize_config_torch_dtype(reader);
  if (!model_args_loader(reader, args)) {
    LOG(ERROR) << "Failed to load model args for model type "
               << resolved_model_type;
    return false;
  }
  args->enable_mla(util::should_enable_mla(model_weights_path, backend));

  return true;
}

}  // namespace xllm
