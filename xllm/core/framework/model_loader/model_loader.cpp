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

#include "core/framework/model_loader/model_loader.h"

#include <glog/logging.h>

#include <filesystem>

#include "core/framework/config/model_config.h"
#include "core/framework/model_loader/hf_model_loader.h"

namespace xllm {

std::unique_ptr<ModelLoader> ModelLoader::create(
    const std::string& model_weights_path) {
  return create(model_weights_path, ModelConfig::get_instance().backend());
}

std::unique_ptr<ModelLoader> ModelLoader::create(
    const std::string& model_weights_path,
    const std::string& requested_family) {
  ModelType model_type = ModelType::INVALID;
  for (const auto& entry :
       std::filesystem::directory_iterator(model_weights_path)) {
    if (entry.path().extension() == ".safetensors" ||
        entry.path().extension() == ".bin") {
      model_type = ModelType::HF_MODEL_TYPE;
      break;
    }
  }

  if (model_type == ModelType::HF_MODEL_TYPE) {
    return std::make_unique<HFModelLoader>(model_weights_path,
                                           requested_family);
  } else {
    LOG(FATAL) << "Only support HF model type currently.";
  }

  return nullptr;
}

}  // namespace xllm
