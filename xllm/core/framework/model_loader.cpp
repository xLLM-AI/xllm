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

#include "model_loader.h"

#include <absl/strings/match.h>
#include <absl/strings/str_replace.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <filesystem>
#include <vector>

#include "hf_model_loader.h"

namespace xllm {

std::unique_ptr<ModelLoader> ModelLoader::create(
    const std::string& model_weights_path) {
  ModelType model_type;
  for (const auto& entry :
       std::filesystem::directory_iterator(model_weights_path)) {
    if (entry.path().extension() == ".safetensors" ||
        entry.path().extension() == ".bin") {
      model_type = ModelType::HF_MODEL_TYPE;
      break;
    }
  }

  if (model_type == ModelType::HF_MODEL_TYPE) {
    return std::make_unique<HFModelLoader>(model_weights_path);
  } else {
    LOG(FATAL) << "Only support HF model type currently.";
  }

  return nullptr;
}

}  // namespace xllm
