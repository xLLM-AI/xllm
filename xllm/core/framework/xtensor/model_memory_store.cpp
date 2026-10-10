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

#include "core/framework/xtensor/model_memory_store.h"

#include "core/virtual_memory/mapped_memory_region.h"

namespace xllm {

ModelMemoryStore::~ModelMemoryStore() = default;

ModelTensors& ModelMemoryStore::get_or_create(const std::string& model_id) {
  return models_[model_id];
}

ModelTensors* ModelMemoryStore::find(const std::string& model_id) {
  auto it = models_.find(model_id);
  return it == models_.end() ? nullptr : &it->second;
}

const ModelTensors* ModelMemoryStore::find(const std::string& model_id) const {
  auto it = models_.find(model_id);
  return it == models_.end() ? nullptr : &it->second;
}

void ModelMemoryStore::clear() { models_.clear(); }

}  // namespace xllm
