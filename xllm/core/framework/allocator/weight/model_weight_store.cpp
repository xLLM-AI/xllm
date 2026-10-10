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

#include "core/framework/allocator/weight/model_weight_store.h"

namespace xllm {

ModelWeightStore::~ModelWeightStore() = default;

WeightAllocation& ModelWeightStore::get_or_create(const std::string& model_id) {
  return models_[model_id];
}

WeightAllocation* ModelWeightStore::find(const std::string& model_id) {
  auto it = models_.find(model_id);
  return it == models_.end() ? nullptr : &it->second;
}

const WeightAllocation* ModelWeightStore::find(
    const std::string& model_id) const {
  auto it = models_.find(model_id);
  return it == models_.end() ? nullptr : &it->second;
}

void ModelWeightStore::clear() { models_.clear(); }

}  // namespace xllm
