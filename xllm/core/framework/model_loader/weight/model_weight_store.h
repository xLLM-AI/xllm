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

#pragma once

#include <string>
#include <unordered_map>

#include "core/framework/model_loader/weight/weight_allocation.h"

namespace xllm {

// Owns model weight reservations independently of KV cache mappings.
// Access is synchronized by the runtime manager.
class ModelWeightStore final {
 public:
  ModelWeightStore() = default;
  ~ModelWeightStore();

  ModelWeightStore(const ModelWeightStore&) = delete;
  ModelWeightStore& operator=(const ModelWeightStore&) = delete;

  WeightAllocation& get_or_create(const std::string& model_id);
  WeightAllocation* find(const std::string& model_id);
  const WeightAllocation* find(const std::string& model_id) const;
  const std::unordered_map<std::string, WeightAllocation>& models() const {
    return models_;
  }
  void clear();

 private:
  std::unordered_map<std::string, WeightAllocation> models_;
};

}  // namespace xllm
