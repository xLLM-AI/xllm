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

#include <torch/types.h>

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/xtensor/weight_allocation.h"

namespace xllm {

class MappedMemoryRegion;

struct ModelTensors {
  std::vector<std::unique_ptr<MappedMemoryRegion>> k_tensors;
  std::vector<std::unique_ptr<MappedMemoryRegion>> v_tensors;
  int64_t num_layers = 0;
  size_t kv_tensor_size_per_layer = 0;
  WeightAllocation weight;
  int32_t dp_size = 0;
  int32_t tp_size = 0;
};

class ModelMemoryStore final {
 public:
  ModelMemoryStore() = default;
  ~ModelMemoryStore();

  ModelMemoryStore(const ModelMemoryStore&) = delete;
  ModelMemoryStore& operator=(const ModelMemoryStore&) = delete;

  ModelTensors& get_or_create(const std::string& model_id);
  ModelTensors* find(const std::string& model_id);
  const ModelTensors* find(const std::string& model_id) const;
  const std::unordered_map<std::string, ModelTensors>& models() const {
    return models_;
  }
  void clear();

 private:
  std::unordered_map<std::string, ModelTensors> models_;
};

}  // namespace xllm
