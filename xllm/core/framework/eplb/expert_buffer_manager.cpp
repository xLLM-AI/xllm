/* Copyright 2025-2026 The xLLM Authors.

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

#include "expert_buffer_manager.h"

#include <glog/logging.h>

namespace xllm {

ExpertBufferManager::ExpertBufferManager(const std::string& service_namespace,
                                         int32_t num_experts,
                                         int32_t num_layers,
                                         int64_t shm_size_per_expert)
    : service_namespace_(service_namespace),
      num_experts_(num_experts),
      num_layers_(num_layers),
      shm_size_per_expert_(shm_size_per_expert) {
  expert_buffers_.reserve(num_experts);
  for (int32_t i = 0; i < num_experts; ++i) {
    expert_buffers_.emplace_back(std::make_unique<ExpertBufferShm>(
        service_namespace_, i, num_layers, shm_size_per_expert));
  }
}

void ExpertBufferManager::add_tensor(int32_t expert_id,
                                     int32_t layer_id,
                                     const std::string& tensor_name,
                                     const torch::Tensor& tensor) {
  CHECK(expert_id >= 0 && expert_id < num_experts_)
      << "Invalid expert ID: " << expert_id << " (num_experts=" << num_experts_
      << ")";
  expert_buffers_[expert_id]->add_tensor(layer_id, tensor_name, tensor);
}

torch::Tensor ExpertBufferManager::get_tensor(int32_t expert_id,
                                              int32_t layer_id,
                                              const std::string& tensor_name) {
  CHECK(expert_id >= 0 && expert_id < num_experts_)
      << "Invalid expert ID: " << expert_id << " (num_experts=" << num_experts_
      << ")";
  return expert_buffers_[expert_id]->get_tensor(layer_id, tensor_name);
}

}  // namespace xllm
