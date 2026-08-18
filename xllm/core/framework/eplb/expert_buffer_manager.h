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

#pragma once

#include <torch/torch.h>

#include <vector>

#include "expert_weight_buffer_shm.h"

namespace xllm {

class ExpertBufferManager {
 public:
  ExpertBufferManager(const std::string& service_namespace,
                      int32_t num_experts,
                      int32_t num_layers,
                      int64_t shm_size_per_expert);

  void add_tensor(int32_t expert_id,
                  int32_t layer_id,
                  const std::string& tensor_name,
                  const torch::Tensor& tensor);

  torch::Tensor get_tensor(int32_t expert_id,
                           int32_t layer_id,
                           const std::string& tensor_name);

 private:
  const std::string service_namespace_;
  std::vector<std::unique_ptr<ExpertBufferShm>> expert_buffers_;
  const int32_t num_experts_;
  const int32_t num_layers_;
  const int64_t shm_size_per_expert_;
};

}  // namespace xllm
