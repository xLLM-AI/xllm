/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace xllm {

class Stream;

enum class ExecutionStreamRole : int8_t {
  COMMUNICATION,
  AUXILIARY_COMPUTE,
};

// Owns streams shared by execution components within one model instance.
class ModelStreamRegistry final {
 public:
  explicit ModelStreamRegistry(const torch::Device& device);

  std::shared_ptr<Stream> get(ExecutionStreamRole role);

 private:
  torch::Device device_;
  std::mutex mutex_;
  std::unordered_map<ExecutionStreamRole, std::shared_ptr<Stream>> streams_;
};

}  // namespace xllm
