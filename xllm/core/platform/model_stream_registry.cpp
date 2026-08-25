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

#include "core/platform/model_stream_registry.h"

#include "core/platform/device.h"

namespace xllm {

ModelStreamRegistry::ModelStreamRegistry(const torch::Device& device)
    : device_(device) {}

std::shared_ptr<Stream> ModelStreamRegistry::get(ExecutionStreamRole role) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = streams_.find(role);
  if (it != streams_.end()) {
    return it->second;
  }

  Device device(device_);
  std::shared_ptr<Stream> stream(device.get_stream_from_pool());
  streams_.emplace(role, stream);
  return stream;
}

}  // namespace xllm
