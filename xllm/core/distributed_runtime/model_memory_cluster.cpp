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

#include "core/distributed_runtime/model_memory_cluster.h"

#include "core/distributed_runtime/model_memory_dist_client.h"
#include "core/distributed_runtime/model_memory_dist_server.h"

namespace xllm {

ModelMemoryCluster::~ModelMemoryCluster() = default;

void ModelMemoryCluster::configure(int32_t world_size,
                                   int32_t dp_size,
                                   int32_t tp_size) {
  world_size_ = world_size;
  dp_size_ = dp_size;
  tp_size_ = tp_size;
  dp_group_clients_.clear();
  dp_group_clients_.resize(dp_size_);
  for (auto& clients : dp_group_clients_) {
    clients.reserve(tp_size_);
  }
  clients_.clear();
  clients_.reserve(world_size_);
}

void ModelMemoryCluster::clear() {
  dp_group_clients_.clear();
  clients_.clear();
  servers_.clear();
  world_size_ = 0;
  dp_size_ = 1;
  tp_size_ = 1;
}

}  // namespace xllm
