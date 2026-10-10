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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace xllm {

class VirtualMemoryDistClient;
class VirtualMemoryDistServer;

class VirtualMemoryCluster final {
 public:
  VirtualMemoryCluster() = default;
  ~VirtualMemoryCluster();

  VirtualMemoryCluster(const VirtualMemoryCluster&) = delete;
  VirtualMemoryCluster& operator=(const VirtualMemoryCluster&) = delete;

  void configure(int32_t world_size, int32_t dp_size, int32_t tp_size);
  void clear();

  int32_t world_size() const { return world_size_; }
  int32_t dp_size() const { return dp_size_; }
  int32_t tp_size() const { return tp_size_; }

  std::vector<std::vector<std::shared_ptr<VirtualMemoryDistClient>>>&
  dp_group_clients() {
    return dp_group_clients_;
  }
  const std::vector<std::vector<std::shared_ptr<VirtualMemoryDistClient>>>&
  dp_group_clients() const {
    return dp_group_clients_;
  }
  std::vector<std::shared_ptr<VirtualMemoryDistClient>>& clients() {
    return clients_;
  }
  const std::vector<std::shared_ptr<VirtualMemoryDistClient>>& clients() const {
    return clients_;
  }
  std::vector<std::unique_ptr<VirtualMemoryDistServer>>& servers() {
    return servers_;
  }
  const std::vector<std::unique_ptr<VirtualMemoryDistServer>>& servers() const {
    return servers_;
  }
  const std::string& collective_server_name() const {
    return collective_server_name_;
  }

 private:
  int32_t world_size_ = 0;
  int32_t dp_size_ = 1;
  int32_t tp_size_ = 1;
  std::vector<std::vector<std::shared_ptr<VirtualMemoryDistClient>>>
      dp_group_clients_;
  std::vector<std::shared_ptr<VirtualMemoryDistClient>> clients_;
  std::vector<std::unique_ptr<VirtualMemoryDistServer>> servers_;
  std::string collective_server_name_{"VirtualMemoryDistCollectiveServer"};
};

}  // namespace xllm
