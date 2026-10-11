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
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/common/types.h"
#include "core/framework/model_loader/weight/weight_memory_manager.h"
#include "core/kv_cache/storage/kv_cache_memory_registry.h"

namespace xllm {

// Owns worker-local KV cache mappings and weight allocations under one device
// lifecycle. Distributed operations are coordinated by a separate service.
class WorkerMemoryResources final {
 public:
  static WorkerMemoryResources& get_instance() {
    static WorkerMemoryResources instance;
    return instance;
  }

  void init(const torch::Device& device);
  void init_physical_page_pool(int64_t num_pages);
  bool is_initialized() const { return initialized_; }
  const torch::Device& device() const { return device_; }
  KVCacheMemoryRegistry& kv_cache_memory() { return kv_cache_memory_; }

  std::vector<torch::Tensor> create_k_tensors(const std::string& model_id,
                                              const std::vector<int64_t>& dims,
                                              torch::Dtype dtype,
                                              int64_t num_layers);
  std::vector<torch::Tensor> create_v_tensors(const std::string& model_id,
                                              const std::vector<int64_t>& dims,
                                              torch::Dtype dtype,
                                              int64_t num_layers);
  bool map_to_kv_tensors(const std::string& model_id,
                         const std::vector<offset_t>& offsets);
  bool unmap_from_kv_tensors(const std::string& model_id,
                             const std::vector<offset_t>& offsets);

  bool alloc_weight_pages(const std::string& model_id, size_t num_pages);
  bool allocate_weight(const std::string& model_id, void*& ptr, size_t size);
  size_t free_weight(const std::string& model_id);

  std::pair<uint64_t, uint64_t> get_global_offsets_for_block(
      const std::string& model_id,
      int64_t layer_id,
      int64_t block_id,
      size_t block_size);
  std::optional<int64_t> get_kv_cache_num_layers(
      const std::string& model_id) const;
  // The allocation snapshot borrows memory owned by this worker.
  std::optional<WeightAllocationInfo> get_weight_allocation_info(
      const std::string& model_id) const;
  std::vector<WeightSegment> get_model_weight_segments(
      const std::string& model_id) const;
  std::unordered_map<std::string, std::vector<WeightSegment>>
  get_all_model_weight_segments() const;

 private:
  friend class DistributedMemoryCoordinator;
  friend class WorkerMemoryResourcesTestPeer;

  WorkerMemoryResources();
  ~WorkerMemoryResources();
  WorkerMemoryResources(const WorkerMemoryResources&) = delete;
  WorkerMemoryResources& operator=(const WorkerMemoryResources&) = delete;

  void init_device_();
  void destroy();

  bool initialized_ = false;
  torch::Device device_{torch::kCPU};
  mutable std::mutex mutex_;
  KVCacheMemoryRegistry kv_cache_memory_;
  WeightMemoryManager weight_memory_;
};

}  // namespace xllm
