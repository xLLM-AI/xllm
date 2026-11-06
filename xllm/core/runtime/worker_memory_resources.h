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

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/common/types.h"
#include "core/framework/model_loader/weight/weight_virtual_memory_manager.h"
#include "core/kv_cache/storage/virtual_kv_cache_memory_registry.h"

namespace xllm {

struct WorkerMemoryCapacity {
  int64_t available_memory = 0;
  int64_t total_memory = 0;
  bool pool_initialized = false;
  size_t pool_capacity_pages = 0;
  size_t pool_committed_pages = 0;
  size_t pool_allocatable_pages = 0;
  size_t execution_memory_reserve_bytes = 0;
  size_t pool_active_pages = 0;
  size_t growth_memory_reserve_bytes = 0;
};

// Owns worker-local KV cache mappings and weight allocations under one device
// lifecycle. Distributed operations are coordinated by a separate service.
class WorkerMemoryResources final {
 public:
  static WorkerMemoryResources& get_instance() {
    static WorkerMemoryResources instance;
    return instance;
  }

  void init(const torch::Device& device);
  void init_physical_page_pool(int64_t num_pages, bool lazy = true);
  void init_physical_page_pool(int64_t max_pages,
                               int64_t initial_pages,
                               bool lazy,
                               size_t growth_memory_reserve_bytes);
  // Wait for the master's RPC to finish pool, reserve and mapping setup.
  // Returns the master's capacity, or zero on timeout or resource shutdown.
  int64_t wait_for_physical_page_pool(std::chrono::milliseconds timeout);
  WorkerMemoryCapacity memory_capacity_snapshot() const;
  void set_execution_memory_reserve(size_t bytes);
  size_t trim_idle_physical_pages(size_t keep_pages = 0);

  // Prepare owns real mappings until the entire worker group commits.
  bool prepare_kv_mapping(uint64_t transaction_id,
                          const std::string& model_id,
                          const std::vector<offset_t>& offsets);
  bool commit_kv_mapping(uint64_t transaction_id);
  bool abort_kv_mapping(uint64_t transaction_id);
  bool prepare_weight_pages(uint64_t transaction_id,
                            const std::string& model_id,
                            size_t num_pages);
  bool commit_weight_pages(uint64_t transaction_id);
  bool abort_weight_pages(uint64_t transaction_id);
  bool is_initialized() const { return initialized_; }
  const torch::Device& device() const { return device_; }
  VirtualKVCacheMemoryRegistry& kv_cache_memory() { return kv_cache_memory_; }

  std::vector<torch::Tensor> create_k_tensors(const std::string& model_id,
                                              const std::vector<int64_t>& dims,
                                              torch::ScalarType dtype,
                                              int64_t num_layers);
  std::vector<torch::Tensor> create_v_tensors(const std::string& model_id,
                                              const std::vector<int64_t>& dims,
                                              torch::ScalarType dtype,
                                              int64_t num_layers);
  bool map_to_kv_tensors(const std::string& model_id,
                         const std::vector<offset_t>& offsets);
  bool unmap_from_kv_tensors(const std::string& model_id,
                             const std::vector<offset_t>& offsets);

  bool alloc_weight_pages(const std::string& model_id, size_t num_pages);
  bool allocate_weight(const std::string& model_id, void*& ptr, size_t size);
  size_t free_weight(const std::string& model_id);

  std::optional<VirtualKVCacheBlockGlobalOffsets> get_global_offsets_for_block(
      const std::string& model_id,
      int64_t layer_id,
      int64_t block_id);
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
  friend class DistributedVmmCoordinator;
  friend class WorkerMemoryResourcesTestPeer;

  WorkerMemoryResources();
  ~WorkerMemoryResources();
  WorkerMemoryResources(const WorkerMemoryResources&) = delete;
  WorkerMemoryResources& operator=(const WorkerMemoryResources&) = delete;

  void init_device();
  void destroy();

  struct KvMappingTransaction {
    uint64_t id = 0;
    std::string model_id;
    std::vector<offset_t> offsets;
    bool committed = false;
  };
  struct WeightMappingTransaction {
    uint64_t id = 0;
    std::string model_id;
    size_t num_pages = 0;
    bool committed = false;
  };

  // Keep one reversible result and a retired watermark. Delayed RPCs must
  // never recreate mappings from an already aborted transaction.
  std::optional<KvMappingTransaction> mapping_transaction_;
  uint64_t retired_mapping_transaction_id_ = 0;
  std::optional<WeightMappingTransaction> weight_transaction_;
  uint64_t retired_weight_transaction_id_ = 0;
  size_t execution_memory_reserve_bytes_ = 0;
  size_t growth_memory_reserve_bytes_ = 0;
  int64_t pool_capacity_pages_ = 0;
  bool initialized_ = false;
  torch::Device device_{torch::kCPU};
  mutable std::mutex mutex_;
  std::condition_variable pool_ready_cv_;
  VirtualKVCacheMemoryRegistry kv_cache_memory_;
  WeightVirtualMemoryManager weight_memory_;
};

}  // namespace xllm
