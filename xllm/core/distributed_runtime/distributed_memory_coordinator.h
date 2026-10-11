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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/distributed_runtime/worker_memory_rpc_options.h"
#include "core/framework/allocator/worker_page_budget.h"
#include "core/kv_cache/storage/kv_cache_page_backend.h"

namespace xllm {

class WorkerMemoryRpcClient;
class WorkerMemoryRpcServer;
class PagedKVCachePageAllocator;
struct KVCacheMappingPlan;

// Coordinates capacity and mapping commands across workers. Worker-local
// resources remain owned by WorkerMemoryResources in the runtime layer.
class DistributedMemoryCoordinator final : public KVCachePageBackend {
 public:
  static DistributedMemoryCoordinator& get_instance() {
    static DistributedMemoryCoordinator instance;
    return instance;
  }

  void setup_worker_memory_rpc(const WorkerMemoryRpcOptions& options,
                               const std::string& master_node_addr,
                               int32_t dp_size);
  int64_t init_physical_page_pools(double max_memory_utilization = 0.9,
                                   int64_t max_cache_size = 0);

  void init_page_budget(size_t total_pages,
                        int32_t dp_size,
                        int32_t worker_count,
                        size_t page_size,
                        bool enable_prealloc = true);
  bool is_initialized() const;
  size_t get_num_total_phy_pages() const;
  PagedKVCachePageAllocator& kv_cache_page_allocator();

  bool reserve_weight_pages(const std::string& model_id, size_t num_pages);
  void set_weight_page_count(const std::string& model_id, size_t num_pages);
  size_t get_weight_page_count(const std::string& model_id) const;
  void release_weight_pages(const std::string& model_id);
  void restore_weight_pages(const std::string& model_id);

  // Reserve every worker's KV and weight demand in one transaction. The KV
  // allocator must freeze the mapping plan throughout the residency transition.
  bool try_reserve_model_memory(const std::string& model_id,
                                const KVCacheMappingPlan& plan);
  void release_kv_mapping_budget(const std::string& model_id,
                                 const KVCacheMappingPlan& plan);
  std::vector<size_t> get_worker_free_page_counts() const;
  size_t get_free_pages_for_model(const std::string& model_id) const;

  bool try_reserve_pages(const std::string& model_id,
                         int32_t dp_rank,
                         size_t physical_pages) override;
  void release_pages(const std::string& model_id,
                     int32_t dp_rank,
                     size_t physical_pages) override;
  size_t available_pages(const std::string& model_id,
                         int32_t dp_rank) const override;
  bool map_pages(const std::string& model_id,
                 int32_t dp_rank,
                 const std::vector<int64_t>& offsets) override;
  bool unmap_pages(const std::string& model_id,
                   int32_t dp_rank,
                   const std::vector<int64_t>& offsets) override;

  void set_model_parallel_strategy(const std::string& model_id,
                                   int32_t dp_size,
                                   int32_t tp_size);
  std::pair<int32_t, int32_t> get_model_parallel_strategy(
      const std::string& model_id) const;

  bool get_kv_cache_offsets(
      int32_t dp_rank,
      const std::string& model_id,
      const std::vector<int32_t>& block_ids,
      uint64_t block_size_bytes,
      std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>&
          layer_offsets);

 private:
  friend class DistributedMemoryCoordinatorTestPeer;
  DistributedMemoryCoordinator();
  ~DistributedMemoryCoordinator() override;
  DistributedMemoryCoordinator(const DistributedMemoryCoordinator&) = delete;
  DistributedMemoryCoordinator& operator=(const DistributedMemoryCoordinator&) =
      delete;

  bool broadcast_map_to_kv_tensors(const std::string& model_id,
                                   int32_t dp_rank,
                                   const std::vector<int64_t>& offsets);
  bool broadcast_unmap_from_kv_tensors(const std::string& model_id,
                                       int32_t dp_rank,
                                       const std::vector<int64_t>& offsets);
  bool broadcast_alloc_weight_pages(const std::string& model_id,
                                    size_t num_pages);
  bool broadcast_free_weight_pages(const std::string& model_id);

  void configure_workers(int32_t world_size, int32_t dp_size, int32_t tp_size);
  void destroy();
  std::pair<int32_t, int32_t> parallel_strategy_locked(
      const std::string& model_id) const;
  std::pair<int32_t, int32_t> dp_worker_range_locked(
      const std::string& model_id,
      int32_t dp_rank) const;
  int32_t model_worker_count_locked(const std::string& model_id) const;
  void update_memory_usage_locked() const;

  // No call into the KV allocator may hold this mutex. KV page operations can
  // enter this backend while holding the allocator's own mutex.
  mutable std::mutex mutex_;
  // Serialize initialization without holding the backend budget lock.
  std::mutex initialization_mutex_;
  WorkerPageBudget page_budget_;
  bool budget_initialized_ = false;
  std::unordered_map<std::string, size_t> weight_page_counts_;
  std::unordered_map<std::string, std::pair<int32_t, int32_t>>
      model_parallel_strategies_;
  int32_t world_size_ = 0;
  int32_t dp_size_ = 1;
  int32_t tp_size_ = 1;
  std::vector<std::shared_ptr<WorkerMemoryRpcClient>> clients_;
  std::vector<std::unique_ptr<WorkerMemoryRpcServer>> servers_;
  std::string collective_server_name_{"WorkerMemoryRpcCollectiveServer"};
  std::unique_ptr<PagedKVCachePageAllocator> kv_cache_pages_;
};

}  // namespace xllm
