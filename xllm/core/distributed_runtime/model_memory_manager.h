/* Copyright 2025-2026 The xLLM Authors.

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
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/common/types.h"
#include "core/distributed_runtime/model_memory_cluster.h"
#include "core/distributed_runtime/model_memory_dist_client.h"
#include "core/distributed_runtime/model_memory_dist_server.h"
#include "core/distributed_runtime/model_memory_options.h"
#include "core/framework/model_loader/weight/weight_memory_manager.h"
#include "core/kv_cache/storage/kv_cache_memory_registry.h"

namespace xllm {

/**
 * Coordinates worker-local KV cache mappings and model weight reservations.
 * Allocation backends own the mappings and weight state; this runtime facade
 * supplies synchronization, device configuration, distributed RPCs and page
 * coordination.
 */
class ModelMemoryManager final {
 public:
  // Get the global singleton instance
  static ModelMemoryManager& get_instance() {
    static ModelMemoryManager instance;
    return instance;
  }

  // Initialize the allocator with device configuration
  void init(const torch::Device& device);

  // Check if initialized
  bool is_initialized() const { return initialized_; }

  KVCacheMemoryRegistry& kv_cache_memory() { return kv_cache_memory_; }

  // ============== KV Cache Interfaces ==============

  // Create K tensors for all layers of a model
  std::vector<torch::Tensor> create_k_tensors(const std::string& model_id,
                                              const std::vector<int64_t>& dims,
                                              torch::Dtype dtype,
                                              int64_t num_layers);

  // Create V tensors for all layers of a model
  std::vector<torch::Tensor> create_v_tensors(const std::string& model_id,
                                              const std::vector<int64_t>& dims,
                                              torch::Dtype dtype,
                                              int64_t num_layers);

  // KV tensor operations (partial mapping by offsets)
  bool map_to_kv_tensors(const std::string& model_id,
                         const std::vector<offset_t>& offsets);
  bool unmap_from_kv_tensors(const std::string& model_id,
                             const std::vector<offset_t>& offsets);

  // ============== Weight Allocation Interfaces ==============

  bool alloc_weight_pages(const std::string& model_id, size_t num_pages);

  // Allocate from pre-allocated weight region (called by model loader)
  // Increments offset within the pre-allocated region
  bool allocate_weight(const std::string& model_id, void*& ptr, size_t size);

  // Free weight allocation (called by sleep), including both contiguous
  // GlobalMemoryRegion and fallback MappedMemoryRegion allocations.
  // Returns the number of pages freed.
  size_t free_weight(const std::string& model_id);

  // ============== Multi-node Setup ==============

  // Multi-node model memory setup (called by rank0 to connect to
  // other workers)
  void setup_multi_node_model_memory_dist(const ModelMemoryOptions& options,
                                          const std::string& master_node_addr,
                                          int32_t dp_size);

  // Initialize PhysicalPagePool on all workers
  int64_t init_physical_page_pools(double max_memory_utilization = 0.9,
                                   int64_t max_cache_size = 0);

  // ============== Model Parallel Strategy ==============

  // Set model-specific parallel strategy (for fork master with different dp/tp)
  // This should be called before broadcast operations for the model
  void set_model_parallel_strategy(const std::string& model_id,
                                   int32_t dp_size,
                                   int32_t tp_size);

  // Get model-specific parallel strategy (returns global values if not set)
  // Returns {dp_size, tp_size}
  std::pair<int32_t, int32_t> get_model_parallel_strategy(
      const std::string& model_id);

  // ============== Broadcast Operations ==============

  // Broadcast KV tensor map/unmap to workers in a specific DP group
  bool broadcast_map_to_kv_tensors(const std::string& model_id,
                                   int32_t dp_rank,
                                   const std::vector<offset_t>& offsets);
  bool broadcast_unmap_from_kv_tensors(const std::string& model_id,
                                       int32_t dp_rank,
                                       const std::vector<offset_t>& offsets);

  // Broadcast weight pages allocation/free to all workers
  bool broadcast_alloc_weight_pages(const std::string& model_id,
                                    size_t num_pages);
  bool broadcast_free_weight_pages(const std::string& model_id);

  // Get model memory RPC clients (for distributed operations)
  const std::vector<std::shared_ptr<ModelMemoryDistClient>>&
  get_model_memory_dist_clients() const {
    return cluster_.clients();
  }

  // Get device
  const torch::Device& device() const { return dev_; }

  // Get MappedMemoryRegion offsets for blocks via RPC (used by Engine in PD
  // disaggregation) Calls worker in the specified DP group to compute offsets
  // Parameters:
  //   dp_rank: Target DP rank (which DP group to query)
  //   model_id: Model identifier
  //   block_ids: Block IDs to get offsets for
  //   block_size_bytes: Size of each block in bytes
  //   layer_offsets: Output, layer_offsets[layer_id] = {k_offsets, v_offsets}
  // Returns: true on success
  bool get_kv_cache_offsets(
      int32_t dp_rank,
      const std::string& model_id,
      const std::vector<int32_t>& block_ids,
      uint64_t block_size_bytes,
      std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>&
          layer_offsets);

  // ============== PD Disaggregation Support (MappedMemoryRegion Mode)
  // ==============

  // Convert a block_id to GlobalMemoryRegion offsets for KV cache transfer.
  // This is only used when --enable_virtual_memory=true for PD disaggregation.
  //
  // Parameters:
  //   model_id: Model identifier
  //   layer_id: Layer index
  //   block_id: Block ID within the KV cache
  //   block_size: Size of each block in bytes
  //
  // Returns: {k_offset, v_offset} relative to GlobalMemoryRegion base address,
  //          or {UINT64_MAX, UINT64_MAX} on error.
  std::pair<uint64_t, uint64_t> get_global_offsets_for_block(
      const std::string& model_id,
      int64_t layer_id,
      int64_t block_id,
      size_t block_size);

  // Return a copy of cache metadata without exposing mutable mappings.
  std::optional<int64_t> get_kv_cache_num_layers(
      const std::string& model_id) const;

  // Return a non-owning snapshot for weight transfer.
  std::optional<WeightAllocationInfo> get_weight_allocation_info(
      const std::string& model_id) const;

  // ============== ETCD Registration Support ==============
  // Get weight segments for a model (supports non-contiguous allocation)
  // Returns ordered list of {offset, size} segments in GlobalMemoryRegion
  std::vector<WeightSegment> get_model_weight_segments(
      const std::string& model_id) const;

  // Get all model weight segments
  std::unordered_map<std::string, std::vector<WeightSegment>>
  get_all_model_weight_segments() const;

 private:
  friend class ModelMemoryManagerTestPeer;

  ModelMemoryManager();
  ~ModelMemoryManager();
  ModelMemoryManager(const ModelMemoryManager&) = delete;
  ModelMemoryManager& operator=(const ModelMemoryManager&) = delete;

  // Device initialization (platform-agnostic)
  void init_device_();

  // Cleanup resources
  void destroy();

  bool initialized_ = false;
  torch::Device dev_{torch::kCPU};

  mutable std::mutex mtx_;

  KVCacheMemoryRegistry kv_cache_memory_;
  WeightMemoryManager weight_memory_;
  std::unordered_map<std::string, std::pair<int32_t, int32_t>>
      model_parallel_strategies_;
  ModelMemoryCluster cluster_;
};

}  // namespace xllm
