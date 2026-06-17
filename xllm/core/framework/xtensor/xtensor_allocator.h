/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/types.h"
#include "options.h"
#include "phy_page.h"
#include "xtensor.h"
#include "xtensor_dist_client.h"
#include "xtensor_dist_server.h"

namespace xllm {

/**
 * Per-model tensor storage
 */
struct ModelTensors {
  // K tensors: one tensor per layer (indexed by layer id)
  std::vector<std::unique_ptr<XTensor>> k_tensors;
  // V tensors: one tensor per layer (indexed by layer id)
  std::vector<std::unique_ptr<XTensor>> v_tensors;
  int64_t num_layers = 0;
  size_t kv_tensor_size_per_layer = 0;

  // ============== Weight Allocation (from GlobalXTensor) ==============
  page_id_t weight_start_page_id =
      -1;                            // Starting page ID of pre-allocated region
  size_t weight_num_pages = 0;       // Number of pages pre-allocated
  void* weight_base_ptr = nullptr;   // Base virtual address
  size_t weight_current_offset = 0;  // Current allocation offset in bytes

  // ============== Fallback Weight Allocation (XTensor with preallocated pages)
  // Used when contiguous allocation fails due to fragmentation
  std::unique_ptr<XTensor> weight_xtensor;
  bool using_weight_xtensor = false;  // True if using XTensor fallback

  // ============== Model-specific Parallel Strategy (for fork master)
  // ============== Each model may have different dp_size/tp_size, used in
  // broadcast operations to select correct workers. 0 means use global values.
  int32_t dp_size = 0;
  int32_t tp_size = 0;

  // ============== Weight Segments (for P2P transfer) ==============
  // Ordered list of weight segments in GlobalXTensor.
  // For contiguous allocation: single segment.
  // For fallback (XTensor): multiple segments from non-contiguous pages.
  std::vector<WeightSegment> weight_segments;
};

/**
 * XTensorAllocator manages XTensor objects for KV cache and model weights.
 *
 * This is a singleton class that:
 * - Creates and manages XTensor objects per model (indexed by model_id)
 * - Handles distributed XTensor operations via RPC
 * - Coordinates PhyPagePool initialization across workers
 */
class XTensorAllocator {
 public:
  // Get the global singleton instance
  static XTensorAllocator& get_instance() {
    static XTensorAllocator instance;
    return instance;
  }

  // Initialize the allocator with device configuration
  void init(const torch::Device& device);

  // Check if initialized
  bool is_initialized() const { return initialized_; }

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

  // Record weight pre-allocation (called by RPC handler after PhyPagePool
  // allocation)
  void record_weight_allocation(const std::string& model_id,
                                page_id_t start_page_id,
                                size_t num_pages);

  // Record weight allocation using fallback mode (non-contiguous pages)
  // Used when contiguous allocation fails due to fragmentation
  void record_weight_fallback_allocation(
      const std::string& model_id,
      const std::vector<page_id_t>& page_ids);

  // Allocate from pre-allocated weight region (called by model loader)
  // Increments offset within the pre-allocated region
  bool allocate_weight(const std::string& model_id, void*& ptr, size_t size);

  // Free weight allocation (called by sleep), including both contiguous
  // GlobalXTensor and fallback XTensor allocations.
  // Returns the number of pages freed.
  size_t free_weight(const std::string& model_id);

  // ============== Multi-node Setup ==============

  // Multi-node XTensor dist setup (called by rank0 to connect to other workers)
  void setup_multi_node_xtensor_dist(const xtensor::Options& options,
                                     const std::string& master_node_addr,
                                     int32_t dp_size);

  // Initialize PhyPagePool on all workers
  int64_t init_phy_page_pools(double max_memory_utilization = 0.9,
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

  // Get XTensor dist clients (for distributed operations)
  const std::vector<std::shared_ptr<XTensorDistClient>>&
  get_xtensor_dist_clients() const {
    return xtensor_dist_clients_;
  }

  // Get device
  const torch::Device& device() const { return dev_; }

  // Get XTensor offsets for blocks via RPC (used by Engine in PD
  // disaggregation) Calls worker in the specified DP group to compute offsets
  // Parameters:
  //   dp_rank: Target DP rank (which DP group to query)
  //   model_id: Model identifier
  //   block_ids: Block IDs to get offsets for
  //   block_size_bytes: Size of each block in bytes
  //   layer_offsets: Output, layer_offsets[layer_id] = {k_offsets, v_offsets}
  // Returns: true on success
  bool get_xtensor_offsets(
      int32_t dp_rank,
      const std::string& model_id,
      const std::vector<int32_t>& block_ids,
      uint64_t block_size_bytes,
      std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>&
          layer_offsets);

  // ============== PD Disaggregation Support (XTensor Mode) ==============

  // Convert a block_id to GlobalXTensor offsets for KV cache transfer.
  // This is only used when --enable_xtensor=true for PD disaggregation.
  //
  // Parameters:
  //   model_id: Model identifier
  //   layer_id: Layer index
  //   block_id: Block ID within the KV cache
  //   block_size: Size of each block in bytes
  //
  // Returns: {k_offset, v_offset} relative to GlobalXTensor base address,
  //          or {UINT64_MAX, UINT64_MAX} on error.
  std::pair<uint64_t, uint64_t> get_global_offsets_for_block(
      const std::string& model_id,
      int64_t layer_id,
      int64_t block_id,
      size_t block_size);

  // Get model tensors (returns nullptr if not found)
  // Public for XTensorDistService to access num_layers
  ModelTensors* get_model_tensors(const std::string& model_id);

  // ============== ETCD Registration Support ==============
  // Get weight segments for a model (supports non-contiguous allocation)
  // Returns ordered list of {offset, size} segments in GlobalXTensor
  std::vector<WeightSegment> get_model_weight_segments(
      const std::string& model_id) const;

  // Get all model weight segments
  std::unordered_map<std::string, std::vector<WeightSegment>>
  get_all_model_weight_segments() const;

 private:
  XTensorAllocator() = default;
  ~XTensorAllocator();
  XTensorAllocator(const XTensorAllocator&) = delete;
  XTensorAllocator& operator=(const XTensorAllocator&) = delete;

  // Get or create model tensors (auto-creates if not exists)
  ModelTensors& get_or_create_model_tensors(const std::string& model_id);

  // Create K/V tensors implementation (handles lock and validation)
  // name: "K", "V", or future types like "index"
  std::vector<torch::Tensor> create_kv_tensors_impl_(
      const std::string& model_id,
      const std::vector<int64_t>& dims,
      torch::Dtype dtype,
      int64_t num_layers,
      const char* name);

  // Create tensors internal (must call with lock held)
  std::vector<torch::Tensor> create_tensors_internal_(
      size_t size,
      const std::vector<int64_t>& dims,
      torch::Dtype dtype,
      int64_t num_layers,
      std::vector<std::unique_ptr<XTensor>>& tensors_out);

  // Device initialization (platform-agnostic)
  void init_device_();

  // Cleanup resources
  void destroy();

  bool initialized_ = false;
  torch::Device dev_{torch::kCPU};

  mutable std::mutex mtx_;

  // Per-model tensors storage (key: model_id)
  std::unordered_map<std::string, ModelTensors> model_tensors_;

  // Multi-node XTensor dist members
  int32_t world_size_ = 0;  // total workers = dp_size * tp_size
  int32_t dp_size_ = 1;
  int32_t tp_size_ = 1;
  // DP group to worker clients mapping: dp_group_clients_[dp_rank][tp_rank]
  std::vector<std::vector<std::shared_ptr<XTensorDistClient>>>
      dp_group_clients_;
  // Flat list for backward compatibility and weight tensor broadcast
  std::vector<std::shared_ptr<XTensorDistClient>> xtensor_dist_clients_;
  std::vector<std::unique_ptr<XTensorDistServer>> xtensor_dist_servers_;
  std::string collective_server_name_{"XTensorAllocatorCollectiveServer"};
};

}  // namespace xllm
