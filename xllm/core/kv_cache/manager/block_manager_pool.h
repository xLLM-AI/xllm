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

#include <atomic>
#include <optional>
#include <string>
#include <vector>

#include "core/kv_cache/events/kv_cache_event_publisher.h"
#include "core/kv_cache/layout/paged_cache_partition.h"
#include "core/kv_cache/layout/recurrent_cache_partition.h"
#include "core/kv_cache/manager/composite_block_manager.h"
#include "core/kv_cache/manager/kv_cache_manager.h"
#include "core/kv_cache/manager/kv_cache_manager_options.h"

namespace xllm {

class VmmKVPageAllocator;

class BlockManagerPool : public KVCacheManager {
 public:
  using Options = KVCacheManagerOptions;
  using HostBlockCounts = Options::HostBlockCounts;

  explicit BlockManagerPool(
      const Options& options,
      int32_t dp_size = 1,
      VmmKVPageAllocator* page_allocator = nullptr,
      std::optional<PagedCachePartition> paged_partition = std::nullopt,
      std::optional<RecurrentCachePartition> recurrent_partition =
          std::nullopt);

  ~BlockManagerPool() override = default;

  bool allocate(Sequence* sequence) override;
  bool allocate(std::vector<Sequence*>& sequences) override;
  bool allocate(Sequence* sequence, size_t num_tokens) override;
  bool exceeds_total_capacity(Sequence* sequence,
                              size_t num_tokens) const override;
  bool exceeds_total_capacity(
      const std::vector<KVCacheReservation>& reservations) const override;
  // Try to allocate blocks with num_tokens,
  // return {} if not enough blocks
  std::vector<Block> allocate(size_t num_tokens, int32_t& dp_rank) override;

  bool try_allocate(Sequence* sequence) override;

  void deallocate(Request* request) override;
  void deallocate(std::vector<Sequence*>& sequences) override;
  void deallocate(Sequence* sequence) override;

  bool allocate_for_profiling(Sequence* sequence, size_t num_tokens) override;
  void deallocate_without_cache(Sequence* sequence) override;
  bool exceeds_decode_capacity(Sequence* sequence) const override;
  bool uses_virtual_memory() const override {
    return options_.enable_vmm_kvcache();
  }
  const std::string& model_id() const override { return options_.model_id(); }

  void allocate_shared(Sequence* sequence) override;
  void publish_prefix(Sequence* sequence) override;
  void publish_prefix(Sequence* sequence, size_t num_tokens) override;

  float average_cache_utilization() const override;

  uint32_t num_blocks() const override;
  size_t block_resource_units(BlockType type, size_t count) const override;
  size_t admission_capacity(int32_t dp_rank) const override;
  int32_t block_size() const override;
  std::vector<size_t> num_blocks_in_prefix_cache() const override;
  std::vector<size_t> num_free_blocks() const override;
  std::vector<size_t> num_used_blocks() const override;
  double kv_cache_utilization() const override;

  int32_t select_dp_rank() const override;

  // The paged partition is bound to the KV leaf in every DP pool.
  const std::optional<PagedCachePartition>& paged_partition() const {
    return paged_partition_;
  }

  const std::optional<RecurrentCachePartition>& recurrent_partition() const {
    return recurrent_partition_;
  }

  // get the options for the block manager
  const Options& options() const { return options_; }

  // Reserve VirtualMemory padding blocks for each DP manager.
  // Should be called after KV tensors are created.
  void reserve_virtual_memory_padding_blocks() override;

 protected:
  // Select the DP rank with the most effective headroom. Prefix-cache-only
  // blocks count as available because they can be evicted during allocation;
  // equal candidates are selected round-robin.
  int32_t get_manager_with_max_available_blocks() const;
  int32_t get_dp_rank(Sequence* sequence) const;

 private:
  friend class BlockManagerPoolTestPeer;

  // Non-owning optional dependency, required for virtual-memory cache leaves.
  // The runtime retains it until this pool and its blocks have been destroyed.
  VmmKVPageAllocator* page_allocator_;
  std::optional<PagedCachePartition> paged_partition_;
  std::optional<RecurrentCachePartition> recurrent_partition_;
  mutable std::atomic<size_t> dp_selection_cursor_{0};

 protected:
  // the options for the block manager
  Options options_;
  std::unique_ptr<KVCacheEventPublisher> event_publisher_;
  std::vector<std::unique_ptr<CompositeBlockManager>> block_managers_;
};

}  // namespace xllm
