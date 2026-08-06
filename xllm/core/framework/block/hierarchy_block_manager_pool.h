/* Copyright 2025-2026 The xLLM Authors.

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

#include <map>

#include "block_manager_pool.h"
#include "composite_block_manager.h"
#include "distributed_runtime/engine.h"
#include "util/blockingconcurrentqueue.h"

namespace xllm {

class Engine;

// OffloadBlockPair carries the src/dst blocks (device + host) plus the block
// type so the completion callback can publish success to the correct Host leaf.
struct OffloadBlockPair {
  Block src;
  Block dst;
  BlockType block_type = BlockType::KV;
};

class HierarchyBlockManagerPool : public BlockManagerPool {
 public:
  using OffloadBlockPairQueue =
      moodycamel::BlockingConcurrentQueue<std::shared_ptr<OffloadBlockPair>>;

  explicit HierarchyBlockManagerPool(const BlockManagerPool::Options& options,
                                     Engine* engine,
                                     int32_t dp_size = 1);
  ~HierarchyBlockManagerPool() = default;

  bool allocate(Sequence* sequence, size_t num_tokens) override;

  void allocate_shared(Sequence* sequence) override;
  bool supports_host_cache_restore() const override { return true; }
  HostCacheRestorePoint select_host_cache_restore(
      Sequence* sequence,
      size_t max_copy_units) override;
  void trim_host_cache(Sequence* sequence,
                       const HostCacheRestorePoint& selected_restore) override;

  void deallocate(Sequence* sequence) override;

  void transfer_blocks(std::vector<Batch>& batches) override;
  void transfer_blocks() override;

  void prefetch_from_storage(std::shared_ptr<Request>& request) override;

  bool update_prefetch_result(std::shared_ptr<Request>& request,
                              const uint32_t timeout) override;

 private:
  friend class HierarchyPoolTestPeer;
  void release_host_match(Sequence* sequence, int32_t dp_rank);
  void collect_offload_pairs(Sequence* sequence,
                             int32_t dp_rank,
                             size_t completed_tokens);
  bool should_probe_prefix_cache(Sequence* sequence) const;
  void transfer_offload_blocks();

  BlockManager* leaf_of(BlockType type, int32_t dp_rank) const;

 private:
  Engine* engine_;
  // Per-DP Host block managers discovered from the device prefix-cache leaves.
  std::vector<CompositeBlockManager::LeafMap> host_block_managers_;

  // Per-DP H2D descriptions waiting to be registered with workers. Blocks stay
  // owned only by the Sequence's Host/device cache states.
  std::vector<std::vector<BlockTransferInfo>> load_block_transfer_infos_;
  std::vector<OffloadBlockPairQueue> offload_block_pair_queues_;
};

}  // namespace xllm
