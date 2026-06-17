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

#include <memory>
#include <vector>

#include "block_manager.h"

namespace xllm {

// BlockManager composed of multiple sub-managers, created from options.
// manager_type: 0 = BlockManagerImpl, 1 = SlidingWindowBlockManager.
// Options come from BlockManager::Options (passed from upstream via
// BlockManagerPool).
class CompositeBlockManager : public BlockManager {
 public:
  explicit CompositeBlockManager(const BlockManager::Options& options);
  ~CompositeBlockManager() override = default;

  bool is_composite() const override { return true; }
  bool allocate_for_sequence(Sequence* seq, size_t num_tokens) override;
  void deallocate_sequence(Sequence* seq) override;

  void deallocate(const Slice<Block>& blocks) override;
  std::vector<Block> allocate(size_t num_blocks) override;
  std::vector<Block> allocate_shared(
      const Slice<int32_t>& tokens_ids,
      const Slice<Block>& existed_shared_blocks = {},
      const MMData& mm_data = MMData()) override;
  void cache(const Slice<int32_t>& token_ids,
             std::vector<Block>& blocks,
             size_t existed_shared_blocks_num = 0,
             const MMData& mm_data = MMData()) override;
  void cache(const std::vector<Block>& blocks) override;
  void get_merged_kvcache_event(KvCacheEvent* event) const override;
  size_t num_blocks_in_prefix_cache() const override;
  size_t num_free_blocks() const override;
  size_t num_used_blocks() const override;
  double kv_cache_utilization() const override;
  void free(int32_t block_id) override;
  Block allocate() override;
  size_t num_total_blocks() const override;

  size_t num_sub_managers() const { return sub_managers_.size(); }

 private:
  std::vector<std::unique_ptr<BlockManager>> sub_managers_;
};

}  // namespace xllm
