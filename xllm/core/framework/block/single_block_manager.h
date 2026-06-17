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

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "block_manager.h"

namespace xllm {

class SingleBlockManager final : public BlockManager {
 public:
  SingleBlockManager(uint32_t num_blocks,
                     std::string resource_name,
                     std::string exhaustion_message = "");
  ~SingleBlockManager() override = default;

  std::vector<Block> allocate(size_t num_blocks) override;
  Block allocate() override;
  void deallocate(const Slice<Block>& blocks) override;

  std::vector<Block> allocate_shared(
      const Slice<int32_t>& token_ids,
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
  size_t num_total_blocks() const override;

 private:
  std::vector<int32_t> free_ids_;
  std::vector<bool> in_use_ids_;
  // Tracks whether `num_used_blocks_` currently includes this block id.
  // Semantics: `num_used_blocks()` reports logical/effective usage. It may
  // temporarily differ from `num_free_blocks()` because `deallocate()` can
  // drop effective usage before the last `Block` reference is released and
  // the id is physically returned to `free_ids_` via `free()`.
  std::vector<bool> usage_accounted_ids_;
  std::atomic<size_t> num_free_blocks_{0};
  std::atomic<size_t> num_used_blocks_{0};
  std::string resource_name_;
  std::string exhaustion_message_;
  mutable std::mutex mutex_;
};

}  // namespace xllm
