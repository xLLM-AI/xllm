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

#include <algorithm>
#include <cstdint>
#include <vector>

#include "core/kv_cache/block/linear_state_block_manager.h"
#include "framework/block/block_manager_pool.h"
#include "framework/block/composite_block_manager.h"

namespace xllm {

class BlockManagerPoolTestPeer final : public BlockManagerPool {
 public:
  using BlockManagerPool::BlockManagerPool;

  LinearStateBlockManager* linear_leaf(int32_t dp_rank) const {
    const auto& composite =
        static_cast<const CompositeBlockManager&>(*block_managers_[dp_rank]);
    return static_cast<LinearStateBlockManager*>(
        composite.leaf_of(/*type=*/BlockType::LINEAR));
  }

  static bool contains(const LinearStateBlockManager* leaf,
                       const XXH3Key& prefix_hash) {
    return std::ranges::any_of(leaf->prefix_cache_entries(),
                               [&prefix_hash](const auto& entry) {
                                 return entry.hash == prefix_hash;
                               });
  }
  static std::vector<Block> match(LinearStateBlockManager* leaf,
                                  const XXH3Key& prefix_hash) {
    const std::vector<int32_t> tokens(leaf->block_size() + 1);
    const Slice<XXH3Key> hashes(&prefix_hash, /*size=*/1);
    return leaf->allocate_shared(
        tokens, /*existed_shared_blocks=*/{}, /*mm_data=*/{}, hashes);
  }

  static int32_t insert_checkpoint(LinearStateBlockManager* leaf,
                                   const XXH3Key& hash) {
    auto checkpoint = leaf->allocate(/*num_blocks=*/1);
    if (checkpoint.empty()) {
      return -1;
    }
    Block& block = checkpoint.front();
    const int32_t id = block.id();
    block.set_hash_value(hash.data);
    leaf->cache(checkpoint);
    return id;
  }
};

}  // namespace xllm
