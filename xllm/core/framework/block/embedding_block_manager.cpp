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

#include "core/framework/block/embedding_block_manager.h"

#include <glog/logging.h>

#include "core/framework/request/sequence.h"

namespace xllm {

EmbeddingBlockManager::EmbeddingBlockManager(uint32_t num_blocks)
    : BlockManagerImpl([num_blocks] {
        Options options;
        options.num_blocks(num_blocks)
            .block_size(/*value=*/1)
            .enable_prefix_cache(/*value=*/false)
            .block_type(/*value=*/BlockType::EMBEDDING);
        return options;
      }()) {}

std::optional<std::vector<Block>> EmbeddingBlockManager::allocate_for_sequence(
    Sequence* seq,
    KVCacheState& kv_state,
    size_t /*num_tokens*/) {
  if (seq == nullptr) {
    return std::nullopt;
  }
  // One block per sequence, reused for its lifetime.
  if (kv_state.num_blocks(block_type()) > 0) {
    return std::vector<Block>{};
  }
  std::vector<Block> blocks = allocate(1);
  if (blocks.empty()) {
    LOG(ERROR) << "Failed to allocate embedding block! free="
               << num_free_blocks() << ", used=" << num_used_blocks()
               << ", total=" << num_total_blocks();
    return std::nullopt;
  }
  return blocks;
}

}  // namespace xllm
