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

#include "core/kv_cache/block/block_manager_impl.h"

namespace xllm {

// Per-sequence single-resource leaf (BlockType::EMBEDDING): the spec-decode
// embedding-row id, exactly one block per sequence, reused for its lifetime.
//
// The physical id pool (free list, allocate / deallocate / free, num_*
// accounting, id-0 padding) is identical to BlockManagerImpl, so this derives
// from it and reuses that machinery. block_size is 1 (each id is one block) and
// there is no prefix cache. What differs is the sequence-level policy: grow
// allocates one block only when the sequence does not already hold one.
class EmbeddingBlockManager final : public BlockManagerImpl {
 public:
  explicit EmbeddingBlockManager(uint32_t num_blocks);
  ~EmbeddingBlockManager() override = default;

  // One block per sequence: empty when already held, else a single block (or
  // std::nullopt when the pool is exhausted). Does not insert into the sequence
  // -- the composite commits the returned block under BlockType::EMBEDDING.
  std::optional<std::vector<Block>> allocate_for_sequence(
      Sequence* seq,
      KVCacheState& kv_state,
      size_t num_tokens) override;
};

}  // namespace xllm
