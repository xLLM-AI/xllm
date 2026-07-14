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

#include <string>

#include "block_manager_impl.h"

namespace xllm {

// Per-sequence single-resource leaf (BlockType::SINGLE): linear-state /
// embedding row id, exactly one block per sequence, reused for its lifetime.
//
// The physical id pool (free list, allocate / deallocate / free, num_*
// accounting, id-0 padding) is identical to BlockManagerImpl, so this derives
// from it and reuses that machinery. block_size is 1 (each id is one block) and
// there is no prefix cache. What differs is the sequence-level policy: grow
// allocates one block only when the sequence does not already hold one.
class SingleBlockManager final : public BlockManagerImpl {
 public:
  SingleBlockManager(uint32_t num_blocks,
                     std::string resource_name,
                     std::string exhaustion_message = "");
  ~SingleBlockManager() override = default;

  // One block per sequence: empty when already held, else a single block (or
  // std::nullopt when the pool is exhausted). Does not insert into the sequence
  // -- the composite commits the returned block under BlockType::SINGLE.
  std::optional<std::vector<Block>> allocate_for_sequence(
      Sequence* seq,
      size_t num_tokens) override;

  // Single-block allocate: reuses the base free list but reports exhaustion
  // with this resource's name / message. Only this leaf calls it directly (the
  // base ctor's padding reservation runs before the vtable points here, so it
  // still uses BlockManagerImpl::allocate()). Bring the base allocate(size_t)
  // overload back into scope: declaring the zero-arg allocate() here would
  // otherwise hide it (name hiding).
  using BlockManagerImpl::allocate;
  Block allocate() override;

  // SINGLE never serves prefix cache; these must not be reached.
  std::vector<Block> allocate_shared(
      const Slice<int32_t>& token_ids,
      const Slice<Block>& existed_shared_blocks = {},
      const MMData& mm_data = MMData(),
      const Slice<XXH3Key>& block_hashes = {},
      size_t* matched_tokens = nullptr) override;
  void cache(const Slice<int32_t>& token_ids,
             std::vector<Block>& blocks,
             size_t existed_shared_blocks_num = 0,
             const MMData& mm_data = MMData(),
             const Slice<XXH3Key>& block_hashes = {}) override;
  void cache(const std::vector<Block>& blocks) override;

 private:
  std::string resource_name_;
  std::string exhaustion_message_;
};

}  // namespace xllm
