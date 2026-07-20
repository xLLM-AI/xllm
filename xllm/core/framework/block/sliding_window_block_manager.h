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

#include "block_manager_impl.h"

namespace xllm {

// Sliding-window leaf of CompositeBlockManager.
//
// The low-level physical block pool (free list, allocate / deallocate / free,
// num_* accounting, id-0 padding) is identical to BlockManagerImpl, so this
// derives from it and reuses that machinery unchanged. Growth is the base flat
// append (allocate_for_sequence inherited). The sliding-window specifics are:
//   - release_out_of_window() drops leading blocks that have slid out of the
//     window; the composite calls it AFTER committing a successful round, so a
//     failed allocate_sequence never releases the sequence's existing blocks;
//   - the SWA block list grows by logical position and never shrinks in size;
//     released leading positions become invalid placeholders;
//   - the pool is sized with slack (see build_composite_leaves) so the peak
//     "old blocks not yet released + new tail" fits without borrowing capacity;
//   - no prefix cache.
class SlidingWindowBlockManager : public BlockManagerImpl {
 public:
  explicit SlidingWindowBlockManager(const Options& options);
  ~SlidingWindowBlockManager() override = default;

  // Release the leading SWA blocks of the sequence that have fully slid out of
  // the window: move them out (leaving invalid placeholders so logical-position
  // indexing stays stable) and return their ids to the pool. Called by the
  // composite after a successful allocate commit. Growth reuses the inherited
  // BlockManagerImpl::allocate_for_sequence (flat append).
  void release_out_of_window(Sequence* seq) override;

  // SWA never serves prefix cache; these must not be reached.
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

  uint32_t swa_blocks_per_seq() const { return options_.swa_blocks_per_seq(); }
};

}  // namespace xllm
