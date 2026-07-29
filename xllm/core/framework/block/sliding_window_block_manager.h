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

// Sliding-window leaf of CompositeBlockManager. Reuses BlockManagerImpl's
// physical pool and flat-append growth. SWA-specific behavior:
//   - release_out_of_window() drops leading slid-out blocks; released
//     positions stay as invalid placeholders so DSA modulo indexing
//     (`(pos/block_size) % semantic_cols`) remains stable.
//   - Prefix cache is gap-tolerant (LinearStatePrefixCache backend); the
//     tail-continuity check lives on the composite. TEXT hasher only.
class SlidingWindowBlockManager : public BlockManagerImpl {
 public:
  explicit SlidingWindowBlockManager(const Options& options);
  ~SlidingWindowBlockManager() override = default;

  // Deallocate leading blocks that have slid out of the window; leaves
  // invalid placeholders in their slots. Called by the composite after a
  // successful allocate commit.
  void release_out_of_window(Sequence* seq) override;

  // Gap-tolerant SWA probe. Delegates to LinearStatePrefixCache::match; see
  // that class for the shape returned. Composite owns the tail-continuity
  // check and the min-across-leaves clamp.
  std::vector<Block> allocate_shared(
      const Slice<int32_t>& token_ids,
      const Slice<Block>& existed_shared_blocks = {},
      const MMData& mm_data = MMData(),
      const Slice<XXH3Key>& block_hashes = {}) override;

  uint32_t swa_blocks_per_seq() const { return options_.swa_blocks_per_seq(); }
};

}  // namespace xllm
