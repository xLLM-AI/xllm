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

#include <cstdint>
#include <vector>

#include "core/framework/multimodal/mm_data.h"
#include "framework/block/block.h"
#include "prefix_cache.h"
#include "util/hash_util.h"
#include "util/slice.h"

namespace xllm {

// Gap-tolerant variant of PrefixCache. Used by SWA (per base block; middle
// misses tolerated because attention only reads the tail) and LINEAR (per
// chunk; deeper hit subsumes earlier ones). Same LRU / insert / evict
// machinery as the base; only match() differs.
class LinearStatePrefixCache final : public PrefixCache {
 public:
  LinearStatePrefixCache(uint32_t block_size, BlockHasherType hasher_type)
      : PrefixCache(block_size, hasher_type) {}

  // Walks the chain per block position: hit -> emplace + LRU-promote; miss
  // -> emplace invalid placeholder. Trims trailing misses. Reach in tokens
  // is `returned.size() * block_size_`.
  std::vector<Block> match(const Slice<int32_t>& token_ids,
                           const Slice<Block>& existed_shared_blocks = {},
                           const MMData& mm_data = MMData(),
                           const Slice<XXH3Key>& block_hashes = {}) override;

  // Gap-tolerant insert for SWA slid-out placeholders. Unlike the base, the
  // block vector may contain invalid placeholders (released window blocks), so
  // the compute path walks the chain from token 0 rather than seeding from
  // blocks[existed_shared_blocks_num - 1] (which may itself be a placeholder
  // with an undefined stamped hash). Placeholders advance the chain cursor but
  // are not stamped or emplaced.
  size_t insert(const Slice<int32_t>& token_ids,
                std::vector<Block>& blocks,
                size_t existed_shared_blocks_num = 0,
                const MMData& mm_data = MMData(),
                const Slice<XXH3Key>& block_hashes = {}) override;
};

}  // namespace xllm
