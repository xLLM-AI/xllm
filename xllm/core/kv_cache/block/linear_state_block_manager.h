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

#include <cstdint>
#include <optional>
#include <vector>

#include "core/kv_cache/block/block_manager_impl.h"
#include "util/hash_util.h"

namespace xllm {

class Sequence;

// CompositeBlockManager leaf for Qwen3.5 GDN linear-state checkpoints.
// Registered under BlockType::LINEAR, enable_prefix_cache=true.
//
// Inherits BlockManagerImpl for the slot id pool and prefix cache. The
// inherited prefix_cache_ serves as the checkpoint index (LRU + eviction that
// skips is_shared() slots). Hash domain is INDEPENDENT of the KV leaf — keys
// are chained per prefill chunk (Sequence::update_linear_state_hashes), not
// from KV block hashes.
//
// The id pool is unified: slot ids [1, num_slots) serve LIVE slots (held by a
// sequence under composite_blocks_[LINEAR]) AND committed CHECKPOINT slots
// (owned by prefix_cache_) interchangeably under reference counting. Slot 0 is
// the inherited padding block.
class LinearStateBlockManager final : public BlockManagerImpl {
 public:
  explicit LinearStateBlockManager(uint32_t num_slots,
                                   int32_t chunk_stride,
                                   bool enable_prefix_cache = true,
                                   bool instance_is_decode = false);
  ~LinearStateBlockManager() override = default;

  std::optional<std::vector<Block>> allocate_for_sequence(
      Sequence* seq,
      size_t num_tokens) override;

  using BlockManagerImpl::allocate;
  Block allocate() override;

  std::vector<Block> allocate_shared(
      const Slice<int32_t>& token_ids,
      const Slice<Block>& existed_shared_blocks = {},
      const MMData& mm_data = MMData(),
      const Slice<XXH3Key>& block_hashes = {}) override;
  void cache(const Slice<int32_t>& token_ids,
             std::vector<Block>& blocks,
             size_t existed_shared_blocks_num = 0,
             const MMData& mm_data = MMData(),
             const Slice<XXH3Key>& block_hashes = {}) override;
  // The pre-hashed cache(const std::vector<Block>&) primitive is reused
  // verbatim for checkpoint insertion, so keep the base version instead of a
  // bespoke one.
  using BlockManagerImpl::cache;

 private:
  // Grants the test peer access to the inherited prefix_cache_ so unit tests
  // can assert by-hash checkpoint presence/lookup directly, mirroring how KV
  // unit tests assert against PrefixCache::find / contains. Production never
  // probes by hash here -- the composite mounts restore sources through
  // allocate_shared -- so no by-hash probe method is exposed on this leaf.
  friend class BlockManagerPoolTestPeer;
};

}  // namespace xllm
