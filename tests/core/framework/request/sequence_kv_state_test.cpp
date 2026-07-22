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

#include "framework/request/sequence_kv_state.h"

#include <gtest/gtest.h>

namespace xllm {

TEST(KVCacheStateTest, TransferCursorTracksAndResets) {
  KVCacheState state;
  EXPECT_EQ(state.next_transfer_block_idx(), 0u);

  state.set_next_transfer_block_idx(2);
  EXPECT_EQ(state.next_transfer_block_idx(), 2u);

  state.advance_transfer_block_idx(5);
  EXPECT_EQ(state.next_transfer_block_idx(), 5u);

  state.advance_transfer_block_idx(3);
  EXPECT_EQ(state.next_transfer_block_idx(), 5u);

  state.reset();
  EXPECT_EQ(state.next_transfer_block_idx(), 0u);
}

TEST(KVCacheStateTest, ResetClearsBeamSourceState) {
  KVCacheState state;
  Block block(/*id=*/1, /*allocator=*/nullptr);
  const std::vector<Block> src_blocks{block};
  const uint32_t external_ref_count = block.ref_count();

  state.set_src_blocks(src_blocks, /*need_swap=*/true);
  EXPECT_FALSE(state.src_blocks().empty());
  EXPECT_TRUE(state.need_swap());
  EXPECT_EQ(block.ref_count(), external_ref_count + 1);

  state.reset();
  EXPECT_TRUE(state.src_blocks().empty());
  EXPECT_FALSE(state.need_swap());
  EXPECT_EQ(block.ref_count(), external_ref_count);
}

// Finding 2 regression: has_any_blocks() must report true for ANY
// cache-bearing type (KV / SWA / C4 / C128) and IGNORE SINGLE. The pool's
// "started_empty" rollback decision relies on this so that a DSV4 sequence
// (which holds SWA/C4/C128 but no KV) is not treated as fresh on a failed grow.
TEST(KVCacheStateTest, HasAnyBlocksIgnoresSingle) {
  auto make_block = [](int32_t id) {
    return std::vector<Block>{Block(id, /*manager=*/nullptr)};
  };

  // Empty state.
  {
    KVCacheState state;
    EXPECT_FALSE(state.has_any_blocks());
  }

  // SINGLE-only must NOT count as cache.
  {
    KVCacheState state;
    state.add_blocks(BlockType::SINGLE, make_block(1));
    EXPECT_FALSE(state.has_any_blocks());
  }

  // Each cache-bearing type alone counts.
  for (const BlockType type :
       {BlockType::KV, BlockType::SWA, BlockType::C4, BlockType::C128}) {
    KVCacheState state;
    state.add_blocks(type, make_block(2));
    EXPECT_TRUE(state.has_any_blocks()) << "type=" << static_cast<int>(type);
  }

  // DSV4-like: SWA/C4/C128 present, no KV -> still true.
  {
    KVCacheState state;
    state.add_blocks(BlockType::SWA, make_block(3));
    state.add_blocks(BlockType::C4, make_block(4));
    state.add_blocks(BlockType::C128, make_block(5));
    state.add_blocks(BlockType::SINGLE, make_block(6));
    EXPECT_TRUE(state.has_any_blocks());
  }
}

}  // namespace xllm
