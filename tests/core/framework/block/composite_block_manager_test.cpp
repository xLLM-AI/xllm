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

#include "composite_block_manager.h"

#include <gtest/gtest.h>

#include <set>

#include "framework/block/block_utils.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "framework/request/stopping_checker.h"
#include "framework/sampling/sampling_params.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {

namespace {

constexpr uint32_t kManagerTypeBlockManagerImpl = 0;
constexpr uint32_t kManagerTypeSlidingWindowBlockManager = 1;
constexpr uint32_t kMaxTokensPerBatch = 1280;

// Base block_size = 128. Two BlockManagerImpl: compress_ratio 4 and 128.
// - Ratio 4: block_size = 128*4 = 512, num_blocks = base_num_blocks/4.
// - Ratio 128: block_size = 128*128 = 16384, num_blocks = base_num_blocks/128.
// Use base_num_blocks = 128*32 = 4096 so that ratio-4 has 1024 blocks,
// ratio-128 has 32 blocks (ratio-4 block count is 32x ratio-128).
BlockManager::Options MakeCompositeOptions(uint32_t base_num_blocks,
                                           uint32_t block_size,
                                           uint32_t window_size,
                                           uint32_t max_seqs_per_batch) {
  const uint32_t swa_blocks_per_seq =
      static_cast<uint32_t>(get_swa_blocks_per_seq(window_size, block_size));
  BlockManager::Options opts;
  opts.num_blocks(base_num_blocks)
      .block_size(block_size)
      .sliding_window_size(window_size)
      .swa_blocks_per_seq(swa_blocks_per_seq)
      .max_tokens_per_batch(kMaxTokensPerBatch)
      .max_seqs_per_batch(max_seqs_per_batch)
      .manager_types({kManagerTypeSlidingWindowBlockManager,
                      kManagerTypeBlockManagerImpl,
                      kManagerTypeBlockManagerImpl})
      .compress_ratios({0, 4, 128});
  return opts;
}

constexpr uint32_t kBaseBlockSize = 128;
constexpr uint32_t kCompressRatio4 = 4;
constexpr uint32_t kCompressRatio128 = 128;
// Sub-manager 1 (ratio 4): block_size = 128*4 = 512.
constexpr uint32_t kBlockSizeRatio4 = kBaseBlockSize * kCompressRatio4;
// Sub-manager 2 (ratio 128): block_size = 128*128 = 16384.
constexpr uint32_t kBlockSizeRatio128 = kBaseBlockSize * kCompressRatio128;

inline size_t CeilBlocks(size_t num_tokens, size_t block_size) {
  return (num_tokens + block_size - 1) / block_size;
}

inline size_t ExpectedSwaLogicalBlocks(size_t num_tokens) {
  return CeilBlocks(num_tokens, kBaseBlockSize);
}

// Creates a minimal Sequence for testing (same pattern as batch_test.cpp).
Sequence MakeTestSequence(size_t index,
                          const std::vector<int32_t>& prompt_token_ids) {
  torch::Device device(Platform::type_torch(), 0);
  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(256);
  SequenceParams seq_params;
  // Large enough to hold DSV4-scale prompts (>= a full C128 block = 16384
  // tokens). Individual tests can still use short prompts; sequence capacity
  // just has to bound `num_tokens + max_generated_tokens`.
  seq_params.seq_capacity = 65536;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = false;
  seq_params.enable_schedule_overlap = false;

  torch::Tensor input_embedding;
  MMData mm_data;
  IncrementalDecoder decoder("", 1, false, false);
  return Sequence(index,
                  prompt_token_ids,
                  input_embedding,
                  mm_data,
                  std::move(decoder),
                  seq_params);
}

// Blocks are keyed by BlockType in the sequence KVCacheState, not by
// sub-manager index. These helpers read the three DSV4 groups by type so the
// tests stay independent of compress_ratios ordering.
std::vector<Block> SwaBlocks(Sequence& seq) {
  const Slice<Block> s = seq.kv_state().blocks(BlockType::SWA);
  return std::vector<Block>(s.begin(), s.end());
}
std::vector<Block> C4Blocks(Sequence& seq) {
  const Slice<Block> s = seq.kv_state().blocks(BlockType::C4);
  return std::vector<Block>(s.begin(), s.end());
}
std::vector<Block> C128Blocks(Sequence& seq) {
  const Slice<Block> s = seq.kv_state().blocks(BlockType::C128);
  return std::vector<Block>(s.begin(), s.end());
}

}  // namespace

TEST(CompositeBlockManagerTest, AllocateForSequence_SingleSeq) {
  const uint32_t base_block_size = kBaseBlockSize;
  const uint32_t base_num_blocks = 4096;  // ratio-4: 1024 blocks, ratio-128: 32
  const uint32_t window_size = 128;
  const uint32_t max_seqs_per_batch = 4;

  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, base_block_size, window_size, max_seqs_per_batch);

  CompositeBlockManager manager(build_composite_leaves(opts));
  EXPECT_TRUE(manager.is_composite());
  EXPECT_EQ(manager.num_sub_managers(), 3u);

  Sequence seq = MakeTestSequence(0, std::vector<int32_t>(1024, 1));
  const size_t num_tokens = 1024;

  EXPECT_TRUE(manager.allocate_sequence(&seq, num_tokens));

  const std::vector<Block> swa = SwaBlocks(seq);
  const std::vector<Block> c4 = C4Blocks(seq);
  const std::vector<Block> c128 = C128Blocks(seq);

  // SlidingWindow group: logical block count follows sequence length.
  EXPECT_EQ(swa.size(), ExpectedSwaLogicalBlocks(num_tokens));
  for (const auto& b : swa) {
    EXPECT_TRUE(b.is_valid());
    EXPECT_EQ(b.size(), base_block_size);
  }

  // BlockManagerImpl compress_ratio 4, block_size=128*4=512.
  const size_t expected_blocks_1 = CeilBlocks(num_tokens, kBlockSizeRatio4);
  EXPECT_EQ(c4.size(), expected_blocks_1);
  for (const auto& b : c4) {
    EXPECT_TRUE(b.is_valid());
    EXPECT_EQ(b.size(), kBlockSizeRatio4);
  }

  // BlockManagerImpl compress_ratio 128, block_size=128*128=16384.
  const size_t expected_blocks_2 = CeilBlocks(num_tokens, kBlockSizeRatio128);
  EXPECT_EQ(c128.size(), expected_blocks_2);
  for (const auto& b : c128) {
    EXPECT_TRUE(b.is_valid());
    EXPECT_EQ(b.size(), kBlockSizeRatio128);
  }

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest, AllocateForSequence_DifferentBatchSeqs) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = 128;
  const uint32_t max_seqs_per_batch = 4;
  const uint32_t sliding_window_blocks_per_sequence = static_cast<uint32_t>(
      get_swa_blocks_per_seq(window_size, kBaseBlockSize));

  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  CompositeBlockManager manager(build_composite_leaves(opts));

  // Seq1: 1024 tokens. Ratio 4: ceil(1024/512)=2; ratio 128:
  // ceil(1024/16384)=1.
  Sequence seq1 = MakeTestSequence(0, std::vector<int32_t>(1024, 1));
  EXPECT_TRUE(manager.allocate_sequence(&seq1, 1024));
  const std::vector<Block> s1_swa = SwaBlocks(seq1);
  const std::vector<Block> s1_c4 = C4Blocks(seq1);
  const std::vector<Block> s1_c128 = C128Blocks(seq1);
  EXPECT_EQ(s1_swa.size(), ExpectedSwaLogicalBlocks(1024));
  EXPECT_EQ(s1_c4.size(), CeilBlocks(1024, kBlockSizeRatio4));
  EXPECT_EQ(s1_c128.size(), CeilBlocks(1024, kBlockSizeRatio128));

  // Seq2: 1400 tokens. Ratio 4: ceil(1400/512)=3; ratio 128:
  // ceil(1400/16384)=1. Keep total SWA logical blocks within the dynamic
  // pool budget derived from max_tokens_per_batch.
  Sequence seq2 = MakeTestSequence(1, std::vector<int32_t>(1400, 1));
  EXPECT_TRUE(manager.allocate_sequence(&seq2, 1400));
  const std::vector<Block> s2_swa = SwaBlocks(seq2);
  const std::vector<Block> s2_c4 = C4Blocks(seq2);
  const std::vector<Block> s2_c128 = C128Blocks(seq2);
  EXPECT_EQ(s2_swa.size(), ExpectedSwaLogicalBlocks(1400));
  EXPECT_EQ(s2_c4.size(), CeilBlocks(1400, kBlockSizeRatio4));
  EXPECT_EQ(s2_c128.size(), CeilBlocks(1400, kBlockSizeRatio128));

  // Blocks allocated to different seqs must not overlap (distinct block ids).
  std::set<int32_t> ids1_0, ids1_1, ids1_2, ids2_0, ids2_1, ids2_2;
  for (const auto& b : s1_swa) ids1_0.insert(b.id());
  for (const auto& b : s1_c4) ids1_1.insert(b.id());
  for (const auto& b : s1_c128) ids1_2.insert(b.id());
  for (const auto& b : s2_swa) ids2_0.insert(b.id());
  for (const auto& b : s2_c4) ids2_1.insert(b.id());
  for (const auto& b : s2_c128) ids2_2.insert(b.id());
  for (int32_t id : ids1_0) EXPECT_EQ(ids2_0.count(id), 0u);
  for (int32_t id : ids1_1) EXPECT_EQ(ids2_1.count(id), 0u);
  for (int32_t id : ids1_2) EXPECT_EQ(ids2_2.count(id), 0u);
  const int32_t max_swa_block_id =
      sliding_window_blocks_per_sequence * max_seqs_per_batch +
      CeilBlocks(kMaxTokensPerBatch, kBaseBlockSize) + max_seqs_per_batch + 1;
  for (int32_t id : ids1_0) EXPECT_LE(id, max_swa_block_id);
  for (int32_t id : ids2_0) EXPECT_LE(id, max_swa_block_id);

  manager.deallocate_for_sequence(&seq1);
  manager.deallocate_for_sequence(&seq2);
}

TEST(CompositeBlockManagerTest, AllocateForSequence_GrowSameSeq) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = 128;
  const uint32_t max_seqs_per_batch = 4;

  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  CompositeBlockManager manager(build_composite_leaves(opts));

  Sequence seq = MakeTestSequence(0, {1, 2, 3});
  // 600 tokens: ratio 4 needs ceil(600/512)=2 blocks, ratio 128 needs 1 block.
  EXPECT_TRUE(manager.allocate_sequence(&seq, 600));
  EXPECT_EQ(SwaBlocks(seq).size(), ExpectedSwaLogicalBlocks(600));
  EXPECT_EQ(C4Blocks(seq).size(), CeilBlocks(600, kBlockSizeRatio4));
  EXPECT_EQ(C128Blocks(seq).size(), CeilBlocks(600, kBlockSizeRatio128));

  // Grow to 1200 tokens: ratio 4 needs 3 blocks, ratio 128 still 1 block.
  EXPECT_TRUE(manager.allocate_sequence(&seq, 1200));
  EXPECT_EQ(SwaBlocks(seq).size(), ExpectedSwaLogicalBlocks(1200));
  EXPECT_EQ(C4Blocks(seq).size(), CeilBlocks(1200, kBlockSizeRatio4));
  EXPECT_EQ(C128Blocks(seq).size(), CeilBlocks(1200, kBlockSizeRatio128));

  // No growth: still 1200 tokens, block counts unchanged.
  EXPECT_TRUE(manager.allocate_sequence(&seq, 1200));
  EXPECT_EQ(C4Blocks(seq).size(), CeilBlocks(1200, kBlockSizeRatio4));
  EXPECT_EQ(C128Blocks(seq).size(), CeilBlocks(1200, kBlockSizeRatio128));

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest, AllocateContinuesAfterSatisfiedTokenManager) {
  BlockManager::Options opts =
      MakeCompositeOptions(4096, kBaseBlockSize, 128, 4);
  opts.compress_ratios({0, 128, 4});
  CompositeBlockManager manager(build_composite_leaves(opts));

  Sequence seq = MakeTestSequence(0, {1});
  EXPECT_TRUE(manager.allocate_sequence(&seq, 1024));
  EXPECT_EQ(C128Blocks(seq).size(), CeilBlocks(1024, kBlockSizeRatio128));
  EXPECT_EQ(C4Blocks(seq).size(), CeilBlocks(1024, kBlockSizeRatio4));

  EXPECT_TRUE(manager.allocate_sequence(&seq, 1500));
  EXPECT_EQ(C128Blocks(seq).size(), CeilBlocks(1500, kBlockSizeRatio128));
  EXPECT_EQ(C4Blocks(seq).size(), CeilBlocks(1500, kBlockSizeRatio4));

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest, AllocateForSequence_NullSeqReturnsFalse) {
  BlockManager::Options opts =
      MakeCompositeOptions(4096, kBaseBlockSize, 128, 4);
  CompositeBlockManager manager(build_composite_leaves(opts));
  EXPECT_FALSE(manager.allocate_sequence(nullptr, 10));
}

TEST(CompositeBlockManagerTest, FailedGrowthRollsBackNewBlocks) {
  BlockManager::Options opts = MakeCompositeOptions(/*base_num_blocks=*/256,
                                                    kBaseBlockSize,
                                                    /*window_size=*/12,
                                                    /*max_seqs_per_batch=*/4);
  CompositeBlockManager manager(build_composite_leaves(opts));

  Sequence seq = MakeTestSequence(0, {1});
  ASSERT_TRUE(manager.allocate_sequence(&seq, 1024));
  const size_t used_before = manager.num_used_blocks();
  const std::vector<Block> before_swa = SwaBlocks(seq);
  const size_t swa_blocks_before = before_swa.size();
  const size_t c4_blocks_before = C4Blocks(seq).size();
  const size_t c128_blocks_before = C128Blocks(seq).size();
  std::vector<int32_t> swa_ids_before;
  swa_ids_before.reserve(before_swa.size());
  for (const auto& block : before_swa) {
    ASSERT_TRUE(block.is_valid());
    swa_ids_before.push_back(block.id());
  }
  seq.kv_state().incr_kv_cache_tokens_num(1024);

  EXPECT_FALSE(manager.allocate_sequence(&seq, 4096));
  EXPECT_EQ(manager.num_used_blocks(), used_before);

  const std::vector<Block> after_swa = SwaBlocks(seq);
  EXPECT_EQ(after_swa.size(), swa_blocks_before);
  EXPECT_EQ(C4Blocks(seq).size(), c4_blocks_before);
  EXPECT_EQ(C128Blocks(seq).size(), c128_blocks_before);
  ASSERT_EQ(after_swa.size(), swa_ids_before.size());
  for (size_t i = 0; i < after_swa.size(); ++i) {
    EXPECT_TRUE(after_swa[i].is_valid());
    EXPECT_EQ(after_swa[i].id(), swa_ids_before[i]);
  }

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest, DeallocateToleratesRolledBackEmptySequence) {
  BlockManager::Options opts = MakeCompositeOptions(/*base_num_blocks=*/128,
                                                    kBaseBlockSize,
                                                    /*window_size=*/12,
                                                    /*max_seqs_per_batch=*/4);
  CompositeBlockManager manager(build_composite_leaves(opts));

  Sequence seq = MakeTestSequence(0, {1});

  EXPECT_FALSE(manager.allocate_sequence(&seq, 4096));
  EXPECT_NO_FATAL_FAILURE(manager.deallocate_for_sequence(&seq));
  EXPECT_FALSE(seq.kv_state().has_multi_block_export());
  EXPECT_EQ(manager.num_used_blocks(), 0u);
}

// Verifies that when seq token count increases, CompositeBlockManager correctly
// adds blocks (only appends new blocks; existing block ids are preserved).
TEST(CompositeBlockManagerTest, TokenIncrease_AddsBlocksIncrementally) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = 128;
  const uint32_t max_seqs_per_batch = 4;

  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  CompositeBlockManager manager(build_composite_leaves(opts));

  Sequence seq = MakeTestSequence(0, {1});
  std::vector<size_t> token_steps = {100, 600, 1200, 2000, 2400};

  std::vector<std::set<int32_t>> prev_ids_1;
  std::vector<std::set<int32_t>> prev_ids_2;

  for (size_t num_tokens : token_steps) {
    EXPECT_TRUE(manager.allocate_sequence(&seq, num_tokens));

    const std::vector<Block> c4 = C4Blocks(seq);
    const std::vector<Block> c128 = C128Blocks(seq);

    const size_t expect_1 = CeilBlocks(num_tokens, kBlockSizeRatio4);
    const size_t expect_2 = CeilBlocks(num_tokens, kBlockSizeRatio128);
    EXPECT_EQ(c4.size(), expect_1)
        << "num_tokens=" << num_tokens << " C4 block count";
    EXPECT_EQ(c128.size(), expect_2)
        << "num_tokens=" << num_tokens << " C128 block count";

    // Check that previously allocated block ids are still present (only
    // append).
    std::set<int32_t> ids_1, ids_2;
    for (const auto& b : c4) ids_1.insert(b.id());
    for (const auto& b : c128) ids_2.insert(b.id());
    for (const auto& prev : prev_ids_1) {
      for (int32_t id : prev) EXPECT_GT(ids_1.count(id), 0u);
    }
    for (const auto& prev : prev_ids_2) {
      for (int32_t id : prev) EXPECT_GT(ids_2.count(id), 0u);
    }
    prev_ids_1.push_back(ids_1);
    prev_ids_2.push_back(ids_2);
  }

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest, SlidingWindowReleasesSkippedPhysicalBlocks) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t sliding_window_blocks_per_sequence = 3;
  const uint32_t window_size =
      sliding_window_blocks_per_sequence * kBaseBlockSize;
  const uint32_t max_seqs_per_batch = 4;

  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  CompositeBlockManager manager(build_composite_leaves(opts));

  Sequence seq = MakeTestSequence(0, {1});
  const size_t window_tokens =
      static_cast<size_t>(sliding_window_blocks_per_sequence) * kBaseBlockSize;

  EXPECT_TRUE(manager.allocate_sequence(&seq, window_tokens));
  const std::vector<Block> initial = SwaBlocks(seq);
  ASSERT_EQ(initial.size(), sliding_window_blocks_per_sequence);
  seq.kv_state().incr_kv_cache_tokens_num(window_tokens);

  std::vector<int32_t> initial_ids;
  initial_ids.reserve(initial.size());
  for (const auto& block : initial) {
    initial_ids.push_back(block.id());
  }

  EXPECT_TRUE(
      manager.allocate_sequence(&seq, window_tokens + 2 * kBaseBlockSize));
  const std::vector<Block> boundary = SwaBlocks(seq);
  ASSERT_EQ(boundary.size(), sliding_window_blocks_per_sequence + 2);
  for (size_t i = 0; i < initial_ids.size(); ++i) {
    EXPECT_EQ(boundary[i].id(), initial_ids[i]);
  }
  seq.kv_state().incr_kv_cache_tokens_num(2 * kBaseBlockSize);

  EXPECT_TRUE(
      manager.allocate_sequence(&seq, window_tokens + 3 * kBaseBlockSize));
  const std::vector<Block> exceeded = SwaBlocks(seq);
  ASSERT_EQ(exceeded.size(), sliding_window_blocks_per_sequence + 3);
  EXPECT_FALSE(exceeded[0].is_valid());
  EXPECT_FALSE(exceeded[1].is_valid());
  EXPECT_EQ(exceeded[0].id(), -1);
  EXPECT_EQ(exceeded[1].id(), -1);
  for (size_t i = 2; i < initial_ids.size(); ++i) {
    EXPECT_EQ(exceeded[i].id(), initial_ids[i]);
  }
  for (size_t i = initial_ids.size(); i < exceeded.size(); ++i) {
    EXPECT_TRUE(exceeded[i].is_valid());
  }

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest, DeallocateSliceDispatchesToOwnerManagers) {
  BlockManager::Options opts =
      MakeCompositeOptions(4096, kBaseBlockSize, 128, 4);
  opts.enable_prefix_cache(false);
  CompositeBlockManager manager(build_composite_leaves(opts));

  Sequence seq = MakeTestSequence(0, {1});
  EXPECT_TRUE(manager.allocate_sequence(&seq, 1500));
  EXPECT_GT(manager.num_used_blocks(), 0u);

  std::vector<Block> flat_blocks;
  for (const BlockType type :
       {BlockType::SWA, BlockType::C4, BlockType::C128}) {
    const Slice<Block> manager_blocks = seq.kv_state().blocks(type);
    flat_blocks.insert(
        flat_blocks.end(), manager_blocks.begin(), manager_blocks.end());
  }

  manager.deallocate(flat_blocks);
  EXPECT_EQ(manager.num_used_blocks(), 0u);

  seq.reset();
}

TEST(CompositeBlockManagerTest,
     DeallocateSliceDispatchesWithoutInflatingRefCount) {
  BlockManager::Options opts =
      MakeCompositeOptions(4096, kBaseBlockSize, 128, 4);
  CompositeBlockManager manager(build_composite_leaves(opts));

  Sequence seq = MakeTestSequence(0, {1});
  EXPECT_TRUE(manager.allocate_sequence(&seq, 1500));
  EXPECT_GT(manager.num_used_blocks(), 0u);

  std::vector<const Block*> flat_blocks;
  for (const BlockType type :
       {BlockType::SWA, BlockType::C4, BlockType::C128}) {
    const Slice<Block> manager_blocks = seq.kv_state().blocks(type);
    for (const auto& block : manager_blocks) {
      flat_blocks.push_back(&block);
    }
  }

  for (const Block* block : flat_blocks) {
    ASSERT_NE(block, nullptr);
    EXPECT_EQ(block->ref_count(), 1u);
  }

  for (const Block* block : flat_blocks) {
    manager.deallocate(Slice<Block>(block, 1));
  }
  EXPECT_EQ(manager.num_used_blocks(), 0u);

  for (const Block* block : flat_blocks) {
    ASSERT_NE(block, nullptr);
    EXPECT_EQ(block->ref_count(), 1u);
  }

  seq.reset();
}

// Finding 3 regression: composite capacity stats must report a single admission
// leaf's raw block count (the smallest-block-size one = C4 here), NOT a min/sum
// mix across C4+C128. C128's raw count (32) must never define pool capacity,
// otherwise schedulers (which read num_free * block_size() as base tokens)
// badly under-estimate capacity.
TEST(CompositeBlockManagerTest, CapacityStatsUseFinestAdmissionLeaf) {
  // base_num_blocks=4096 -> C4: 4096/4=1024 blocks (bs=512);
  //                         C128: 4096/128=32 blocks (bs=16384).
  BlockManager::Options opts =
      MakeCompositeOptions(4096, kBaseBlockSize, 128, 4);
  CompositeBlockManager manager(build_composite_leaves(opts));

  // num_total_blocks must equal the C4 leaf's total (1024 - padding), i.e. far
  // larger than C128's 32. Assert it is well above the C128 count so a min/sum
  // regression (which would yield ~32 or 1024+32) is caught.
  const size_t total = manager.num_total_blocks();
  EXPECT_GT(total, 900u);   // C4 ~1023, not C128's ~31
  EXPECT_LT(total, 1100u);  // not C4+C128 sum territory either

  // Free (no sequence yet) equals total; used is 0.
  EXPECT_EQ(manager.num_free_blocks(), total);
  EXPECT_EQ(manager.num_used_blocks(), 0u);

  // After allocating one sequence, used reflects ONLY the C4 leaf (capacity
  // leaf), not a C4+C128 sum.
  Sequence seq = MakeTestSequence(0, std::vector<int32_t>(1024, 1));
  ASSERT_TRUE(manager.allocate_sequence(&seq, 1024));
  const size_t c4_used = C4Blocks(seq).size();  // capacity leaf's used count
  EXPECT_EQ(manager.num_used_blocks(), c4_used);
  EXPECT_EQ(manager.num_free_blocks(), total - c4_used);

  manager.deallocate_for_sequence(&seq);
}

// DSV4 prefix cache: a fresh sequence with the same prompt as a previously
// released sequence should mount shared blocks from all three prefix-cache
// leaves (SWA / C4 / C128). The composite takes the cross-leaf min hit length
// clamped to a C128 block, so the prompt has to span at least one C128 block
// (128 * base = 16384 tokens) for the hit to be non-trivial. Uses a 2*C128
// prompt so we get a meaningful hit even after the exact-repeat pop.
TEST(CompositeBlockManagerTest, Dsv4PrefixCacheHitOnRepeatedPrefix) {
  const uint32_t base_num_blocks = 4096;
  // Sliding window covers a couple C128 blocks worth of tokens so the SWA
  // tail-continuity check has room to succeed even after the exact-repeat pop.
  const uint32_t window_size = 4 * kBaseBlockSize;
  const uint32_t max_seqs_per_batch = 4;
  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  // max_tokens_per_batch has to accommodate the prompt so allocate_sequence
  // does not exceed the SWA burst budget.
  opts.max_tokens_per_batch(3 * kBlockSizeRatio128);
  ASSERT_TRUE(opts.enable_prefix_cache());
  CompositeBlockManager manager(build_composite_leaves(opts));

  // First sequence: a 2*C128 prompt (32768 tokens) so both C4 and C128 have
  // multiple full blocks worth of cacheable prefix. Mark all tokens as
  // forwarded so the pre-grow hook (fired during the NEXT allocate) has data
  // to insert -- and we drive one extra allocate to trigger it explicitly.
  const size_t num_tokens = 2 * kBlockSizeRatio128;
  const std::vector<int32_t> prompt(num_tokens, 7);
  Sequence seq_first = MakeTestSequence(0, prompt);
  ASSERT_TRUE(manager.allocate_sequence(&seq_first, num_tokens));
  seq_first.kv_state().incr_kv_cache_tokens_num(num_tokens);
  // deallocate_for_sequence runs the pre-grow hook once more (via
  // cache_for_sequence's fan-out), flushing the residual full blocks.
  manager.deallocate_for_sequence(&seq_first);
  seq_first.reset();

  // The prefix cache under SWA / C4 / C128 should now hold the prompt's full
  // blocks. On a second sequence with the same prompt, admission mounts from
  // all three leaves and pins kv_cache_tokens_num to the safe min hit.
  Sequence seq_hit = MakeTestSequence(1, prompt);
  manager.allocate_shared_for_sequence(&seq_hit);
  EXPECT_GT(seq_hit.kv_state().shared_blocks_num(BlockType::SWA), 0u);
  EXPECT_GT(seq_hit.kv_state().shared_blocks_num(BlockType::C4), 0u);
  EXPECT_GT(seq_hit.kv_state().shared_blocks_num(BlockType::C128), 0u);
  const size_t kv_tokens = seq_hit.kv_state().kv_cache_tokens_num();
  EXPECT_GT(kv_tokens, 0u);
  EXPECT_LT(kv_tokens, num_tokens);  // exact-repeat pop kept at least one c128
  EXPECT_EQ(kv_tokens % kBlockSizeRatio128, 0u)
      << "safe_hit_tokens must be a multiple of the C128 block stride";
  // SWA vector length matches safe_hit / base and the tail carries valid
  // blocks.
  const std::vector<Block> hit_swa = SwaBlocks(seq_hit);
  EXPECT_EQ(hit_swa.size(), kv_tokens / kBaseBlockSize);
  EXPECT_TRUE(hit_swa.back().is_valid());

  manager.deallocate_for_sequence(&seq_hit);
}

// DSV4 prefix cache: a fresh sequence with a completely different prompt should
// miss cleanly (no shared mount, kv_cache_tokens_num unchanged).
TEST(CompositeBlockManagerTest, Dsv4PrefixCacheMissCleanly) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = 4 * kBaseBlockSize;
  const uint32_t max_seqs_per_batch = 4;
  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  opts.max_tokens_per_batch(3 * kBlockSizeRatio128);
  CompositeBlockManager manager(build_composite_leaves(opts));

  const size_t num_tokens = 2 * kBlockSizeRatio128;
  const std::vector<int32_t> prompt_a(num_tokens, 7);
  Sequence seq_a = MakeTestSequence(0, prompt_a);
  ASSERT_TRUE(manager.allocate_sequence(&seq_a, num_tokens));
  seq_a.kv_state().incr_kv_cache_tokens_num(num_tokens);
  manager.deallocate_for_sequence(&seq_a);
  seq_a.reset();

  // A sequence with a totally different prompt should not share any block.
  std::vector<int32_t> prompt_b(num_tokens, 0);
  for (size_t i = 0; i < prompt_b.size(); ++i) {
    prompt_b[i] = static_cast<int32_t>(i + 100);
  }
  Sequence seq_b = MakeTestSequence(1, prompt_b);
  manager.allocate_shared_for_sequence(&seq_b);
  EXPECT_EQ(seq_b.kv_state().shared_blocks_num(BlockType::SWA), 0u);
  EXPECT_EQ(seq_b.kv_state().shared_blocks_num(BlockType::C4), 0u);
  EXPECT_EQ(seq_b.kv_state().shared_blocks_num(BlockType::C128), 0u);
  EXPECT_EQ(seq_b.kv_state().kv_cache_tokens_num(), 0u);

  manager.deallocate_for_sequence(&seq_b);
}

// SWA slid-out blocks should enter the prefix cache via the pre-grow hook (v2b:
// hook fires at every allocate_sequence, insertion is incremental and cached
// blocks survive slid-out release through the ref<=2u path). A follow-up
// sequence with the same prompt should hit those cached blocks.
TEST(CompositeBlockManagerTest, SlidingWindowSlidOutBlocksEnterPrefixCache) {
  const uint32_t base_num_blocks = 4096;
  // Window covers a small number of base blocks so slide-out happens quickly.
  const uint32_t sliding_window_blocks_per_sequence = 3;
  const uint32_t window_size =
      sliding_window_blocks_per_sequence * kBaseBlockSize;
  const uint32_t max_seqs_per_batch = 4;
  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  opts.max_tokens_per_batch(3 * kBlockSizeRatio128);
  ASSERT_TRUE(opts.enable_prefix_cache());
  CompositeBlockManager manager(build_composite_leaves(opts));

  // Prompt spans two full C128 blocks so the min-across-leaves gate can
  // survive AND the exact-repeat pop (one c128 stride) still leaves shared
  // blocks behind. A single-c128 prompt would end up entirely popped.
  const size_t num_tokens = 2 * kBlockSizeRatio128;
  const std::vector<int32_t> prompt(num_tokens, 7);
  Sequence seq_first = MakeTestSequence(0, prompt);
  ASSERT_TRUE(manager.allocate_sequence(&seq_first, num_tokens));
  seq_first.kv_state().incr_kv_cache_tokens_num(num_tokens);
  // The SWA leaf should have released everything but the last window-sized
  // slab back to the pool (as invalid placeholders), and those blocks landed
  // in the prefix cache via the pre-grow hook before release.
  manager.deallocate_for_sequence(&seq_first);
  seq_first.reset();

  // A second sequence with the same prompt should hit the cached SWA tail.
  Sequence seq_hit = MakeTestSequence(1, prompt);
  manager.allocate_shared_for_sequence(&seq_hit);
  EXPECT_GT(seq_hit.kv_state().shared_blocks_num(BlockType::SWA), 0u);
  EXPECT_GT(seq_hit.kv_state().shared_blocks_num(BlockType::C4), 0u);
  EXPECT_GT(seq_hit.kv_state().shared_blocks_num(BlockType::C128), 0u);

  manager.deallocate_for_sequence(&seq_hit);
}

// v2b: pre-grow hook advances KVCacheState::num_cached_blocks incrementally
// across chunked-prefill steps. First chunk: cursor at 0. Second chunk (after
// forwarding chunk 1's tokens): cursor at chunk-1's full-block count.
TEST(CompositeBlockManagerTest, Dsv4PrefixCachePreHookCursorAdvances) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = 4 * kBaseBlockSize;
  const uint32_t max_seqs_per_batch = 4;
  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  opts.max_tokens_per_batch(4 * kBlockSizeRatio128);
  CompositeBlockManager manager(build_composite_leaves(opts));

  // Two-chunk prompt (2*C128). Chunk 1 is a single C128 block wide.
  const size_t chunk = kBlockSizeRatio128;
  const size_t total = 2 * chunk;
  Sequence seq = MakeTestSequence(0, std::vector<int32_t>(total, 7));

  // Chunk 1: allocate + forward.
  ASSERT_TRUE(manager.allocate_sequence(&seq, chunk));
  EXPECT_EQ(seq.kv_state().num_cached_blocks(BlockType::SWA), 0u)
      << "pre-grow hook has nothing to cache on the very first allocate: kv=0";
  EXPECT_EQ(seq.kv_state().num_cached_blocks(BlockType::C4), 0u);
  EXPECT_EQ(seq.kv_state().num_cached_blocks(BlockType::C128), 0u);
  seq.kv_state().incr_kv_cache_tokens_num(chunk);

  // Chunk 2 allocation triggers the pre-grow hook, which now sees kv=chunk
  // worth of forwarded tokens and inserts them.
  ASSERT_TRUE(manager.allocate_sequence(&seq, total));
  EXPECT_EQ(seq.kv_state().num_cached_blocks(BlockType::SWA),
            chunk / kBaseBlockSize);
  EXPECT_EQ(seq.kv_state().num_cached_blocks(BlockType::C4),
            chunk / kBlockSizeRatio4);
  EXPECT_EQ(seq.kv_state().num_cached_blocks(BlockType::C128),
            chunk / kBlockSizeRatio128);

  manager.deallocate_for_sequence(&seq);
}

// v2b: exact-repeat safe-hit hits the pop path -- when a fresh sequence has a
// prompt whose entire length is cached, safe_hit_tokens is popped by one
// c128 block worth of tokens so the forward has data to process. The test
// verifies kv_cache_tokens_num after mount is strictly less than the prompt.
TEST(CompositeBlockManagerTest, Dsv4PrefixCacheExactRepeatPopsOneC128) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = 4 * kBaseBlockSize;
  const uint32_t max_seqs_per_batch = 4;
  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  opts.max_tokens_per_batch(4 * kBlockSizeRatio128);
  CompositeBlockManager manager(build_composite_leaves(opts));

  const size_t num_tokens = 3 * kBlockSizeRatio128;
  const std::vector<int32_t> prompt(num_tokens, 42);
  Sequence seq_first = MakeTestSequence(0, prompt);
  ASSERT_TRUE(manager.allocate_sequence(&seq_first, num_tokens));
  seq_first.kv_state().incr_kv_cache_tokens_num(num_tokens);
  manager.deallocate_for_sequence(&seq_first);
  seq_first.reset();

  Sequence seq_hit = MakeTestSequence(1, prompt);
  manager.allocate_shared_for_sequence(&seq_hit);
  // Full-length hit → pop one c128 block. Expect kv = num_tokens - c128.
  EXPECT_EQ(seq_hit.kv_state().kv_cache_tokens_num(),
            num_tokens - kBlockSizeRatio128);

  manager.deallocate_for_sequence(&seq_hit);
}

}  // namespace xllm
