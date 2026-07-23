/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "block_manager_impl.h"
#include "block_manager_pool.h"
#include "common/global_flags.h"
#include "core/framework/config/scheduler_config.h"
#include "framework/block/block_manager_pool_test_peer.h"
#include "framework/block/linear_state_block_manager.h"
#include "framework/model/model_input_params.h"
#include "framework/prefix_cache/block_hasher.h"
#include "framework/request/incremental_decoder.h"

namespace xllm {

namespace {

template <typename T>
class ScopedValue final {
 public:
  ScopedValue(T* target, T value) : target_(target), old_(*target) {
    *target_ = value;
  }
  ~ScopedValue() { *target_ = old_; }

  ScopedValue(const ScopedValue&) = delete;
  ScopedValue& operator=(const ScopedValue&) = delete;

 private:
  T* target_;
  T old_;
};

LinearStatePrefixHash make_prefix_hash(uint8_t tag) {
  LinearStatePrefixHash hash{};
  hash.fill(tag);
  return hash;
}

LinearStatePrefixHash compute_linear_state_prefix_hash_for_test(
    const Slice<int32_t>& token_ids,
    int32_t block_size,
    size_t boundary_tokens) {
  LinearStatePrefixHash hash{};
  if (block_size <= 0 || boundary_tokens == 0) {
    return hash;
  }
  const size_t stride = static_cast<size_t>(block_size);
  if (boundary_tokens % stride != 0 || boundary_tokens > token_ids.size()) {
    return hash;
  }

  const size_t boundary_blocks = boundary_tokens / stride;
  const uint8_t* previous_hash = nullptr;
  for (size_t block_idx = 0; block_idx < boundary_blocks; ++block_idx) {
    xxh3_128bits_hash(
        previous_hash,
        token_ids.slice(block_idx * stride, (block_idx + 1) * stride),
        hash.data());
    previous_hash = hash.data();
  }
  return hash;
}

int32_t insert_linear_state_checkpoint(LinearStateBlockManager* cache,
                                       const XXH3Key& hash) {
  if (BlockManagerPoolTestPeer::contains(cache, hash)) {
    Block matched = BlockManagerPoolTestPeer::match(cache, hash);
    return matched.is_valid() ? matched.id() : -1;
  }
  Block slot_block = cache->allocate();
  if (!slot_block.is_valid()) {
    return -1;
  }
  const int32_t slot = slot_block.id();
  slot_block.set_hash_value(hash.data);
  std::vector<Block> checkpoint;
  checkpoint.emplace_back(std::move(slot_block));
  cache->cache(checkpoint);
  return slot;
}

int32_t insert_linear_state_checkpoint(LinearStateBlockManager* cache,
                                       const LinearStatePrefixHash& hash) {
  return insert_linear_state_checkpoint(cache, XXH3Key(hash.data()));
}

// Deferred linear-state saves are now executed by the LINEAR leaf at the top of
// allocate_for_sequence (the production path runs it once per scheduler step
// per sequence). Drive that same entry here to exercise the save. The leaf only
// RETURNS freshly acquired slots; CompositeBlockManager::allocate_sequence is
// what appends them to the sequence, so mirror that add_blocks step here.
void apply_pending_linear_saves_via_leaf(
    LinearStateBlockManager* leaf,
    const std::vector<Sequence*>& sequences) {
  for (Sequence* seq : sequences) {
    if (seq == nullptr) {
      continue;
    }
    std::optional<std::vector<Block>> blocks =
        leaf->allocate_for_sequence(seq, seq->num_tokens());
    if (blocks.has_value() && !blocks->empty()) {
      seq->kv_state().add_blocks(BlockType::LINEAR, *blocks);
    }
  }
}

Sequence make_sequence(size_t index,
                       const std::vector<int32_t>& prompt_tokens) {
  RequestSamplingParam sampling_param;
  sampling_param.beam_width = 0;
  sampling_param.is_embeddings = false;

  StoppingChecker stopping_checker;

  SequenceParams params;
  params.seq_capacity = prompt_tokens.size() + 8;
  params.echo = false;
  params.skip_special_tokens = true;
  params.streaming = false;
  params.enable_schedule_overlap = false;
  params.rec_type = RecType::kNone;
  params.bos_token_id = 0;
  params.request_id = "block_manager_pool_test";
  params.sampling_param = &sampling_param;
  params.stopping_checker = &stopping_checker;

  IncrementalDecoder decoder(
      /*prompt=*/"prompt",
      /*num_prompt_tokens=*/prompt_tokens.size(),
      /*echo=*/params.echo,
      /*skip_special_tokens=*/params.skip_special_tokens);

  return Sequence(index,
                  prompt_tokens,
                  /*input_embedding=*/torch::Tensor(),
                  /*mm_data=*/MMData(),
                  decoder,
                  params);
}

BlockManagerPool::Options make_linear_state_pool_options(
    int32_t linear_state_num_slots) {
  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  // Zero out max_seqs_per_batch so the SINGLE pool is sized purely by
  // num_single_blocks (otherwise the pool takes the max of the two).
  options.max_seqs_per_batch(0)
      .num_single_blocks(4)
      .enable_linear_state(true)
      .linear_state_num_slots(linear_state_num_slots);
  return options;
}

}  // namespace

TEST(BlockManagerTest, Basic) {
  const uint32_t n_blocks = 10;
  const uint32_t block_size = 2;
  BlockManager::Options options;
  options.num_blocks(n_blocks).block_size(block_size);
  BlockManagerImpl manager(options);

  EXPECT_EQ(manager.num_free_blocks(), n_blocks - 1);
  EXPECT_EQ(manager.block_size(), block_size);

  // Allocate a block
  {
    Block block = manager.allocate();
    EXPECT_EQ(block.id(), 1);
    EXPECT_EQ(block.size(), block_size);
    EXPECT_EQ(block.is_shared(), false);
    EXPECT_EQ(block.ref_count(), 1);

    EXPECT_EQ(manager.num_free_blocks(), n_blocks - 2);
  }
  // the block should be freed after the scope
  EXPECT_EQ(manager.num_free_blocks(), n_blocks - 1);

  // Allocate a list of blocks
  {
    std::vector<Block> blocks;
    for (uint32_t i = 1; i < n_blocks; ++i) {
      auto block = manager.allocate();
      EXPECT_EQ(block.id(), i);
      EXPECT_EQ(block.size(), block_size);
      EXPECT_EQ(block.is_shared(), false);
      EXPECT_EQ(block.ref_count(), 1);
      blocks.push_back(std::move(block));
    }
    EXPECT_EQ(manager.num_free_blocks(), 0);
    for (const auto& block : blocks) {
      EXPECT_EQ(block.ref_count(), 1);
      EXPECT_EQ(block.is_shared(), false);
    }

    // Test CHECK failure
    EXPECT_DEATH(manager.allocate(), "No more blocks available");
  }

  // all blocks should be freed after the scope
  EXPECT_EQ(manager.num_free_blocks(), n_blocks - 1);

  // Test shared blocks
  {
    Block block = manager.allocate();
    EXPECT_EQ(block.ref_count(), 1);
    EXPECT_EQ(block.is_shared(), false);
    // test copy constructor
    {
      // NOLINTNEXTLINE
      const Block block2 = block;
      EXPECT_EQ(block.ref_count(), 2);
      EXPECT_EQ(block.is_shared(), true);
      EXPECT_EQ(block2.ref_count(), 2);
      EXPECT_EQ(block2.is_shared(), true);
      EXPECT_EQ(block2, block);
    }
    EXPECT_EQ(block.ref_count(), 1);
    EXPECT_EQ(block.is_shared(), false);

    // test assignment operator
    {
      Block block4 = manager.allocate();
      block4 = block;
      EXPECT_EQ(block.ref_count(), 2);
      EXPECT_EQ(block.is_shared(), true);
      EXPECT_EQ(block4.ref_count(), 2);
      EXPECT_EQ(block4.is_shared(), true);
      EXPECT_EQ(block4, block);

      Block invalid_block;
      invalid_block = block;
      EXPECT_EQ(block.ref_count(), 3);
      EXPECT_EQ(block.is_shared(), true);
      EXPECT_EQ(invalid_block.ref_count(), 3);
      EXPECT_EQ(invalid_block.is_shared(), true);
      EXPECT_EQ(invalid_block, block);
    }
    EXPECT_EQ(block.ref_count(), 1);
    EXPECT_EQ(block.is_shared(), false);

    // test move constructor
    {
      Block block3 = std::move(block);
      EXPECT_FALSE(block.is_valid());

      EXPECT_EQ(block3.ref_count(), 1);
      EXPECT_EQ(block3.is_shared(), false);
      EXPECT_FALSE(block3 == block);
    }
    EXPECT_FALSE(block.is_valid());
  }
}

TEST(BlockManagerPoolTest, AllocateAssignsSingleBlockWhenEnabled) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 0);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  options.num_single_blocks(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seq = make_sequence(0, /*prompt_tokens=*/{1, 2, 3});
  EXPECT_TRUE(pool.allocate(&seq));
  EXPECT_TRUE(seq.get_single_block_id() >= 0);
  // id 0 is the reserved padding slot, so a real assignment is strictly
  // positive.
  EXPECT_GT(seq.get_single_block_id(), 0);
}

TEST(BlockManagerPoolTest, DeallocateReleasesSingleBlockId) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 0);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  options.num_single_blocks(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seq1 = make_sequence(0, /*prompt_tokens=*/{1, 2, 3});
  ASSERT_TRUE(pool.allocate(&seq1));
  const int32_t id1 = seq1.get_single_block_id();
  pool.deallocate(&seq1);
  EXPECT_FALSE(seq1.get_single_block_id() >= 0);

  Sequence seq2 = make_sequence(1, /*prompt_tokens=*/{4, 5, 6});
  ASSERT_TRUE(pool.allocate(&seq2));
  EXPECT_EQ(seq2.get_single_block_id(), id1);
}

TEST(BlockManagerPoolTest, SingleBlockCapacityUsesOptionsMaxSeqs) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 0);

  BlockManagerPool::Options options;
  options.num_blocks(16)
      .host_num_blocks(0)
      .block_size(1)
      .enable_prefix_cache(false)
      .max_seqs_per_batch(4);
  options.enable_linear_state(true).linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  std::vector<Sequence> sequences;
  sequences.reserve(4);
  for (size_t i = 0; i < 4; ++i) {
    sequences.emplace_back(make_sequence(i, /*prompt_tokens=*/{1}));
    EXPECT_TRUE(pool.allocate(&sequences.back()));
    EXPECT_TRUE(sequences.back().get_single_block_id() >= 0);
  }
}

TEST(BlockManagerPoolTest, TryAllocateKvFailureRollsBackSingleBlock) {
  BlockManagerPool::Options options;
  options.num_blocks(3).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  // id 0 is reserved for padding, so capacity 3 exposes 2 usable single-block
  // ids, enough for the two sequences allocated after the rollback. Zero out
  // max_seqs_per_batch so num_single_blocks alone drives the SINGLE pool size.
  options.max_seqs_per_batch(0)
      .num_single_blocks(3)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  // This sequence needs far more KV blocks than available, forcing KV failure
  // after embedding and linear ids are allocated.
  std::vector<int32_t> huge_prompt(100, 1);
  Sequence fail_seq = make_sequence(0, huge_prompt);
  EXPECT_FALSE(pool.try_allocate(&fail_seq));
  EXPECT_FALSE(fail_seq.get_single_block_id() >= 0);

  // The unified slot must have been rolled back, leaving enough capacity for
  // two new sequences to allocate.
  Sequence seq1 = make_sequence(1, /*prompt_tokens=*/{1});
  Sequence seq2 = make_sequence(2, /*prompt_tokens=*/{2});
  EXPECT_TRUE(pool.try_allocate(&seq1));
  EXPECT_TRUE(pool.try_allocate(&seq2));
  EXPECT_TRUE(seq1.get_single_block_id() >= 0);
  EXPECT_TRUE(seq2.get_single_block_id() >= 0);
}

TEST(BlockManagerPoolTest, SingleBlockCapacityCanBeLowerThanMaxSeqs) {
  BlockManagerPool::Options options;
  // id 0 is reserved for padding, so capacity 4 exposes 3 usable single blocks.
  // max_seqs_per_batch is 8 in the scheduler view, but num_single_blocks caps
  // the SINGLE pool at 4 (pool takes the max, so num_single_blocks wins only
  // when it is >= max_seqs_per_batch + 2 -- here it is 4 vs 8+2=10, so the
  // 8+2 path would win; force num_single_blocks to be the winner by lowering
  // the scheduler side).
  options.num_blocks(16)
      .host_num_blocks(0)
      .block_size(1)
      .max_seqs_per_batch(0)
      .num_single_blocks(4)
      .enable_prefix_cache(false);
  options.enable_linear_state(true).linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seq0 = make_sequence(0, /*prompt_tokens=*/{1});
  Sequence seq1 = make_sequence(1, /*prompt_tokens=*/{2});
  Sequence seq2 = make_sequence(2, /*prompt_tokens=*/{3});
  Sequence seq3 = make_sequence(3, /*prompt_tokens=*/{4});

  EXPECT_TRUE(pool.try_allocate(&seq0));
  EXPECT_TRUE(pool.try_allocate(&seq1));
  EXPECT_TRUE(pool.try_allocate(&seq2));
  EXPECT_FALSE(pool.try_allocate(&seq3));

  EXPECT_TRUE(seq0.get_single_block_id() >= 0);
  EXPECT_TRUE(seq1.get_single_block_id() >= 0);
  EXPECT_TRUE(seq2.get_single_block_id() >= 0);
  EXPECT_FALSE(seq3.get_single_block_id() >= 0);
}

TEST(BlockManagerPoolTest, DpRankSelectionSkipsExhaustedSingleBlockPool) {
  BlockManagerPool::Options options;
  // id 0 is reserved for padding, so capacity 2 exposes 1 usable block per
  // rank. Zero out max_seqs_per_batch so num_single_blocks alone drives the
  // SINGLE pool size.
  options.num_blocks(16)
      .host_num_blocks(0)
      .block_size(1)
      .max_seqs_per_batch(0)
      .num_single_blocks(2)
      .enable_prefix_cache(false);
  options.enable_linear_state(true).linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/2);

  Sequence seq0 = make_sequence(0, /*prompt_tokens=*/{1});
  ASSERT_TRUE(pool.try_allocate(&seq0));
  EXPECT_EQ(seq0.dp_rank(), 0);

  Sequence seq1 = make_sequence(1, /*prompt_tokens=*/{2});
  ASSERT_TRUE(pool.try_allocate(&seq1));
  EXPECT_EQ(seq1.dp_rank(), 1);
}

TEST(BlockManagerPoolTest, SingleBlockExhaustionBehavesLikeKvBlockExhaustion) {
  BlockManagerPool::Options options;
  // id 0 is reserved for padding, so capacity 2 exposes 1 usable block per
  // rank. Zero out max_seqs_per_batch so num_single_blocks alone drives the
  // SINGLE pool size.
  options.num_blocks(16)
      .host_num_blocks(0)
      .block_size(1)
      .max_seqs_per_batch(0)
      .num_single_blocks(2)
      .enable_prefix_cache(false);
  options.enable_linear_state(true).linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/2);

  Sequence seq0 = make_sequence(0, /*prompt_tokens=*/{1});
  Sequence seq1 = make_sequence(1, /*prompt_tokens=*/{2});
  ASSERT_TRUE(pool.try_allocate(&seq0));
  ASSERT_TRUE(pool.try_allocate(&seq1));

  Sequence retry = make_sequence(2, /*prompt_tokens=*/{3});
  EXPECT_FALSE(pool.try_allocate(&retry));
  EXPECT_EQ(retry.dp_rank(), 0);

  pool.deallocate(&seq0);
  ASSERT_TRUE(pool.try_allocate(&retry));
  EXPECT_EQ(retry.dp_rank(), 0);
}

TEST(BlockManagerPoolTest, AllocateAssignsSingleBlockWhenLinearStateDisabled) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seq = make_sequence(0, /*prompt_tokens=*/{1, 2});
  EXPECT_TRUE(pool.allocate(&seq));
  EXPECT_TRUE(seq.get_single_block_id() >= 0);
}

TEST(BlockManagerPoolTest, CapacityReportsKvPoolWhenLinearStateEnabled) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  // KV blocks are coarse (block_size=8) while the LINEAR leaf's slot pool has
  // block_size=1 and many more slots. The scheduler-facing capacity must follow
  // the KV admission leaf; the LINEAR leaf is a per-sequence resource slot and
  // must never win capacity_leaf() (it would otherwise report the linear slot
  // pool and hand the scheduler a wrong block budget). num_linear_state_slots
  // is set well above num_blocks so the two pools are unambiguous.
  constexpr size_t kKvBlocks = 32;
  constexpr int32_t kLinearStateSlots = 256;
  BlockManagerPool::Options options;
  options.num_blocks(kKvBlocks)
      .host_num_blocks(0)
      .block_size(8)
      .enable_prefix_cache(false);
  options.num_single_blocks(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(kLinearStateSlots);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 8);
  BlockManagerPool pool(options, /*dp_size=*/1);

  // Each leaf reserves slot 0 as padding, so a fresh KV pool reports
  // kKvBlocks - 1 free. The key invariant: capacity tracks the KV pool, never
  // the (much larger) linear slot pool.
  const std::vector<size_t> free = pool.num_free_blocks();
  ASSERT_EQ(free.size(), 1u);
  EXPECT_EQ(free[0], kKvBlocks - 1);
  EXPECT_LT(free[0], static_cast<size_t>(kLinearStateSlots));
}

TEST(BlockManagerPoolTest, SequenceCopyDoesNotReuseSingleBlockSlot) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  options.num_single_blocks(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence src = make_sequence(0, /*prompt_tokens=*/{1, 2, 3});
  ASSERT_TRUE(pool.allocate(&src));
  ASSERT_TRUE(src.get_single_block_id() >= 0);
  ASSERT_TRUE(src.has_linear_state_slot());
  const int32_t src_single_block_id = src.get_single_block_id();
  const int32_t src_linear_state_slot_id = src.get_linear_state_slot_id();

  Sequence clone(src);
  EXPECT_FALSE(clone.get_single_block_id() >= 0);
  EXPECT_EQ(clone.get_single_block_id(), -1);
  EXPECT_FALSE(clone.has_linear_state_slot());
  EXPECT_EQ(clone.get_linear_state_slot_id(), -1);

  ASSERT_TRUE(pool.allocate(&clone));
  EXPECT_TRUE(clone.get_single_block_id() >= 0);
  EXPECT_NE(clone.get_single_block_id(), src_single_block_id);
  EXPECT_TRUE(clone.has_linear_state_slot());
  EXPECT_NE(clone.get_linear_state_slot_id(), src_linear_state_slot_id);
}

TEST(BlockManagerPoolTest, AllocateAfterPrefixCacheHitAllocatesSuffixBlocks) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 2);

  BlockManagerPool::Options options;
  options.num_blocks(16).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.cache(&cached_seq);
  pool.deallocate(&cached_seq);

  Sequence hit_seq =
      make_sequence(1, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_TRUE(pool.allocate(&hit_seq, hit_seq.num_tokens()));
  EXPECT_EQ(hit_seq.kv_state().shared_blocks_num(BlockType::KV), 2);
  EXPECT_GE(hit_seq.kv_state().current_max_tokens_capacity(),
            hit_seq.num_tokens());
}

TEST(BlockManagerPoolTest, LinearStateBlockManagerMatchesOnlyCheckpointHashes) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.num_single_blocks(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64)
      .linear_chunk_stride(4);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.cache(&cached_seq);
  const LinearStatePrefixHash checkpoint_hash =
      compute_linear_state_prefix_hash_for_test(
          cached_seq.tokens(), options.block_size(), /*boundary_tokens=*/8);
  pool.deallocate_without_cache(&cached_seq);

  // Without a linear-state checkpoint for the prefix boundary, prefix reuse is
  // trimmed away: the recurrent state cannot be restored, so a hit would be
  // unsafe.
  Sequence miss_seq =
      make_sequence(1, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  const std::vector<size_t> used_blocks_before_miss = pool.num_used_blocks();
  ASSERT_TRUE(pool.allocate(&miss_seq, miss_seq.num_tokens()));
  EXPECT_EQ(miss_seq.kv_state().shared_blocks_num(BlockType::KV), 0u);
  pool.deallocate_without_cache(&miss_seq);
  EXPECT_EQ(pool.num_used_blocks(), used_blocks_before_miss);

  // Pin a checkpoint for the boundary hash directly in the slot pool, the same
  // way the scheduler does while resolving cache ops. Now the prefix is
  // reusable up to that boundary.
  LinearStateBlockManager* prefix_cache =
      BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  EXPECT_GE(insert_linear_state_checkpoint(prefix_cache, checkpoint_hash), 1);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(
      prefix_cache, XXH3Key(checkpoint_hash.data())));

  Sequence hit_seq =
      make_sequence(2,
                    /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  ASSERT_TRUE(pool.allocate(&hit_seq, hit_seq.num_tokens()));
  EXPECT_EQ(hit_seq.kv_state().shared_blocks_num(BlockType::KV), 2u);
  pool.deallocate_without_cache(&hit_seq);
}

// An exact prompt match must leave at least one token for the current forward.
// The matched tail boundary (h2 here) is checkpointed, but reusing it would pop
// back to the previous, uncheckpointed boundary (h1) and lose restorable state,
// so prefix reuse must be 0 instead.
TEST(BlockManagerPoolTest, ExactPromptCannotReuseUncheckpointedTailBoundary) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  BlockManagerPool::Options options;
  options.num_blocks(32).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.num_single_blocks(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64)
      .linear_chunk_stride(4);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.cache(&cached_seq);
  const LinearStatePrefixHash tail_hash =
      compute_linear_state_prefix_hash_for_test(
          cached_seq.tokens(), options.block_size(), /*boundary_tokens=*/8);
  pool.deallocate_without_cache(&cached_seq);

  LinearStateBlockManager* prefix_cache =
      BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  EXPECT_GE(insert_linear_state_checkpoint(prefix_cache, tail_hash), 1);

  // Exact 8-token prompt: max_reusable_blocks = floor((8 - 1) / 4) = 1, so the
  // checkpointed boundary at block 2 is out of reach and reuse falls to 0.
  Sequence exact_seq =
      make_sequence(1, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&exact_seq, exact_seq.num_tokens()));
  EXPECT_EQ(exact_seq.kv_state().shared_blocks_num(BlockType::KV), 0u);
  pool.deallocate_without_cache(&exact_seq);
}

// One token past the checkpoint boundary leaves work for the current forward,
// so the checkpointed boundary becomes reusable.
TEST(BlockManagerPoolTest, PromptPastCheckpointReusesCheckpointBoundary) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  BlockManagerPool::Options options;
  options.num_blocks(32).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.num_single_blocks(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64)
      .linear_chunk_stride(4);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.cache(&cached_seq);
  const LinearStatePrefixHash tail_hash =
      compute_linear_state_prefix_hash_for_test(
          cached_seq.tokens(), options.block_size(), /*boundary_tokens=*/8);
  pool.deallocate_without_cache(&cached_seq);

  LinearStateBlockManager* prefix_cache =
      BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  EXPECT_GE(insert_linear_state_checkpoint(prefix_cache, tail_hash), 1);

  // 9-token prompt: max_reusable_blocks = floor((9 - 1) / 4) = 2, so the
  // checkpointed boundary at block 2 is reusable.
  Sequence hit_seq =
      make_sequence(1, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8, 9});
  ASSERT_TRUE(pool.allocate(&hit_seq, hit_seq.num_tokens()));
  EXPECT_EQ(hit_seq.kv_state().shared_blocks_num(BlockType::KV), 2u);
  pool.deallocate_without_cache(&hit_seq);
}

TEST(BlockManagerPoolTest,
     PendingLinearSaveEvictsUnpinnedCheckpointButNotPinned) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(make_linear_state_pool_options(
                            /*linear_state_num_slots=*/4),
                        /*dp_size=*/1);

  LinearStateBlockManager* prefix_cache =
      BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);

  // Allocate the saving sequence first so its live slot does not contend with
  // the checkpoints we are about to insert.
  Sequence saving_seq =
      make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&saving_seq, saving_seq.num_tokens()));
  saving_seq.kv_state().set_kv_cache_tokens_num(4);
  const int32_t saving_live_slot = saving_seq.get_linear_state_slot_id();
  ASSERT_GE(saving_live_slot, 1);

  const LinearStatePrefixHash pinned_hash = make_prefix_hash(1);
  const LinearStatePrefixHash evictable_hash = make_prefix_hash(2);
  const LinearStatePrefixHash save_hash = make_prefix_hash(3);
  // Insert evictable first so that it is the LRU front; pinned_hash is inserted
  // after and then held by a live Block handle so eviction must skip it.
  ASSERT_GE(insert_linear_state_checkpoint(prefix_cache, evictable_hash), 1);
  ASSERT_GE(insert_linear_state_checkpoint(prefix_cache, pinned_hash), 1);
  // Pool is now full (padding + saving live + evictable + pinned). Pin the
  // checkpoint the way production does -- a held Block handle (refcount+1) that
  // keeps PrefixCache from evicting it while the sequence still needs it.
  Block pinned = BlockManagerPoolTestPeer::match(prefix_cache,
                                                 XXH3Key(pinned_hash.data()));
  ASSERT_TRUE(pinned.is_valid());

  // The batch builder records the pending save on the sequence at build time;
  // the LINEAR leaf executes it at the next step's allocate_for_sequence. With
  // the pool full, the rotation's fresh live slot must be reclaimed from the
  // unpinned LRU checkpoint (evictable), never from the pinned one.
  saving_seq.set_pending_linear_save(XXH3Key(save_hash.data()));
  std::vector<Sequence*> sequences = {&saving_seq};
  apply_pending_linear_saves_via_leaf(prefix_cache, sequences);

  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(prefix_cache,
                                                 XXH3Key(save_hash.data())));
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(prefix_cache,
                                                 XXH3Key(pinned_hash.data())));
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(
      prefix_cache, XXH3Key(evictable_hash.data())));

  pool.deallocate_without_cache(&saving_seq);
}

TEST(BlockManagerPoolTest,
     PendingLinearSavePromotesLiveSlotAndMountsRestoreSrc) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(make_linear_state_pool_options(
                            /*linear_state_num_slots=*/3),
                        /*dp_size=*/1);

  Sequence sequence =
      make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&sequence, sequence.num_tokens()));
  sequence.kv_state().set_kv_cache_tokens_num(4);
  const int32_t old_live_slot = sequence.get_linear_state_slot_id();
  ASSERT_GE(old_live_slot, 1);

  const LinearStatePrefixHash save_hash = make_prefix_hash(9);
  // The batch builder records the pending save on the sequence at build time.
  sequence.set_pending_linear_save(XXH3Key(save_hash.data()));
  std::vector<Sequence*> sequences = {&sequence};

  LinearStateBlockManager* prefix_cache =
      BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(prefix_cache,
                                                  XXH3Key(save_hash.data())));

  // allocate_for_sequence executes the deferred save: inserts old slot as
  // checkpoint, mounts it as the sequence's class-B restore source, and rotates
  // the sequence onto a fresh slot.
  apply_pending_linear_saves_via_leaf(prefix_cache, sequences);

  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(prefix_cache,
                                                 XXH3Key(save_hash.data())));
  Block matched =
      BlockManagerPoolTestPeer::match(prefix_cache, XXH3Key(save_hash.data()));
  ASSERT_TRUE(matched.is_valid());
  EXPECT_EQ(matched.id(), old_live_slot);
  const int32_t new_live_slot = sequence.get_linear_state_slot_id();
  EXPECT_NE(new_live_slot, old_live_slot);
  EXPECT_FALSE(sequence.has_pending_linear_save());

  // The save-rotation mounted the just-checkpointed slot as the restore source
  // (block-carried, no scheduler-side find()): the next build consumes it to
  // fill restore_src_slot_id. It must point at the old (checkpointed) slot.
  ASSERT_TRUE(sequence.has_linear_restore_src_block());
  std::optional<Block> restore_src = sequence.take_linear_restore_src_block();
  ASSERT_TRUE(restore_src.has_value());
  EXPECT_EQ(restore_src->id(), old_live_slot);

  pool.deallocate_without_cache(&sequence);
}

TEST(BlockManagerPoolTest, PendingLinearSaveSkipsWhenCheckpointAlreadyExists) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(make_linear_state_pool_options(
                            /*linear_state_num_slots=*/4),
                        /*dp_size=*/1);

  LinearStateBlockManager* prefix_cache =
      BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);

  Sequence sequence = make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4});
  ASSERT_TRUE(pool.allocate(&sequence, sequence.num_tokens()));
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_tokens());
  const int32_t live_slot = sequence.get_linear_state_slot_id();
  ASSERT_GE(live_slot, 1);

  // A checkpoint for this boundary already exists in the index (e.g. another
  // sequence in the same batch saved it first). The contains() guard in
  // allocate_for_sequence is the production dedup: the second save is a
  // no-op, so the sequence keeps its live slot warm instead of rotating.
  const LinearStatePrefixHash save_hash = make_prefix_hash(8);
  const int32_t existing_slot =
      insert_linear_state_checkpoint(prefix_cache, save_hash);
  ASSERT_GE(existing_slot, 1);
  ASSERT_NE(existing_slot, live_slot);

  sequence.set_pending_linear_save(XXH3Key(save_hash.data()));
  std::vector<Sequence*> sequences = {&sequence};
  apply_pending_linear_saves_via_leaf(prefix_cache, sequences);

  // The pending flag is consumed, but the save is skipped: no rotation, no
  // restore-src mount, and the pre-existing checkpoint is untouched.
  EXPECT_FALSE(sequence.has_pending_linear_save());
  EXPECT_EQ(sequence.get_linear_state_slot_id(), live_slot);
  EXPECT_FALSE(sequence.has_linear_restore_src_block());
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(prefix_cache,
                                                 XXH3Key(save_hash.data())));
  EXPECT_EQ(
      BlockManagerPoolTestPeer::match(prefix_cache, XXH3Key(save_hash.data()))
          .id(),
      existing_slot);

  pool.deallocate_without_cache(&sequence);
}

TEST(BlockManagerPoolTest, ApplyPendingSaveSkipsWhenSlotRemoved) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(make_linear_state_pool_options(
                            /*linear_state_num_slots=*/4),
                        /*dp_size=*/1);

  Sequence sequence =
      make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&sequence, sequence.num_tokens()));
  sequence.kv_state().set_kv_cache_tokens_num(4);
  const int32_t old_live_slot = sequence.get_linear_state_slot_id();
  ASSERT_GE(old_live_slot, 1);

  const LinearStatePrefixHash save_hash = make_prefix_hash(10);
  // The batch builder records the pending save on the sequence at build time.
  sequence.set_pending_linear_save(XXH3Key(save_hash.data()));
  std::vector<Sequence*> sequences = {&sequence};
  EXPECT_TRUE(sequence.has_pending_linear_save());

  // Simulate the linear slot being removed between build and apply
  // (e.g. sequence preempted then re-scheduled without a LINEAR slot). This is
  // the same primitive production uses to drop a sequence's private LINEAR
  // block (see Sequence's forked-sequence constructor).
  sequence.kv_state().erase_blocks(BlockType::LINEAR);
  // erase_blocks(LINEAR) also clears the deferred save (the recorded slot is
  // gone), so the apply below simply takes the first-allocate path.
  EXPECT_FALSE(sequence.has_pending_linear_save());

  LinearStateBlockManager* prefix_cache =
      BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  apply_pending_linear_saves_via_leaf(prefix_cache, sequences);

  // No checkpoint is published (the deferred save was cleared with the slot).
  // But allocate_for_sequence owns the full slot lifecycle: after the save is
  // skipped it takes the first-allocate path and hands the slot-less sequence a
  // fresh live slot, exactly as a re-scheduled sequence expects.
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(prefix_cache,
                                                  XXH3Key(save_hash.data())));
  EXPECT_TRUE(sequence.has_linear_state_slot());
  EXPECT_FALSE(sequence.has_pending_linear_save());

  pool.deallocate_without_cache(&sequence);
}

TEST(BlockManagerPoolTest,
     PendingLinearSaveDedupsDuplicateSavesAcrossSequences) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(make_linear_state_pool_options(
                            /*linear_state_num_slots=*/5),
                        /*dp_size=*/1);

  LinearStateBlockManager* prefix_cache =
      BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);

  Sequence seq1 = make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  Sequence seq2 =
      make_sequence(1, /*prompt_tokens=*/{9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_TRUE(pool.allocate(&seq1, seq1.num_tokens()));
  ASSERT_TRUE(pool.allocate(&seq2, seq2.num_tokens()));
  seq1.kv_state().set_kv_cache_tokens_num(4);
  seq2.kv_state().set_kv_cache_tokens_num(4);
  const int32_t live_slot1 = seq1.get_linear_state_slot_id();
  const int32_t live_slot2 = seq2.get_linear_state_slot_id();
  ASSERT_GE(live_slot1, 1);
  ASSERT_GE(live_slot2, 1);
  ASSERT_NE(live_slot1, live_slot2);

  const LinearStatePrefixHash save_hash = make_prefix_hash(7);
  // Two sequences in the same batch save the same prefix hash. Each records its
  // own pending save at build time (thread-safe: touches only its own field).
  seq1.set_pending_linear_save(XXH3Key(save_hash.data()));
  seq2.set_pending_linear_save(XXH3Key(save_hash.data()));
  std::vector<Sequence*> sequences = {&seq1, &seq2};

  EXPECT_TRUE(seq1.has_pending_linear_save());
  EXPECT_TRUE(seq2.has_pending_linear_save());
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(prefix_cache,
                                                  XXH3Key(save_hash.data())));

  // Apply runs per sequence: seq1 publishes its checkpoint and rotates; seq2
  // then hits allocate_for_sequence's contains() guard (the production
  // dedup) and no-ops, keeping its live slot warm instead of double-publishing.
  apply_pending_linear_saves_via_leaf(prefix_cache, sequences);

  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(prefix_cache,
                                                 XXH3Key(save_hash.data())));
  Block matched =
      BlockManagerPoolTestPeer::match(prefix_cache, XXH3Key(save_hash.data()));
  ASSERT_TRUE(matched.is_valid());
  EXPECT_EQ(matched.id(), live_slot1);
  EXPECT_FALSE(seq1.has_pending_linear_save());
  EXPECT_FALSE(seq2.has_pending_linear_save());
  // seq1 rotated onto a fresh slot and mounted its checkpoint as restore src;
  // seq2 was deduped, so it kept its live slot and mounted nothing.
  EXPECT_NE(seq1.get_linear_state_slot_id(), live_slot1);
  EXPECT_TRUE(seq1.has_linear_restore_src_block());
  EXPECT_EQ(seq2.get_linear_state_slot_id(), live_slot2);
  EXPECT_FALSE(seq2.has_linear_restore_src_block());

  pool.deallocate_without_cache(&seq1);
  pool.deallocate_without_cache(&seq2);
}

TEST(BlockManagerPoolTest, SparseLinearStateCheckpointTrimsSharedKVBlocks) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  const auto check_shared_blocks = [](int32_t checkpoint_block_index,
                                      size_t expected_shared_blocks) {
    BlockManagerPool::Options options;
    options.num_blocks(48).host_num_blocks(0).block_size(4).enable_prefix_cache(
        true);
    options.num_single_blocks(FLAGS_max_seqs_per_batch + 2)
        .enable_linear_state(true)
        .linear_state_num_slots(64)
        .linear_chunk_stride(4);
    ScopedValue<int32_t> chunk_guard(
        &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
    BlockManagerPool pool(options, /*dp_size=*/1);

    Sequence cached_seq = make_sequence(
        0,
        /*prompt_tokens=*/{
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    ASSERT_TRUE(pool.allocate(&cached_seq));
    cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
    pool.cache(&cached_seq);

    LinearStateBlockManager* prefix_cache =
        BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0);
    ASSERT_NE(prefix_cache, nullptr);
    if (checkpoint_block_index > 0) {
      const size_t boundary_tokens =
          static_cast<size_t>(checkpoint_block_index) *
          static_cast<size_t>(options.block_size());
      const LinearStatePrefixHash checkpoint_hash =
          compute_linear_state_prefix_hash_for_test(
              cached_seq.tokens(), options.block_size(), boundary_tokens);
      EXPECT_GE(insert_linear_state_checkpoint(prefix_cache, checkpoint_hash),
                1);
    }
    pool.deallocate_without_cache(&cached_seq);

    Sequence hit_seq = make_sequence(
        1,
        /*prompt_tokens=*/{
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
    ASSERT_TRUE(pool.allocate(&hit_seq, hit_seq.num_tokens()));
    EXPECT_EQ(hit_seq.kv_state().shared_blocks_num(BlockType::KV),
              expected_shared_blocks);
    pool.deallocate_without_cache(&hit_seq);
  };

  // KV prefix cache stores h1-h4. The linear-state cache may store only one of
  // those shared hashes; reuse must stop at the latest checkpoint-backed hash.
  check_shared_blocks(/*checkpoint_block_index=*/4,
                      /*expected_shared_blocks=*/4);
  check_shared_blocks(/*checkpoint_block_index=*/3,
                      /*expected_shared_blocks=*/3);
  check_shared_blocks(/*checkpoint_block_index=*/0,
                      /*expected_shared_blocks=*/0);
}

TEST(BlockManagerPoolTest, SparseLinearStateCheckpointCannotExceedKVMatch) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  BlockManagerPool::Options options;
  options.num_blocks(48).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.num_single_blocks(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64)
      .linear_chunk_stride(4);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence short_cached_seq =
      make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&short_cached_seq));
  short_cached_seq.kv_state().set_kv_cache_tokens_num(
      short_cached_seq.num_tokens());
  pool.cache(&short_cached_seq);
  pool.deallocate_without_cache(&short_cached_seq);

  Sequence long_cached_seq =
      make_sequence(1,
                    /*prompt_tokens=*/{
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_TRUE(pool.allocate(&long_cached_seq));
  const LinearStatePrefixHash kv_boundary_hash =
      compute_linear_state_prefix_hash_for_test(long_cached_seq.tokens(),
                                                options.block_size(),
                                                /*boundary_tokens=*/8);
  const LinearStatePrefixHash long_checkpoint_hash =
      compute_linear_state_prefix_hash_for_test(long_cached_seq.tokens(),
                                                options.block_size(),
                                                /*boundary_tokens=*/16);
  pool.deallocate_without_cache(&long_cached_seq);

  LinearStateBlockManager* prefix_cache =
      BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  EXPECT_GE(insert_linear_state_checkpoint(prefix_cache, kv_boundary_hash), 1);
  EXPECT_GE(insert_linear_state_checkpoint(prefix_cache, long_checkpoint_hash),
            1);

  Sequence hit_seq =
      make_sequence(2, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8, 17});
  ASSERT_TRUE(pool.allocate(&hit_seq, hit_seq.num_tokens()));
  EXPECT_EQ(hit_seq.kv_state().shared_blocks_num(BlockType::KV), 2u);
  pool.deallocate_without_cache(&hit_seq);
}

TEST(BlockManagerPoolTest, ExactPromptStopsAtEarlierCheckpoint) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  BlockManagerPool::Options options;
  options.num_blocks(32).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.num_single_blocks(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64)
      .linear_chunk_stride(4);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      make_sequence(0,
                    /*prompt_tokens=*/{
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_TRUE(pool.allocate(&cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.cache(&cached_seq);
  const LinearStatePrefixHash inner_hash =
      compute_linear_state_prefix_hash_for_test(
          cached_seq.tokens(), options.block_size(), /*boundary_tokens=*/8);
  const LinearStatePrefixHash tail_hash =
      compute_linear_state_prefix_hash_for_test(
          cached_seq.tokens(), options.block_size(), /*boundary_tokens=*/16);
  pool.deallocate_without_cache(&cached_seq);

  LinearStateBlockManager* prefix_cache =
      BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  EXPECT_GE(insert_linear_state_checkpoint(prefix_cache, inner_hash), 1);
  EXPECT_GE(insert_linear_state_checkpoint(prefix_cache, tail_hash), 1);

  // Exact 16-token prompt: max_reusable_blocks = floor((16 - 1) / 4) = 3, so
  // the tail checkpoint at block 4 is out of reach; reuse stops at the inner
  // checkpoint (block 2).
  Sequence exact_seq =
      make_sequence(1,
                    /*prompt_tokens=*/{
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_TRUE(pool.allocate(&exact_seq, exact_seq.num_tokens()));
  EXPECT_EQ(exact_seq.kv_state().shared_blocks_num(BlockType::KV), 2u);
  pool.deallocate_without_cache(&exact_seq);
}

namespace {

// Independently compute the chained block hashes for `tokens`, mirroring
// xxh3_128bits_hash() so we can check Sequence::update_block_hashes().
std::vector<XXH3Key> ExpectedChain(const std::vector<int32_t>& tokens,
                                   uint32_t block_size) {
  const size_t n_blocks = tokens.size() / block_size;
  std::vector<XXH3Key> hashes;
  hashes.reserve(n_blocks);
  const Slice<int32_t> slice(tokens);
  for (size_t b = 0; b < n_blocks; ++b) {
    XXH3Key key;
    const uint8_t* pre = (b == 0) ? nullptr : hashes.back().data;
    xxh3_128bits_hash(
        pre, slice.slice(b * block_size, (b + 1) * block_size), key.data);
    hashes.emplace_back(key);
  }
  return hashes;
}

}  // namespace

// Validates the production hash builder Sequence::update_block_hashes():
// correct chain, idempotency, and invalidation after a token rewrite.
TEST(BlockManagerPoolTest, SequenceUpdateBlockHashes) {
  const uint32_t block_size = 4;
  const uint32_t n_blocks = 5;
  std::vector<int32_t> prompt;
  prompt.reserve(n_blocks * block_size);
  for (uint32_t i = 0; i < n_blocks * block_size; ++i) {
    prompt.push_back(static_cast<int32_t>(i * 7 + 1));
  }

  // Build the Sequence in-test so the sampling/stopping params outlive it.
  RequestSamplingParam sampling_param;
  sampling_param.beam_width = 0;
  sampling_param.is_embeddings = false;
  StoppingChecker stopping_checker;
  SequenceParams params;
  params.seq_capacity = prompt.size() + 8;
  params.bos_token_id = 0;
  params.request_id = "seq_block_hash_test";
  params.sampling_param = &sampling_param;
  params.stopping_checker = &stopping_checker;
  IncrementalDecoder decoder(/*prompt=*/"prompt",
                             /*num_prompt_tokens=*/prompt.size(),
                             /*echo=*/false,
                             /*skip_special_tokens=*/true);
  Sequence seq(/*index=*/0,
               prompt,
               /*input_embedding=*/torch::Tensor(),
               /*mm_data=*/MMData(),
               decoder,
               params);

  const std::vector<XXH3Key> expected = ExpectedChain(prompt, block_size);

  seq.update_block_hashes(block_size, BlockHasherType::TEXT);
  ASSERT_EQ(seq.block_hashes().size(), n_blocks);
  for (uint32_t i = 0; i < n_blocks; ++i) {
    EXPECT_EQ(std::memcmp(seq.block_hashes()[i].data,
                          expected[i].data,
                          XXH3_128BITS_HASH_VALUE_LEN),
              0);
  }

  // Idempotent: no new full block -> nothing recomputed/appended.
  seq.update_block_hashes(block_size, BlockHasherType::TEXT);
  EXPECT_EQ(seq.block_hashes().size(), n_blocks);

  // Rewriting a token in block index 2 invalidates block 2 and everything
  // after it; blocks 0 and 1 survive.
  seq.update_token(2 * block_size + 1, Token(/*id=*/999999));
  EXPECT_EQ(seq.block_hashes().size(), 2u);

  // Recompute: blocks 0/1 unchanged, block 2 differs from the old chain.
  seq.update_block_hashes(block_size, BlockHasherType::TEXT);
  ASSERT_EQ(seq.block_hashes().size(), n_blocks);
  EXPECT_EQ(std::memcmp(seq.block_hashes()[0].data,
                        expected[0].data,
                        XXH3_128BITS_HASH_VALUE_LEN),
            0);
  EXPECT_EQ(std::memcmp(seq.block_hashes()[1].data,
                        expected[1].data,
                        XXH3_128BITS_HASH_VALUE_LEN),
            0);
  EXPECT_NE(std::memcmp(seq.block_hashes()[2].data,
                        expected[2].data,
                        XXH3_128BITS_HASH_VALUE_LEN),
            0);
}

// In-batch prefix cache publishes only the full blocks covered by the given
// token budget. When the budget is overestimated, cache() must clamp to the
// sequence's own tokens and register just the complete blocks.
TEST(BlockManagerPoolTest,
     CachePrefixClampsToSequenceTokensWhenBudgetIsOverestimated) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(16).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  // Leak the pool intentionally: with prefix cache enabled, the cached block
  // stays referenced by the prefix-cache table at teardown, which would trip
  // the free-list check in ~BlockManagerImpl.
  auto* pool = new BlockManagerPool(options, /*dp_size=*/1);

  Sequence seq = make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7});
  ASSERT_TRUE(pool->allocate(&seq));

  // num_tokens (8) is larger than the 7 real tokens. cache() must clamp to the
  // sequence tokens, so only the single full block (tokens [0, 4)) is cached.
  EXPECT_NO_FATAL_FAILURE(pool->cache(&seq, /*num_tokens=*/8));
  EXPECT_EQ(pool->num_blocks_in_prefix_cache()[0], 1u);
}

}  // namespace xllm
