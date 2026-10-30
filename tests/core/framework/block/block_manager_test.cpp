/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "common/global_flags.h"
#include "core/framework/batch/batch.h"
#include "core/framework/block/block_manager_pool.h"
#include "core/framework/block/kv_cache_manager_factory.h"
#include "core/framework/config/scheduler_config.h"
#include "core/kv_cache/block/block_manager_impl.h"
#include "core/kv_cache/block/linear_state_block_manager.h"
#include "core/kv_cache/prefix/block_hasher.h"
#include "core/scheduler/dp_rank_selector.h"
#include "framework/block/block_manager_pool_test_peer.h"
#include "framework/model/model_params_view.h"
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

bool allocate_kv_prefix_for_test(BlockManagerPool& pool, Sequence& sequence) {
  const int32_t dp_rank = 0;
  std::vector<Block> blocks = pool.allocate(sequence.num_tokens(), dp_rank);
  if (blocks.empty()) {
    return false;
  }
  sequence.set_dp_rank(dp_rank);
  sequence.add_blocks(BlockType::KV, std::move(blocks));
  return true;
}

bool allocate_next_linear_chunk(BlockManagerPool& pool, Sequence& sequence) {
  pool.allocate_shared(&sequence);
  const size_t chunk_stride =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  const size_t target = std::min(sequence.num_tokens(),
                                 sequence.kv_cache_tokens_num() + chunk_stride);
  return pool.allocate(&sequence, target);
}

class BlockManagerPoolTest : public ::testing::Test {
 protected:
  BlockManagerPoolTest() {
    sampling_param_.beam_width = 0;
    sampling_param_.is_embeddings = false;
  }

  Sequence make_sequence(size_t index,
                         const std::vector<int32_t>& prompt_tokens) {
    SequenceParams params;
    params.seq_capacity = prompt_tokens.size() + 8;
    params.echo = false;
    params.skip_special_tokens = true;
    params.streaming = false;
    params.enable_schedule_overlap = false;
    params.rec_type = RecType::kNone;
    params.bos_token_id = 0;
    params.request_id = "block_manager_pool_test";
    params.sampling_param = &sampling_param_;
    params.stopping_checker = &stopping_checker_;

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

  RequestSamplingParam sampling_param_;
  StoppingChecker stopping_checker_;
};

BlockManagerPool::Options make_linear_state_pool_options(
    int32_t linear_state_num_slots) {
  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  // The EMBEDDING pool is sized directly by num_embedding_blocks and is only
  // built under spec decode, so opt in here.
  options.max_seqs_per_batch(0)
      .num_embedding_blocks(4)
      .num_speculative_tokens(1)
      .enable_linear_state(true)
      .linear_state_num_slots(linear_state_num_slots);
  return options;
}

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

TEST(KVCacheManagerFactoryTest,
     RejectsVirtualMemoryHostOffloadBeforeAllocation) {
  BlockManagerPool::Options options;
  options.enable_vmm_kvcache(true).enable_host_offload(true);

  EXPECT_DEATH(KVCacheManagerFactory::create(KVCacheCapacity{},
                                             ModelArgs{},
                                             /*world_size=*/1,
                                             torch::kBFloat16,
                                             options,
                                             /*transfer_coordinator=*/nullptr),
               "Virtual memory does not support host KV cache offload");
}

TEST(KVCacheManagerFactoryTest,
     RejectsVirtualMemoryHostOffloadConfigurationBeforeAllocation) {
  BlockManagerPool::Options options;
  options.enable_vmm_kvcache(true);
  HostCacheValidationOptions host_options;
  host_options.host_blocks_factor = 2.0;

  EXPECT_DEATH(KVCacheManagerFactory::create(KVCacheCapacity{},
                                             ModelArgs{},
                                             /*world_size=*/1,
                                             torch::kBFloat16,
                                             options,
                                             /*transfer_coordinator=*/nullptr,
                                             /*dp_size=*/1,
                                             host_options),
               "Virtual memory does not support host KV cache offload");
}

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

TEST_F(BlockManagerPoolTest, AllocateAssignsSingleBlockWhenEnabled) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 0);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  options.num_embedding_blocks(FLAGS_max_seqs_per_batch + 2)
      .num_speculative_tokens(1)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seq = make_sequence(0, /*prompt_tokens=*/{1, 2, 3});
  EXPECT_TRUE(pool.allocate(&seq));
  EXPECT_TRUE(seq.get_embedding_block_id() >= 0);
  // id 0 is the reserved padding slot, so a real assignment is strictly
  // positive.
  EXPECT_GT(seq.get_embedding_block_id(), 0);
}

TEST_F(BlockManagerPoolTest, SingleRankReleasesUnboundOwnedBlocks) {
  for (const bool without_cache : {false, true}) {
    BlockManagerPool::Options options;
    options.num_blocks(/*num_blocks=*/8)
        .block_size(/*block_size=*/2)
        .enable_prefix_cache(/*enable_prefix_cache=*/false);
    BlockManagerPool pool(options, /*dp_size=*/1);
    std::vector<Block> blocks = pool.allocate(/*num_tokens=*/4, /*dp_rank=*/0);
    ASSERT_EQ(blocks.size(), 2u);
    Sequence seq = make_sequence(/*index=*/0, /*prompt_tokens=*/{1, 2, 3, 4});
    seq.add_blocks(BlockType::KV, blocks);
    blocks.clear();
    ASSERT_EQ(seq.dp_rank(), -1);
    ASSERT_TRUE(seq.has_any_blocks());

    if (without_cache) {
      pool.deallocate_without_cache(&seq);
    } else {
      pool.deallocate(&seq);
    }

    EXPECT_EQ(seq.dp_rank(), 0);
    EXPECT_FALSE(seq.has_any_blocks());
    EXPECT_EQ(pool.num_used_blocks(), (std::vector<size_t>{0}));
  }
}

TEST_F(BlockManagerPoolTest, DpLinearStateAndPrefixCachesAreIsolated) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool::Options options;
  options.num_blocks(8)
      .host_num_blocks(0)
      .block_size(4)
      .enable_prefix_cache(true)
      .enable_linear_state(true)
      .linear_state_num_slots(4);
  BlockManagerPool pool(options, /*dp_size=*/2);

  Sequence first = make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4});
  Sequence second = make_sequence(1, /*prompt_tokens=*/{5, 6, 7, 8});
  first.set_dp_rank(0);
  second.set_dp_rank(1);

  ASSERT_TRUE(pool.allocate(&first));
  ASSERT_TRUE(pool.allocate(&second));
  EXPECT_EQ(first.dp_rank(), 0);
  EXPECT_EQ(second.dp_rank(), 1);
  // Slot 0 is reserved independently in each DP-local slot namespace, so both
  // replicas may validly assign the same positive slot id.
  const int32_t second_output_id = second.get_linear_state_slot_id();
  EXPECT_GT(second_output_id, 0);
  EXPECT_EQ(first.get_linear_state_slot_id(), second_output_id);

  LinearStateBlockManager* first_linear =
      BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0);
  LinearStateBlockManager* second_linear =
      BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/1);
  ASSERT_NE(first_linear, nullptr);
  ASSERT_NE(second_linear, nullptr);
  ASSERT_NE(first_linear, second_linear);

  const LinearStatePrefixHash first_only_hash = make_prefix_hash(42);
  ASSERT_GE(insert_linear_state_checkpoint(first_linear, first_only_hash), 1);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(
      first_linear, XXH3Key(first_only_hash.data())));
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(
      second_linear, XXH3Key(first_only_hash.data())));

  pool.deallocate_without_cache(&first);
  EXPECT_EQ(second.dp_rank(), 1);
  EXPECT_EQ(second.get_linear_state_slot_id(), second_output_id);
  pool.deallocate_without_cache(&second);
}

TEST_F(BlockManagerPoolTest,
       DpSelectionCountsEvictablePrefixBlocksAsAvailable) {
  BlockManagerPool::Options options;
  options.num_blocks(4).block_size(1).enable_prefix_cache(true);
  BlockManagerPool pool(options, /*dp_size=*/2);
  DpRankSelector selector;

  Sequence active = make_sequence(0, /*prompt_tokens=*/{1});
  active.set_dp_rank(0);
  ASSERT_TRUE(pool.try_allocate(&active));

  Sequence cached = make_sequence(1, /*prompt_tokens=*/{2, 3, 4});
  cached.set_dp_rank(1);
  ASSERT_TRUE(pool.try_allocate(&cached));
  pool.deallocate(&cached);

  const std::vector<size_t> free_blocks = pool.num_free_blocks();
  const std::vector<size_t> used_blocks = pool.num_used_blocks();
  ASSERT_EQ(free_blocks.size(), 2u);
  ASSERT_EQ(used_blocks.size(), 2u);
  EXPECT_GT(free_blocks[0], free_blocks[1]);
  EXPECT_GT(used_blocks[0], used_blocks[1]);
  EXPECT_DOUBLE_EQ(pool.kv_cache_utilization(), 0.0);
  EXPECT_EQ(selector.select(pool.dp_rank_loads()), 1);

  pool.deallocate(&active);
}

TEST_F(BlockManagerPoolTest, DeallocateReleasesSingleBlockId) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 0);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  options.num_embedding_blocks(FLAGS_max_seqs_per_batch + 2)
      .num_speculative_tokens(1)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seq1 = make_sequence(0, /*prompt_tokens=*/{1, 2, 3});
  ASSERT_TRUE(pool.allocate(&seq1));
  const int32_t id1 = seq1.get_embedding_block_id();
  pool.deallocate(&seq1);
  EXPECT_FALSE(seq1.get_embedding_block_id() >= 0);

  Sequence seq2 = make_sequence(1, /*prompt_tokens=*/{4, 5, 6});
  ASSERT_TRUE(pool.allocate(&seq2));
  EXPECT_EQ(seq2.get_embedding_block_id(), id1);
}

TEST_F(BlockManagerPoolTest, EmbeddingBlockCapacityUsesNumEmbeddingBlocks) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 0);

  BlockManagerPool::Options options;
  // The EMBEDDING pool is sized directly by num_embedding_blocks (id 0 is
  // reserved for padding, so 5 exposes 4 usable ids for the 4 sequences).
  options.num_blocks(16)
      .host_num_blocks(0)
      .block_size(1)
      .enable_prefix_cache(false)
      .max_seqs_per_batch(4);
  options.num_embedding_blocks(5)
      .num_speculative_tokens(1)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  std::vector<Sequence> sequences;
  sequences.reserve(4);
  for (size_t i = 0; i < 4; ++i) {
    sequences.emplace_back(make_sequence(i, /*prompt_tokens=*/{1}));
    EXPECT_TRUE(pool.allocate(&sequences.back()));
    EXPECT_TRUE(sequences.back().get_embedding_block_id() >= 0);
  }
}

TEST_F(BlockManagerPoolTest, TryAllocateKvFailureRollsBackSingleBlock) {
  BlockManagerPool::Options options;
  options.num_blocks(3).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  // id 0 is reserved for padding, so capacity 3 exposes 2 usable single-block
  // ids, enough for the two sequences allocated after the rollback. Zero out
  // max_seqs_per_batch so num_embedding_blocks alone drives the EMBEDDING pool
  // size.
  options.max_seqs_per_batch(0)
      .num_embedding_blocks(3)
      .num_speculative_tokens(1)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  std::vector<int32_t> huge_prompt(4, 1);
  Sequence fail_seq = make_sequence(0, huge_prompt);
  EXPECT_FALSE(pool.try_allocate(&fail_seq));
  EXPECT_FALSE(fail_seq.get_embedding_block_id() >= 0);

  // The unified slot must have been rolled back, leaving enough capacity for
  // two new sequences to allocate.
  Sequence seq1 = make_sequence(1, /*prompt_tokens=*/{1});
  Sequence seq2 = make_sequence(2, /*prompt_tokens=*/{2});
  EXPECT_TRUE(pool.try_allocate(&seq1));
  EXPECT_TRUE(pool.try_allocate(&seq2));
  EXPECT_TRUE(seq1.get_embedding_block_id() >= 0);
  EXPECT_TRUE(seq2.get_embedding_block_id() >= 0);
}

TEST_F(BlockManagerPoolTest, SingleBlockCapacityCanBeLowerThanMaxSeqs) {
  BlockManagerPool::Options options;
  // The EMBEDDING pool is sized directly by num_embedding_blocks, independent
  // of max_seqs_per_batch. id 0 is reserved for padding, so 4 exposes 3 usable
  // ids -- a deliberately smaller pool than the scheduler's max_seqs view.
  options.num_blocks(16)
      .host_num_blocks(0)
      .block_size(1)
      .max_seqs_per_batch(0)
      .num_embedding_blocks(4)
      .enable_prefix_cache(false);
  options.num_speculative_tokens(1)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
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

  EXPECT_TRUE(seq0.get_embedding_block_id() >= 0);
  EXPECT_TRUE(seq1.get_embedding_block_id() >= 0);
  EXPECT_TRUE(seq2.get_embedding_block_id() >= 0);
  EXPECT_FALSE(seq3.get_embedding_block_id() >= 0);
}

TEST_F(BlockManagerPoolTest, DpRankSelectionSkipsExhaustedSingleBlockPool) {
  BlockManagerPool::Options options;
  // id 0 is reserved for padding, so capacity 2 exposes 1 usable block per
  // rank. Zero out max_seqs_per_batch so num_embedding_blocks alone drives the
  // EMBEDDING pool size.
  options.num_blocks(16)
      .host_num_blocks(0)
      .block_size(1)
      .max_seqs_per_batch(0)
      .num_embedding_blocks(2)
      .enable_prefix_cache(false);
  options.num_speculative_tokens(1)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/2);
  DpRankSelector selector;

  Sequence seq0 = make_sequence(0, /*prompt_tokens=*/{1});
  seq0.set_dp_rank(selector.select(pool.dp_rank_loads()));
  ASSERT_TRUE(pool.try_allocate(&seq0));
  EXPECT_EQ(seq0.dp_rank(), 0);

  Sequence seq1 = make_sequence(1, /*prompt_tokens=*/{2});
  seq1.set_dp_rank(selector.select(pool.dp_rank_loads()));
  ASSERT_TRUE(pool.try_allocate(&seq1));
  EXPECT_EQ(seq1.dp_rank(), 1);
}

TEST_F(BlockManagerPoolTest,
       SingleBlockExhaustionBehavesLikeKvBlockExhaustion) {
  BlockManagerPool::Options options;
  // id 0 is reserved for padding, so capacity 2 exposes 1 usable block per
  // rank. Zero out max_seqs_per_batch so num_embedding_blocks alone drives the
  // EMBEDDING pool size.
  options.num_blocks(16)
      .host_num_blocks(0)
      .block_size(1)
      .max_seqs_per_batch(0)
      .num_embedding_blocks(2)
      .enable_prefix_cache(false);
  options.num_speculative_tokens(1)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/2);
  DpRankSelector selector;

  Sequence seq0 = make_sequence(0, /*prompt_tokens=*/{1});
  Sequence seq1 = make_sequence(1, /*prompt_tokens=*/{2});
  seq0.set_dp_rank(selector.select(pool.dp_rank_loads()));
  ASSERT_TRUE(pool.try_allocate(&seq0));
  seq1.set_dp_rank(selector.select(pool.dp_rank_loads()));
  ASSERT_TRUE(pool.try_allocate(&seq1));

  Sequence retry = make_sequence(2, /*prompt_tokens=*/{3});
  retry.set_dp_rank(selector.select(pool.dp_rank_loads()));
  EXPECT_FALSE(pool.try_allocate(&retry));
  const int32_t retry_dp_rank = retry.dp_rank();
  ASSERT_TRUE(retry_dp_rank == seq0.dp_rank() ||
              retry_dp_rank == seq1.dp_rank());

  Sequence* selected_sequence = retry_dp_rank == seq0.dp_rank() ? &seq0 : &seq1;
  pool.deallocate(selected_sequence);
  ASSERT_TRUE(pool.try_allocate(&retry));
  EXPECT_EQ(retry.dp_rank(), retry_dp_rank);
}

TEST_F(BlockManagerPoolTest,
       AllocateAssignsSingleBlockWhenLinearStateDisabled) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  options.num_speculative_tokens(1).num_embedding_blocks(4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seq = make_sequence(0, /*prompt_tokens=*/{1, 2});
  EXPECT_TRUE(pool.allocate(&seq));
  EXPECT_TRUE(seq.get_embedding_block_id() >= 0);
}

TEST_F(BlockManagerPoolTest, CapacityReportsKvPoolWhenLinearStateEnabled) {
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
  options.num_embedding_blocks(FLAGS_max_seqs_per_batch + 2)
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

TEST_F(BlockManagerPoolTest, SequenceCopyDoesNotReuseSingleBlockSlot) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  options.num_embedding_blocks(FLAGS_max_seqs_per_batch + 2)
      .num_speculative_tokens(1)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence src = make_sequence(0, /*prompt_tokens=*/{1, 2, 3});
  ASSERT_TRUE(pool.allocate(&src));
  ASSERT_TRUE(src.get_embedding_block_id() >= 0);
  ASSERT_TRUE(src.has_linear_state_slot());
  const int32_t src_single_block_id = src.get_embedding_block_id();
  const int32_t src_linear_state_slot_id = src.get_linear_state_slot_id();

  Sequence clone(src);
  EXPECT_FALSE(clone.get_embedding_block_id() >= 0);
  EXPECT_EQ(clone.get_embedding_block_id(), -1);
  EXPECT_FALSE(clone.has_linear_state_slot());
  EXPECT_EQ(clone.get_linear_state_slot_id(), -1);

  ASSERT_TRUE(pool.allocate(&clone));
  EXPECT_TRUE(clone.get_embedding_block_id() >= 0);
  EXPECT_NE(clone.get_embedding_block_id(), src_single_block_id);
  EXPECT_TRUE(clone.has_linear_state_slot());
  EXPECT_NE(clone.get_linear_state_slot_id(), src_linear_state_slot_id);
}

TEST_F(BlockManagerPoolTest, AllocateAfterPrefixCacheHitAllocatesSuffixBlocks) {
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

TEST_F(BlockManagerPoolTest, PdPrefillPublishesReusableComputedPrefix) {
  auto journal = std::make_shared<KVCacheEventJournal>();
  BlockManagerPool::Options options;
  options.num_blocks(16)
      .block_size(4)
      .enable_prefix_cache(true)
      .enable_disagg_pd(true)
      .event_journal(journal);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence source =
      make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&source));
  EXPECT_TRUE(journal->read("", 0)["events"].empty());
  source.kv_state().set_kv_cache_tokens_num(source.num_tokens());
  pool.cache(&source);
  pool.deallocate(&source);

  const nlohmann::json snapshot = journal->read("", 0);
  ASSERT_EQ(snapshot["events"].size(), 2);
  for (const auto& event : snapshot["events"]) {
    EXPECT_EQ(event["medium"], "gpu");
  }
  Sequence hit =
      make_sequence(1, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_TRUE(pool.allocate(&hit, hit.num_tokens()));
  EXPECT_EQ(hit.kv_state().shared_blocks_num(BlockType::KV), 2);
  pool.deallocate(&hit);
}

TEST_F(BlockManagerPoolTest, PdDecodeCannotPublishUnconfirmedRemoteKv) {
  BlockManagerPool::Options options;
  options.num_blocks(16)
      .block_size(4)
      .enable_prefix_cache(true)
      .enable_disagg_pd(true)
      .instance_is_decode(true)
      .event_journal(std::make_shared<KVCacheEventJournal>());
  EXPECT_DEATH(BlockManagerPool(options, /*dp_size=*/1),
               "KV events in disaggregated PD require a PREFILL instance");
}

TEST_F(BlockManagerPoolTest,
       LinearStateBlockManagerMatchesOnlyCheckpointHashes) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.num_embedding_blocks(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(allocate_kv_prefix_for_test(pool, cached_seq));
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
  ASSERT_TRUE(allocate_next_linear_chunk(pool, miss_seq));
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
  ASSERT_TRUE(allocate_next_linear_chunk(pool, hit_seq));
  EXPECT_EQ(hit_seq.kv_state().shared_blocks_num(BlockType::KV), 2u);
  pool.deallocate_without_cache(&hit_seq);
}

// An exact prompt match must leave at least one token for the current forward.
// The matched tail boundary (h2 here) is checkpointed, but reusing it would pop
// back to the previous, uncheckpointed boundary (h1) and lose restorable state,
// so prefix reuse must be 0 instead.
TEST_F(BlockManagerPoolTest, ExactPromptCannotReuseUncheckpointedTailBoundary) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  BlockManagerPool::Options options;
  options.num_blocks(32).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.num_embedding_blocks(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(allocate_kv_prefix_for_test(pool, cached_seq));
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
  ASSERT_TRUE(allocate_next_linear_chunk(pool, exact_seq));
  EXPECT_EQ(exact_seq.kv_state().shared_blocks_num(BlockType::KV), 0u);
  pool.deallocate_without_cache(&exact_seq);
}

// One token past the checkpoint boundary leaves work for the current forward,
// so the checkpointed boundary becomes reusable.
TEST_F(BlockManagerPoolTest, PromptPastCheckpointReusesCheckpointBoundary) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  BlockManagerPool::Options options;
  options.num_blocks(32).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.num_embedding_blocks(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(allocate_kv_prefix_for_test(pool, cached_seq));
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
  ASSERT_TRUE(allocate_next_linear_chunk(pool, hit_seq));
  EXPECT_EQ(hit_seq.kv_state().shared_blocks_num(BlockType::KV), 2u);
  ASSERT_EQ(hit_seq.kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_TRUE(hit_seq.kv_state().blocks(BlockType::LINEAR)[0].is_valid());
  pool.deallocate_without_cache(&hit_seq);
}

TEST_F(BlockManagerPoolTest, LinearAllocationEvictsOnlyUnpinnedCheckpoint) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(make_linear_state_pool_options(4), 1);
  LinearStateBlockManager* leaf =
      BlockManagerPoolTestPeer::linear_leaf(pool, 0);
  ASSERT_NE(leaf, nullptr);
  Sequence sequence = make_sequence(0, {1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&sequence, 4));
  Batch batch(&sequence);
  sequence.kv_state().set_kv_cache_tokens_num(4);
  const int32_t source_id = sequence.get_linear_state_slot_id();
  const Block retained_old_read =
      sequence.kv_state().blocks(BlockType::LINEAR)[0];
  const LinearStatePrefixHash evictable_hash = make_prefix_hash(2);
  const LinearStatePrefixHash pinned_hash = make_prefix_hash(1);
  ASSERT_GE(insert_linear_state_checkpoint(leaf, evictable_hash), 1);
  ASSERT_GE(insert_linear_state_checkpoint(leaf, pinned_hash), 1);
  Block pinned =
      BlockManagerPoolTestPeer::match(leaf, XXH3Key(pinned_hash.data()));
  ASSERT_TRUE(pinned.is_valid());
  ASSERT_TRUE(pool.allocate(&sequence, 8));
  EXPECT_NE(sequence.get_linear_state_slot_id(), source_id);
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_EQ(sequence.kv_state().blocks(BlockType::LINEAR)[0].id(), source_id);
  EXPECT_TRUE(
      BlockManagerPoolTestPeer::contains(leaf, XXH3Key(pinned_hash.data())));
  EXPECT_FALSE(
      BlockManagerPoolTestPeer::contains(leaf, XXH3Key(evictable_hash.data())));
  sequence.update_linear_state_hashes(4);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(
      leaf, sequence.linear_state_hashes()[0]));
  pool.deallocate_without_cache(&sequence);
}

TEST_F(BlockManagerPoolTest,
       PrefixMountSurvivesOutputAllocationFailureAndRetry) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(make_linear_state_pool_options(3), 1);
  LinearStateBlockManager* leaf =
      BlockManagerPoolTestPeer::linear_leaf(pool, 0);
  ASSERT_NE(leaf, nullptr);
  Sequence cached = make_sequence(0, {1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(allocate_kv_prefix_for_test(pool, cached));
  cached.kv_state().set_kv_cache_tokens_num(8);
  pool.cache(&cached);
  cached.update_linear_state_hashes(4);
  const int32_t source_id =
      insert_linear_state_checkpoint(leaf, cached.linear_state_hashes()[1]);
  ASSERT_GE(source_id, 1);
  pool.deallocate_without_cache(&cached);
  std::vector<Block> occupied = leaf->allocate(1);
  ASSERT_EQ(occupied.size(), 1u);
  Sequence consumer = make_sequence(1, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  pool.allocate_shared(&consumer);
  ASSERT_EQ(consumer.kv_cache_tokens_num(), 8u);
  EXPECT_FALSE(pool.allocate(&consumer, 9));
  EXPECT_EQ(consumer.kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_EQ(consumer.get_linear_state_slot_id(), source_id);
  pool.allocate_shared(&consumer);
  EXPECT_EQ(consumer.get_linear_state_slot_id(), source_id);
  leaf->deallocate(occupied);
  occupied.clear();
  ASSERT_TRUE(pool.allocate(&consumer, 9));
  ASSERT_EQ(consumer.kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_EQ(consumer.kv_state().blocks(BlockType::LINEAR)[0].id(), source_id);
  pool.deallocate_without_cache(&consumer);
}

TEST_F(BlockManagerPoolTest, PrefixUsesOnlyExactLinearStateCheckpoint) {
  constexpr int32_t kCanonicalBlockSize = 256;
  constexpr int32_t kChunkStride = 128;
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(),
      kChunkStride);

  const auto check_prefix = [this, kCanonicalBlockSize, kChunkStride](
                                std::initializer_list<size_t> checkpoints,
                                size_t expected_tokens) {
    BlockManagerPool::Options options;
    options.num_blocks(8)
        .host_num_blocks(0)
        .block_size(kCanonicalBlockSize)
        .enable_prefix_cache(true);
    options.num_embedding_blocks(FLAGS_max_seqs_per_batch + 2)
        .enable_linear_state(true)
        .linear_state_num_slots(16);
    BlockManagerPool pool(options, /*dp_size=*/1);

    const std::vector<int32_t> prompt_tokens(513, 1);
    Sequence cached_seq = make_sequence(0, prompt_tokens);
    ASSERT_TRUE(allocate_kv_prefix_for_test(pool, cached_seq));
    cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
    pool.cache(&cached_seq);

    LinearStateBlockManager* prefix_cache =
        BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0);
    ASSERT_NE(prefix_cache, nullptr);
    int32_t expected_slot = -1;
    for (size_t checkpoint : checkpoints) {
      const LinearStatePrefixHash hash =
          compute_linear_state_prefix_hash_for_test(
              cached_seq.tokens(), kChunkStride, checkpoint);
      const int32_t slot = insert_linear_state_checkpoint(prefix_cache, hash);
      ASSERT_GE(slot, 1);
      if (checkpoint == expected_tokens) {
        expected_slot = slot;
      }
    }
    pool.deallocate_without_cache(&cached_seq);

    Sequence hit_seq = make_sequence(1, prompt_tokens);
    ASSERT_TRUE(allocate_next_linear_chunk(pool, hit_seq));
    EXPECT_EQ(hit_seq.kv_state().kv_cache_tokens_num(), expected_tokens);
    EXPECT_EQ(hit_seq.kv_state().shared_blocks_num(BlockType::KV),
              expected_tokens / kCanonicalBlockSize);
    if (expected_tokens == 0) {
      EXPECT_EQ(hit_seq.kv_state().num_blocks(BlockType::LINEAR), 1u);
    } else {
      const Slice<Block> blocks = hit_seq.kv_state().blocks(BlockType::LINEAR);
      ASSERT_EQ(blocks.size(), 2u);
      EXPECT_EQ(blocks.front().id(), expected_slot);
    }
    pool.deallocate_without_cache(&hit_seq);
  };

  check_prefix({384}, 0);
  check_prefix({256, 384}, 256);
  check_prefix({512}, 512);
}

TEST_F(BlockManagerPoolTest, SparseLinearStateCheckpointCannotExceedKVMatch) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  BlockManagerPool::Options options;
  options.num_blocks(48).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.num_embedding_blocks(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence short_cached_seq =
      make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(allocate_kv_prefix_for_test(pool, short_cached_seq));
  short_cached_seq.kv_state().set_kv_cache_tokens_num(
      short_cached_seq.num_tokens());
  pool.cache(&short_cached_seq);
  pool.deallocate_without_cache(&short_cached_seq);

  Sequence long_cached_seq =
      make_sequence(1,
                    /*prompt_tokens=*/{
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_TRUE(allocate_kv_prefix_for_test(pool, long_cached_seq));
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
  ASSERT_TRUE(allocate_next_linear_chunk(pool, hit_seq));
  EXPECT_EQ(hit_seq.kv_state().shared_blocks_num(BlockType::KV), 2u);
  pool.deallocate_without_cache(&hit_seq);
}

TEST_F(BlockManagerPoolTest, ExactPromptStopsAtEarlierCheckpoint) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  BlockManagerPool::Options options;
  options.num_blocks(32).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.num_embedding_blocks(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      make_sequence(0,
                    /*prompt_tokens=*/{
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_TRUE(allocate_kv_prefix_for_test(pool, cached_seq));
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
  ASSERT_TRUE(allocate_next_linear_chunk(pool, exact_seq));
  EXPECT_EQ(exact_seq.kv_state().shared_blocks_num(BlockType::KV), 2u);
  pool.deallocate_without_cache(&exact_seq);
}

// Validates the production hash builder Sequence::update_block_hashes():
// correct chain, idempotency, and invalidation after a token rewrite.
TEST_F(BlockManagerPoolTest, SequenceUpdateBlockHashes) {
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
TEST_F(BlockManagerPoolTest,
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
