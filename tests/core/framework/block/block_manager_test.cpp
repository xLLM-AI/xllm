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

#include <type_traits>
#include <utility>

#include "block_manager_impl.h"
#include "block_manager_pool.h"
#include "core/framework/config/scheduler_config.h"
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

template <typename T, typename = void>
struct HasEnableLinearStateOption : std::false_type {};

template <typename T>
struct HasEnableLinearStateOption<
    T,
    std::void_t<decltype(std::declval<T&>().enable_linear_state(true)),
                decltype(std::declval<const T&>().enable_linear_state())>>
    : std::true_type {};

template <typename T, typename = void>
struct HasSequenceSingleBlockApi : std::false_type {};

template <typename T>
struct HasSequenceSingleBlockApi<
    T,
    std::void_t<decltype(std::declval<const T&>().get_single_block_id())>>
    : std::true_type {};

template <typename OptionsT>
bool EnableLinearStateOrFail(OptionsT& options) {
  if constexpr (HasEnableLinearStateOption<OptionsT>::value) {
    options.enable_linear_state(true);
    return true;
  }
  ADD_FAILURE() << "Task 2 missing APIs: BlockManagerPool::Options "
                   "enable_linear_state";
  return false;
}

template <typename SeqT>
bool HasSingleBlockIdOrFail(const SeqT& seq) {
  if constexpr (HasSequenceSingleBlockApi<SeqT>::value) {
    return seq.get_single_block_id() >= 0;
  }
  ADD_FAILURE() << "Missing APIs: Sequence single-block handle";
  return false;
}

template <typename SeqT>
int32_t GetSingleBlockIdOrFail(const SeqT& seq) {
  if constexpr (HasSequenceSingleBlockApi<SeqT>::value) {
    return seq.get_single_block_id();
  }
  ADD_FAILURE() << "Missing APIs: Sequence single-block handle";
  return -1;
}

Sequence MakeSequence(size_t index, const std::vector<int32_t>& prompt_tokens) {
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
  ASSERT_TRUE(EnableLinearStateOrFail(options));
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seq = MakeSequence(0, /*prompt_tokens=*/{1, 2, 3});
  EXPECT_TRUE(pool.allocate(&seq));
  EXPECT_TRUE(HasSingleBlockIdOrFail(seq));
  // id 0 is the reserved padding slot, so a real assignment is strictly
  // positive.
  EXPECT_GT(GetSingleBlockIdOrFail(seq), 0);
}

TEST(BlockManagerPoolTest, DeallocateReleasesSingleBlockId) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 0);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  ASSERT_TRUE(EnableLinearStateOrFail(options));
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seq1 = MakeSequence(0, /*prompt_tokens=*/{1, 2, 3});
  ASSERT_TRUE(pool.allocate(&seq1));
  const int32_t id1 = GetSingleBlockIdOrFail(seq1);
  pool.deallocate(&seq1);
  EXPECT_FALSE(HasSingleBlockIdOrFail(seq1));

  Sequence seq2 = MakeSequence(1, /*prompt_tokens=*/{4, 5, 6});
  ASSERT_TRUE(pool.allocate(&seq2));
  EXPECT_EQ(GetSingleBlockIdOrFail(seq2), id1);
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
  ASSERT_TRUE(EnableLinearStateOrFail(options));
  BlockManagerPool pool(options, /*dp_size=*/1);

  std::vector<Sequence> sequences;
  sequences.reserve(4);
  for (size_t i = 0; i < 4; ++i) {
    sequences.emplace_back(MakeSequence(i, /*prompt_tokens=*/{1}));
    EXPECT_TRUE(pool.allocate(&sequences.back()));
    EXPECT_TRUE(HasSingleBlockIdOrFail(sequences.back()));
  }
}

TEST(BlockManagerPoolTest, TryAllocateKvFailureRollsBackSingleBlock) {
  // unified scheduler-side single-block pool has 2 ids.
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 0);

  BlockManagerPool::Options options;
  options.num_blocks(3).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  ASSERT_TRUE(EnableLinearStateOrFail(options));
  BlockManagerPool pool(options, /*dp_size=*/1);

  // This sequence needs far more KV blocks than available, forcing KV failure
  // after embedding and linear ids are allocated.
  std::vector<int32_t> huge_prompt(100, 1);
  Sequence fail_seq = MakeSequence(0, huge_prompt);
  EXPECT_FALSE(pool.try_allocate(&fail_seq));
  EXPECT_FALSE(HasSingleBlockIdOrFail(fail_seq));

  // The unified slot must have been rolled back, leaving enough capacity for
  // two new sequences to allocate.
  Sequence seq1 = MakeSequence(1, /*prompt_tokens=*/{1});
  Sequence seq2 = MakeSequence(2, /*prompt_tokens=*/{2});
  EXPECT_TRUE(pool.try_allocate(&seq1));
  EXPECT_TRUE(pool.try_allocate(&seq2));
  EXPECT_TRUE(HasSingleBlockIdOrFail(seq1));
  EXPECT_TRUE(HasSingleBlockIdOrFail(seq2));
}

TEST(BlockManagerPoolTest, AllocateAssignsSingleBlockWhenLinearStateDisabled) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seq = MakeSequence(0, /*prompt_tokens=*/{1, 2});
  EXPECT_TRUE(pool.allocate(&seq));
  EXPECT_TRUE(HasSingleBlockIdOrFail(seq));
}

TEST(BlockManagerPoolTest, SequenceCopyDoesNotReuseSingleBlockSlot) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  ASSERT_TRUE(EnableLinearStateOrFail(options));
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence src = MakeSequence(0, /*prompt_tokens=*/{1, 2, 3});
  ASSERT_TRUE(pool.allocate(&src));
  ASSERT_TRUE(HasSingleBlockIdOrFail(src));

  Sequence clone(src);
  EXPECT_FALSE(HasSingleBlockIdOrFail(clone));
  EXPECT_EQ(clone.get_single_block_id(), -1);

  ASSERT_TRUE(pool.allocate(&clone));
  EXPECT_TRUE(HasSingleBlockIdOrFail(clone));
  EXPECT_NE(GetSingleBlockIdOrFail(clone), GetSingleBlockIdOrFail(src));
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

  Sequence seq = MakeSequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7});
  ASSERT_TRUE(pool->allocate(&seq));

  // num_tokens (8) is larger than the 7 real tokens. cache() must clamp to the
  // sequence tokens, so only the single full block (tokens [0, 4)) is cached.
  EXPECT_NO_FATAL_FAILURE(pool->cache(&seq, /*num_tokens=*/8));
  EXPECT_EQ(pool->num_blocks_in_prefix_cache()[0], 1u);
}

// prefix_cache_match_length() reports the leading prefix-hit length in blocks
// and must leave the cache (size, LRU, ref counts -> utilization) untouched.
TEST(BlockManagerTest, PrefixCacheMatchLengthIsReadOnly) {
  BlockManager::Options options;
  options.num_blocks(16).block_size(2).enable_prefix_cache(true);
  // Leak intentionally: the prefix cache keeps the cached blocks referenced at
  // teardown, which would otherwise trip the free-list check in
  // ~BlockManagerImpl.
  auto* manager = new BlockManagerImpl(options);

  // Three full blocks: [1,2] [3,4] [5,6].
  const std::vector<int32_t> tokens = {1, 2, 3, 4, 5, 6};
  std::vector<Block> blocks = manager->allocate(/*num_blocks=*/3);
  ASSERT_EQ(blocks.size(), 3u);
  manager->cache(
      Slice<int32_t>(tokens), blocks, /*existed_shared_blocks_num=*/0);
  ASSERT_EQ(manager->num_blocks_in_prefix_cache(), 3u);

  const size_t cached_before = manager->num_blocks_in_prefix_cache();
  const double util_before = manager->kv_cache_utilization();

  // Full hit, and repeated probing is idempotent.
  const size_t matched1 =
      manager->prefix_cache_match_length(Slice<int32_t>(tokens));
  const size_t matched2 =
      manager->prefix_cache_match_length(Slice<int32_t>(tokens));
  EXPECT_EQ(matched1, 3u);
  EXPECT_EQ(matched2, matched1);

  // A shorter prefix matches only its leading full blocks.
  const std::vector<int32_t> short_tokens = {1, 2, 3, 4};
  EXPECT_EQ(manager->prefix_cache_match_length(Slice<int32_t>(short_tokens)),
            2u);

  // A diverging prefix matches only the shared leading block.
  const std::vector<int32_t> diverged = {1, 2, 9, 9};
  EXPECT_EQ(manager->prefix_cache_match_length(Slice<int32_t>(diverged)), 1u);

  // Probing mutated nothing.
  EXPECT_EQ(manager->num_blocks_in_prefix_cache(), cached_before);
  EXPECT_DOUBLE_EQ(manager->kv_cache_utilization(), util_before);
}

// With cache-aware routing on, a request sharing a cached prefix is routed to
// the rank holding that prefix even when another rank has more free blocks.
TEST(BlockManagerPoolTest, CacheAwareDpRoutingPrefersLongestPrefixRank) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(32)
      .host_num_blocks(0)
      .block_size(2)
      .enable_prefix_cache(true)
      .enable_cache_aware_dp(true)
      .cache_aware_match_threshold(0.5)
      // Disable the imbalance guard so this test isolates affinity behavior.
      .cache_aware_imbalance_threshold(1.0);
  // Leak: prefix cache holds block refs at teardown (see test above).
  auto* pool = new BlockManagerPool(options, /*dp_size=*/2);

  // First allocation ties on free blocks across ranks, so it lands on rank 0;
  // publishing its blocks seeds rank 0's prefix cache.
  Sequence seq_a = MakeSequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6});
  ASSERT_TRUE(pool->allocate(&seq_a));
  ASSERT_EQ(seq_a.dp_rank(), 0);
  pool->cache(&seq_a, /*num_tokens=*/6);
  ASSERT_GT(pool->num_blocks_in_prefix_cache()[0], 0u);

  const std::vector<size_t> cached_before = pool->num_blocks_in_prefix_cache();

  // Rank 1 has more free blocks now, but the shared prefix wins: route to 0.
  Sequence seq_b = MakeSequence(1, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6});
  pool->allocate_shared(&seq_b);
  EXPECT_EQ(seq_b.dp_rank(), 0);

  // Routing only probes the prefix cache; it must not change cache contents.
  EXPECT_EQ(pool->num_blocks_in_prefix_cache(), cached_before);

  // A request with no cached prefix falls back to the rank with more free
  // blocks (rank 1, since rank 0 still holds seq_a's blocks).
  Sequence seq_c = MakeSequence(2, /*prompt_tokens=*/{7, 8, 9, 10});
  pool->allocate_shared(&seq_c);
  EXPECT_EQ(seq_c.dp_rank(), 1);
}

// Cold start: a rank holding a tiny shared prefix (one block out of a long
// request) must NOT attract the request. The match-fraction threshold keeps
// routing load-balanced until a meaningful prefix accumulates.
TEST(BlockManagerPoolTest, CacheAwareDpRoutingIgnoresTinyPrefixOnColdStart) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(64)
      .host_num_blocks(0)
      .block_size(2)
      .enable_prefix_cache(true)
      .enable_cache_aware_dp(true)
      .cache_aware_match_threshold(0.5)
      // Isolate the match-fraction guard from the imbalance guard.
      .cache_aware_imbalance_threshold(1.0);
  auto* pool = new BlockManagerPool(options, /*dp_size=*/2);

  // Seed rank 0 with a single cached block [1,2].
  Sequence seq_a = MakeSequence(0, /*prompt_tokens=*/{1, 2});
  ASSERT_TRUE(pool->allocate(&seq_a));
  ASSERT_EQ(seq_a.dp_rank(), 0);
  pool->cache(&seq_a, /*num_tokens=*/2);
  ASSERT_GT(pool->num_blocks_in_prefix_cache()[0], 0u);

  // A long request shares only that one leading block (1/5 = 20% < 50%), so it
  // is routed by free blocks (rank 1) instead of being herded to rank 0.
  Sequence seq_b =
      MakeSequence(1, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  pool->allocate_shared(&seq_b);
  EXPECT_EQ(seq_b.dp_rank(), 1);
}

// A popular prefix that fully matches still yields to load balancing once the
// holding rank becomes too loaded relative to the emptiest rank.
TEST(BlockManagerPoolTest, CacheAwareDpRoutingRebalancesWhenOverloaded) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(32)
      .host_num_blocks(0)
      .block_size(2)
      .enable_prefix_cache(true)
      .enable_cache_aware_dp(true)
      .cache_aware_match_threshold(0.5)
      // Tight imbalance budget: rank 0 holding a few blocks already trips it.
      .cache_aware_imbalance_threshold(0.05);
  auto* pool = new BlockManagerPool(options, /*dp_size=*/2);

  Sequence seq_a = MakeSequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6});
  ASSERT_TRUE(pool->allocate(&seq_a));
  ASSERT_EQ(seq_a.dp_rank(), 0);
  pool->cache(&seq_a, /*num_tokens=*/6);

  // Same full prefix, but rank 0 is now too loaded vs rank 1, so the imbalance
  // guard routes the request to the least-loaded rank (rank 1).
  Sequence seq_b = MakeSequence(1, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6});
  pool->allocate_shared(&seq_b);
  EXPECT_EQ(seq_b.dp_rank(), 1);
}

// With cache-aware routing off, routing ignores the prefix and follows free
// blocks, so the same shared-prefix request lands on the emptier rank.
TEST(BlockManagerPoolTest, CacheAwareDpRoutingDisabledUsesMaxFreeRank) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(32)
      .host_num_blocks(0)
      .block_size(2)
      .enable_prefix_cache(true)
      .enable_cache_aware_dp(false);
  auto* pool = new BlockManagerPool(options, /*dp_size=*/2);

  Sequence seq_a = MakeSequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6});
  ASSERT_TRUE(pool->allocate(&seq_a));
  ASSERT_EQ(seq_a.dp_rank(), 0);
  pool->cache(&seq_a, /*num_tokens=*/6);

  // Same prefix, but routing ignores it and picks the rank with more free
  // blocks (rank 1).
  Sequence seq_b = MakeSequence(1, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6});
  pool->allocate_shared(&seq_b);
  EXPECT_EQ(seq_b.dp_rank(), 1);
}

}  // namespace xllm
