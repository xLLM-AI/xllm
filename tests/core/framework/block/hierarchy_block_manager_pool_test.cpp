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

#include "hierarchy_block_manager_pool.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "block_manager_impl.h"
#include "common/global_flags.h"
#include "framework/block/block_manager_pool.h"
#include "framework/block/block_utils.h"
#include "framework/block/composite_block_manager.h"
#include "framework/block/sliding_window_block_manager.h"
#include "framework/config/scheduler_config.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "framework/request/stopping_checker.h"
#include "framework/sampling/sampling_params.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {

// Peer that reaches inside HierarchyBlockManagerPool for verification of host
// leaf construction. The pool is heavy to spin up; we exercise the plumbing
// with an Engine stub set to nullptr because the constructor only touches the
// engine during allocate / transfer paths, not at build time.
class HierarchyPoolTestPeer final {
 public:
  static const std::vector<CompositeBlockManager::LeafMap>& host_block_managers(
      const HierarchyBlockManagerPool& pool) {
    return pool.host_block_managers_;
  }

  static std::vector<CompositeBlockManager::LeafMap>&
  mutable_host_block_managers(HierarchyBlockManagerPool& pool) {
    return pool.host_block_managers_;
  }

  static std::vector<BlockTransferInfo> pending_load_infos(
      const HierarchyBlockManagerPool& pool) {
    std::vector<BlockTransferInfo> infos;
    for (const auto& per_dp : pool.load_block_transfer_infos_) {
      infos.insert(infos.end(), per_dp.begin(), per_dp.end());
    }
    return infos;
  }

  static CompositeBlockManager* device_composite(
      HierarchyBlockManagerPool& pool) {
    return static_cast<CompositeBlockManager*>(
        pool.block_managers_.front().get());
  }

  static void dispatch_pending_h2d(HierarchyBlockManagerPool& pool) {
    for (auto& per_dp : pool.load_block_transfer_infos_) {
      per_dp.clear();
    }
  }

  static void collect_offload_pairs(HierarchyBlockManagerPool& pool,
                                    Sequence* sequence) {
    pool.collect_offload_pairs(sequence,
                               /*dp_rank=*/0,
                               sequence->kv_state().kv_cache_tokens_num());
  }

  static size_t pending_offload_pair_count(
      const HierarchyBlockManagerPool& pool) {
    size_t count = 0;
    for (const auto& queue : pool.offload_block_pair_queues_) {
      count += queue.size_approx();
    }
    return count;
  }
};

namespace {

BlockManagerPool::Options make_flat_kv_options() {
  BlockManagerPool::Options opts;
  opts.num_blocks(64)
      .host_num_blocks(128)
      .block_size(128)
      .enable_prefix_cache(true)
      .enable_host_offload(true);
  return opts;
}

BlockManagerPool::Options make_typed_cache_options() {
  constexpr uint32_t kBaseBlockSize = 128;
  constexpr uint32_t kWindow = 128;
  const uint32_t swa_blocks_per_seq =
      static_cast<uint32_t>(get_swa_blocks_per_seq(kWindow, kBaseBlockSize));

  BlockManagerPool::Options opts;
  opts.num_blocks(4096)
      .block_size(kBaseBlockSize)
      .enable_prefix_cache(true)
      .enable_host_offload(true)
      .sliding_window_size(kWindow)
      .swa_blocks_per_seq(swa_blocks_per_seq)
      .max_tokens_per_batch(32768)
      .max_seqs_per_batch(4)
      // SlidingWindow + BlockManagerImpl (C4) + BlockManagerImpl (C128).
      // The 0/4/128 compress_ratios drive the sub-manager block sizes.
      .manager_types({1u, 0u, 0u})
      .compress_ratios({0u, 4u, 128u})
      .host_num_blocks_by_type(
          {{BlockType::SWA, 512}, {BlockType::C4, 128}, {BlockType::C128, 16}});
  return opts;
}

Sequence make_test_sequence(size_t index,
                            const std::vector<int32_t>& prompt_token_ids) {
  torch::Device device(Platform::type_torch(), 0);
  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(16);
  SequenceParams seq_params;
  seq_params.seq_capacity =
      std::max<size_t>(32768, prompt_token_ids.size() + 16);
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

void seed_host_prefix(BlockManager* leaf, const std::vector<int32_t>& tokens) {
  ASSERT_NE(leaf, nullptr);
  const size_t block_count = tokens.size() / leaf->block_size();
  std::vector<Block> blocks = leaf->allocate(block_count);
  ASSERT_EQ(blocks.size(), block_count);
  leaf->cache(tokens, blocks);
  leaf->deallocate(blocks);
  blocks.clear();
}

bool allocate_with_host_cache_budget(HierarchyBlockManagerPool* pool,
                                     Sequence* sequence,
                                     size_t num_tokens,
                                     size_t max_copy_units) {
  pool->allocate_shared(sequence);
  const HostCacheRestorePoint selected =
      pool->select_host_cache_restore(sequence, max_copy_units);
  pool->trim_host_cache(sequence, selected);
  return pool->allocate(sequence, num_tokens);
}

}  // namespace

// A flat KV layout still creates a single KV host leaf. num_total_blocks
// reports one less than the raw count because block id 0 is reserved as a
// sentinel by BlockManagerImpl.
TEST(HierarchyBlockManagerPoolTest, FlatKvHasSingleHostKvLeaf) {
  HierarchyBlockManagerPool pool(make_flat_kv_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  const auto& per_dp = HierarchyPoolTestPeer::host_block_managers(pool);
  ASSERT_EQ(per_dp.size(), 1u);
  const auto& per_type = per_dp.front();
  ASSERT_EQ(per_type.size(), 1u);
  ASSERT_TRUE(per_type.count(BlockType::KV) == 1);
  EXPECT_EQ(per_type.at(BlockType::KV).leaf->block_size(), 128);
  EXPECT_EQ(per_type.at(BlockType::KV).leaf->num_total_blocks(), 127u);
}

// A typed SWA/C4/C128 layout creates matching Host leaves with the expected
// block_size and count. num_total_blocks() reports one less than the raw
// count because block id 0 is reserved as a sentinel by BlockManagerImpl and
// SlidingWindowBlockManager (a subclass).
TEST(HierarchyBlockManagerPoolTest, TypedLayoutHasSwaC4C128HostLeaves) {
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  const auto& per_dp = HierarchyPoolTestPeer::host_block_managers(pool);
  ASSERT_EQ(per_dp.size(), 1u);
  const auto& per_type = per_dp.front();
  ASSERT_EQ(per_type.size(), 3u);

  // SWA: gap-tolerant leaf with the base block size.
  ASSERT_TRUE(per_type.count(BlockType::SWA) == 1);
  EXPECT_EQ(per_type.at(BlockType::SWA).leaf->block_size(), 128);
  EXPECT_EQ(per_type.at(BlockType::SWA).leaf->num_total_blocks(), 511u);

  // C4: block_size = base * 4.
  ASSERT_TRUE(per_type.count(BlockType::C4) == 1);
  EXPECT_EQ(per_type.at(BlockType::C4).leaf->block_size(), 128 * 4);
  EXPECT_EQ(per_type.at(BlockType::C4).leaf->num_total_blocks(), 127u);

  // C128: block_size = base * 128.
  ASSERT_TRUE(per_type.count(BlockType::C128) == 1);
  EXPECT_EQ(per_type.at(BlockType::C128).leaf->block_size(), 128 * 128);
  EXPECT_EQ(per_type.at(BlockType::C128).leaf->num_total_blocks(), 15u);
}

// Multi-DP: each DP rank owns its own host leaf triplet with fresh block-id
// spaces.
TEST(HierarchyBlockManagerPoolTest, TypedLayoutHasPerDpRankLeaves) {
  constexpr int32_t kDpSize = 2;
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/kDpSize);
  const auto& per_dp = HierarchyPoolTestPeer::host_block_managers(pool);
  ASSERT_EQ(per_dp.size(), static_cast<size_t>(kDpSize));
  for (const auto& per_type : per_dp) {
    EXPECT_EQ(per_type.size(), 3u);
    EXPECT_TRUE(per_type.count(BlockType::SWA) == 1);
    EXPECT_TRUE(per_type.count(BlockType::C4) == 1);
    EXPECT_TRUE(per_type.count(BlockType::C128) == 1);
  }
}

TEST(HierarchyBlockManagerPoolTest,
     DecodeTypedLayoutKeepsSwaHostLeafForOffloadOnly) {
  BlockManagerPool::Options options = make_typed_cache_options();
  options.instance_is_decode(true).enable_disagg_pd(true);
  HierarchyBlockManagerPool pool(options,
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);

  const auto& per_type =
      HierarchyPoolTestPeer::host_block_managers(pool).front();
  ASSERT_EQ(per_type.size(), 3u);
  ASSERT_TRUE(per_type.count(BlockType::SWA) == 1);
  ASSERT_TRUE(per_type.count(BlockType::C4) == 1);
  ASSERT_TRUE(per_type.count(BlockType::C128) == 1);

  // Decode never probes or restores Host SWA, but it still needs an SWA Host
  // destination so completed decode blocks can be offloaded. Compressed leaves
  // continue to participate in prefix matching.
  EXPECT_FALSE(per_type.at(BlockType::SWA).supports_prefix_cache);
  EXPECT_TRUE(per_type.at(BlockType::C4).supports_prefix_cache);
  EXPECT_TRUE(per_type.at(BlockType::C128).supports_prefix_cache);
}

TEST(HierarchyBlockManagerPoolTest, DecodeTypedLayoutProbesOnlyDeviceC4C128) {
  constexpr size_t kPromptTokens = 20001;
  BlockManagerPool::Options options = make_typed_cache_options();
  options.instance_is_decode(true).enable_disagg_pd(true);
  HierarchyBlockManagerPool pool(options,
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 83);

  CompositeBlockManager* device = HierarchyPoolTestPeer::device_composite(pool);
  auto& host = HierarchyPoolTestPeer::mutable_host_block_managers(pool).front();
  seed_host_prefix(device->leaf_entries().at(BlockType::C4).leaf.get(), tokens);
  seed_host_prefix(device->leaf_entries().at(BlockType::C128).leaf.get(),
                   tokens);
  seed_host_prefix(host.at(BlockType::SWA).leaf.get(), tokens);
  seed_host_prefix(host.at(BlockType::C4).leaf.get(), tokens);
  seed_host_prefix(host.at(BlockType::C128).leaf.get(), tokens);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  sequence.kv_state().set_kv_cache_tokens_num(kPromptTokens);
  ASSERT_EQ(sequence.stage(), SequenceStage::DECODE);
  pool.allocate_shared(&sequence);

  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::SWA), 0u);
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::C4), 32u);
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::C128), 1u);
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 16384u);
  EXPECT_FALSE(sequence.host_kv_state().has_any_blocks());
  EXPECT_FALSE(sequence.has_host_cache_match());
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());
}

TEST(HierarchyBlockManagerPoolTest,
     DecodeTypedLayoutOffloadsNonPrefixSwaAndCompressedLeaves) {
  constexpr size_t kPromptTokens = 20001;
  BlockManagerPool::Options options = make_typed_cache_options();
  options.instance_is_decode(true).enable_disagg_pd(true);
  HierarchyBlockManagerPool pool(options,
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 89);
  Sequence sequence = make_test_sequence(/*index=*/0, tokens);

  sequence.kv_state().set_kv_cache_tokens_num(kPromptTokens);
  ASSERT_TRUE(pool.allocate(&sequence, kPromptTokens));
  sequence.kv_state().set_kv_cache_tokens_num(kPromptTokens);
  pool.deallocate(&sequence);

  // Only completed blocks are offloaded: 156 SWA blocks, 39 C4 blocks, and
  // one C128 checkpoint. The partial SWA tail is not inserted or offloaded.
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 196u);
}

TEST(HierarchyBlockManagerPoolTest, AllocateSharedMountsMatchesWithoutH2d) {
  constexpr size_t kPromptTokens = 20001;
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 5);
  auto& host_leaves =
      HierarchyPoolTestPeer::mutable_host_block_managers(pool).front();
  seed_host_prefix(host_leaves.at(BlockType::SWA).leaf.get(), tokens);
  seed_host_prefix(host_leaves.at(BlockType::C4).leaf.get(), tokens);
  seed_host_prefix(host_leaves.at(BlockType::C128).leaf.get(), tokens);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  pool.allocate_shared(&sequence);

  EXPECT_EQ(sequence.kv_cache_tokens_num(), 16384u);
  EXPECT_EQ(sequence.host_cache_copy_units(), 1u);
  EXPECT_FALSE(sequence.kv_state().has_any_blocks());
  EXPECT_TRUE(sequence.host_kv_state().has_any_blocks());
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());
}

TEST(HierarchyBlockManagerPoolTest,
     AllocateSharedAdvertisesC128AlignedCopyUnits) {
  constexpr size_t kPromptTokens = 65537;
  BlockManagerPool::Options options = make_typed_cache_options();
  options.host_num_blocks_by_type(
      {{BlockType::SWA, 640}, {BlockType::C4, 160}, {BlockType::C128, 8}});
  HierarchyBlockManagerPool pool(options, /*engine=*/nullptr, /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 6);
  auto& host = HierarchyPoolTestPeer::mutable_host_block_managers(pool).front();
  seed_host_prefix(host.at(BlockType::SWA).leaf.get(), tokens);
  seed_host_prefix(host.at(BlockType::C4).leaf.get(), tokens);
  seed_host_prefix(host.at(BlockType::C128).leaf.get(), tokens);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  pool.allocate_shared(&sequence);

  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_FALSE(sequence.kv_state().has_any_blocks());
  EXPECT_EQ(sequence.host_kv_state().kv_cache_tokens_num(), 65536u);
  EXPECT_EQ(sequence.host_kv_state().shared_blocks_num(BlockType::SWA), 512u);
  EXPECT_EQ(sequence.host_kv_state().shared_blocks_num(BlockType::C4), 128u);
  EXPECT_EQ(sequence.host_kv_state().shared_blocks_num(BlockType::C128), 4u);
  EXPECT_EQ(sequence.kv_cache_tokens_num(), 65536u);
  EXPECT_EQ(sequence.host_cache_copy_units(), 4u);
  const HostCacheRestorePoint selected =
      pool.select_host_cache_restore(&sequence, /*max_copy_units=*/3);
  EXPECT_EQ(selected.restore_target_tokens, 49152u);
  EXPECT_EQ(selected.copy_units, 3u);
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());

  pool.trim_host_cache(&sequence, selected);
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_EQ(sequence.host_kv_state().kv_cache_tokens_num(), 49152u);
  EXPECT_EQ(sequence.kv_cache_tokens_num(), 49152u);
  EXPECT_EQ(sequence.host_kv_state().num_blocks(BlockType::SWA), 512u);
  EXPECT_EQ(sequence.host_kv_state().num_blocks(BlockType::C4), 128u);
  EXPECT_EQ(sequence.host_kv_state().num_blocks(BlockType::C128), 4u);
  EXPECT_EQ(sequence.host_kv_state().num_cached_blocks(BlockType::SWA), 384u);
  EXPECT_EQ(sequence.host_kv_state().num_cached_blocks(BlockType::C4), 96u);
  EXPECT_EQ(sequence.host_kv_state().num_cached_blocks(BlockType::C128), 3u);

  pool.deallocate(&sequence);
}

TEST(HierarchyBlockManagerPoolTest,
     SuccessfulAllocateBuildsTypedC128AlignedH2dPlan) {
  constexpr size_t kPromptTokens = 20001;
  constexpr size_t kSafeHitTokens = 16384;
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 7);

  auto& host_leaves =
      HierarchyPoolTestPeer::mutable_host_block_managers(pool).front();
  seed_host_prefix(host_leaves.at(BlockType::SWA).leaf.get(), tokens);
  seed_host_prefix(host_leaves.at(BlockType::C4).leaf.get(), tokens);
  seed_host_prefix(host_leaves.at(BlockType::C128).leaf.get(), tokens);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  ASSERT_TRUE(allocate_with_host_cache_budget(
      &pool,
      &sequence,
      /*num_tokens=*/kPromptTokens,
      /*max_copy_units=*/std::numeric_limits<size_t>::max()));
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), kSafeHitTokens);
  EXPECT_EQ(sequence.host_kv_state().kv_cache_tokens_num(), kSafeHitTokens);
  EXPECT_EQ(sequence.host_kv_state().shared_blocks_num(BlockType::SWA), 156u);
  EXPECT_EQ(sequence.host_kv_state().shared_blocks_num(BlockType::C4), 39u);
  EXPECT_EQ(sequence.host_kv_state().shared_blocks_num(BlockType::C128), 1u);

  const std::vector<BlockTransferInfo> infos =
      HierarchyPoolTestPeer::pending_load_infos(pool);
  size_t swa_count = 0;
  size_t c4_count = 0;
  size_t c128_count = 0;
  for (const BlockTransferInfo& info : infos) {
    EXPECT_EQ(info.transfer_type, TransferType::H2D);
    const Slice<Block> host_blocks =
        sequence.host_kv_state().blocks(info.block_type);
    const Slice<Block> hbm_blocks = sequence.kv_state().blocks(info.block_type);
    const auto host_it = std::find_if(
        host_blocks.begin(), host_blocks.end(), [&](const Block& b) {
          return b.id() == info.src_block_id;
        });
    const auto hbm_it =
        std::find_if(hbm_blocks.begin(), hbm_blocks.end(), [&](const Block& b) {
          return b.id() == info.dst_block_id;
        });
    ASSERT_NE(host_it, host_blocks.end());
    ASSERT_NE(hbm_it, hbm_blocks.end());
    const XXH3Key hbm_hash(hbm_it->get_immutable_hash_value());
    EXPECT_TRUE(XXH3Key(host_it->get_immutable_hash_value()) == hbm_hash);
    EXPECT_TRUE(XXH3Key(info.hash_key) == hbm_hash);
    switch (info.block_type) {
      case BlockType::SWA:
        ++swa_count;
        break;
      case BlockType::C4:
        ++c4_count;
        break;
      case BlockType::C128:
        ++c128_count;
        break;
      default:
        ADD_FAILURE() << "Unexpected H2D block type: "
                      << static_cast<int32_t>(info.block_type);
    }
  }
  EXPECT_EQ(swa_count, 1u);
  EXPECT_EQ(c4_count, 32u);
  EXPECT_EQ(c128_count, 1u);
}

TEST(HierarchyBlockManagerPoolTest,
     ComplementaryHostTailDoesNotAdvanceIncompleteHostState) {
  constexpr size_t kPromptTokens = 20001;
  constexpr size_t kSafeHitTokens = 16384;
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 17);

  CompositeBlockManager* device = HierarchyPoolTestPeer::device_composite(pool);
  auto& host = HierarchyPoolTestPeer::mutable_host_block_managers(pool).front();
  seed_host_prefix(device->leaf_entries().at(BlockType::C4).leaf.get(), tokens);
  seed_host_prefix(device->leaf_entries().at(BlockType::C128).leaf.get(),
                   tokens);
  seed_host_prefix(host.at(BlockType::SWA).leaf.get(), tokens);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  pool.allocate_shared(&sequence);
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_EQ(sequence.host_kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_EQ(sequence.kv_cache_tokens_num(), kSafeHitTokens);

  ASSERT_TRUE(allocate_with_host_cache_budget(
      &pool,
      &sequence,
      /*num_tokens=*/kPromptTokens,
      /*max_copy_units=*/std::numeric_limits<size_t>::max()));

  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), kSafeHitTokens);
  EXPECT_GE(sequence.kv_state().current_max_tokens_capacity(), kSafeHitTokens);
  EXPECT_EQ(sequence.host_kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_GE(sequence.host_kv_state().current_max_tokens_capacity(),
            kPromptTokens);
  const std::vector<BlockTransferInfo> infos =
      HierarchyPoolTestPeer::pending_load_infos(pool);
  ASSERT_FALSE(infos.empty());
  for (const BlockTransferInfo& info : infos) {
    EXPECT_EQ(info.block_type, BlockType::SWA);
    EXPECT_EQ(info.transfer_type, TransferType::H2D);
  }
}

TEST(HierarchyBlockManagerPoolTest, FlatKvRestoreBudgetRemainsBlockLinear) {
  constexpr size_t kPromptTokens = 1025;
  HierarchyBlockManagerPool pool(make_flat_kv_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 23);
  auto& host = HierarchyPoolTestPeer::mutable_host_block_managers(pool).front();
  seed_host_prefix(host.at(BlockType::KV).leaf.get(), tokens);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  pool.allocate_shared(&sequence);
  EXPECT_EQ(sequence.kv_cache_tokens_num(), 1024u);
  EXPECT_EQ(sequence.host_cache_copy_units(), 8u);

  ASSERT_TRUE(allocate_with_host_cache_budget(
      &pool, &sequence, /*num_tokens=*/kPromptTokens, /*max_copy_units=*/4));
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 512u);
  const std::vector<BlockTransferInfo> infos =
      HierarchyPoolTestPeer::pending_load_infos(pool);
  ASSERT_EQ(infos.size(), 4u);
  for (const BlockTransferInfo& info : infos) {
    EXPECT_EQ(info.transfer_type, TransferType::H2D);
    EXPECT_EQ(info.block_type, BlockType::KV);
  }

  HierarchyPoolTestPeer::dispatch_pending_h2d(pool);
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/kPromptTokens));
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());

  pool.deallocate(&sequence);
}

TEST(HierarchyBlockManagerPoolTest,
     ZeroCopyRestoreTargetDoesNotReprobeHostCache) {
  constexpr size_t kPromptTokens = 1025;
  HierarchyBlockManagerPool pool(make_flat_kv_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 67);
  auto& host = HierarchyPoolTestPeer::mutable_host_block_managers(pool).front();
  seed_host_prefix(host.at(BlockType::KV).leaf.get(), tokens);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  pool.allocate_shared(&sequence);
  const HostCacheRestorePoint selected =
      pool.select_host_cache_restore(&sequence, /*max_copy_units=*/0);
  ASSERT_EQ(selected.restore_target_tokens, 0u);
  ASSERT_EQ(selected.copy_units, 0u);

  pool.trim_host_cache(&sequence, selected);
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/1));
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());
  EXPECT_FALSE(sequence.has_host_cache_match());

  pool.deallocate(&sequence);
}

TEST(HierarchyBlockManagerPoolTest, FailedHbmAllocationDoesNotQueueH2d) {
  constexpr size_t kPromptTokens = 1025;
  BlockManagerPool::Options options = make_flat_kv_options();
  options.num_blocks(8);
  HierarchyBlockManagerPool pool(options, /*engine=*/nullptr, /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 29);
  auto& host = HierarchyPoolTestPeer::mutable_host_block_managers(pool).front();
  seed_host_prefix(host.at(BlockType::KV).leaf.get(), tokens);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  pool.allocate_shared(&sequence);
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());
  EXPECT_TRUE(sequence.host_kv_state().has_any_blocks());
  ASSERT_TRUE(sequence.has_host_cache_match());
  const std::vector<size_t> used_before = pool.num_used_blocks();

  EXPECT_FALSE(allocate_with_host_cache_budget(
      &pool, &sequence, /*num_tokens=*/kPromptTokens, /*max_copy_units=*/4));
  EXPECT_FALSE(sequence.kv_state().has_any_blocks());
  EXPECT_FALSE(sequence.host_kv_state().has_any_blocks());
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_EQ(sequence.host_kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());
  EXPECT_EQ(pool.num_used_blocks(), used_before);
}

TEST(HierarchyBlockManagerPoolTest, ExistingBlocksSkipHostCacheRematch) {
  constexpr size_t kPromptTokens = 4097;
  HierarchyBlockManagerPool pool(make_flat_kv_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 31);
  auto& host = HierarchyPoolTestPeer::mutable_host_block_managers(pool).front();
  seed_host_prefix(host.at(BlockType::KV).leaf.get(), tokens);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  pool.allocate_shared(&sequence);
  ASSERT_TRUE(allocate_with_host_cache_budget(
      &pool, &sequence, /*num_tokens=*/1024, /*max_copy_units=*/4));
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 512u);
  sequence.kv_state().incr_kv_cache_tokens_num_up_to(/*new_target=*/1024);
  HierarchyPoolTestPeer::dispatch_pending_h2d(pool);
  pool.cache(&sequence);

  pool.allocate_shared(&sequence);
  const HostCacheRestorePoint selected =
      pool.select_host_cache_restore(&sequence, /*max_copy_units=*/8);
  ASSERT_EQ(selected.restore_target_tokens, 1024u);
  ASSERT_EQ(selected.copy_units, 0u);

  pool.trim_host_cache(&sequence, selected);
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/2304));
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 1024u);
  EXPECT_GE(sequence.kv_state().current_max_tokens_capacity(), 2304u);
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());

  pool.deallocate(&sequence);
}

TEST(HierarchyBlockManagerPoolTest, ExistingHostBlocksSkipPrefixRematch) {
  constexpr size_t kPromptTokens = 1025;
  constexpr size_t kInitialCachedTokens = 128;
  HierarchyBlockManagerPool pool(make_flat_kv_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 43);
  auto& host = HierarchyPoolTestPeer::mutable_host_block_managers(pool).front();
  seed_host_prefix(host.at(BlockType::KV).leaf.get(),
                   std::vector<int32_t>(tokens.begin(),
                                        tokens.begin() + kInitialCachedTokens));

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  pool.allocate_shared(&sequence);
  ASSERT_FALSE(sequence.kv_state().has_any_blocks());
  ASSERT_TRUE(sequence.host_kv_state().has_any_blocks());
  ASSERT_EQ(sequence.kv_cache_tokens_num(), kInitialCachedTokens);

  seed_host_prefix(host.at(BlockType::KV).leaf.get(), tokens);
  pool.allocate_shared(&sequence);
  EXPECT_EQ(sequence.kv_cache_tokens_num(), kInitialCachedTokens);
  EXPECT_EQ(sequence.host_kv_state().num_blocks(BlockType::KV), 1u);

  pool.deallocate(&sequence);
}

TEST(HierarchyBlockManagerPoolTest, ExistingBlocksSkipHbmPrefixRematch) {
  constexpr size_t kPromptTokens = 2305;
  HierarchyBlockManagerPool pool(make_flat_kv_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 47);
  Sequence sequence = make_test_sequence(/*index=*/0, tokens);

  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/1024));
  sequence.kv_state().incr_kv_cache_tokens_num_up_to(/*new_target=*/768);

  CompositeBlockManager* device = HierarchyPoolTestPeer::device_composite(pool);
  seed_host_prefix(device->leaf_entries().at(BlockType::KV).leaf.get(),
                   std::vector<int32_t>(tokens.begin(), tokens.begin() + 1280));

  pool.allocate_shared(&sequence);
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 768u);
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/1536));
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 768u);
  EXPECT_GE(sequence.kv_state().current_max_tokens_capacity(), 1536u);

  pool.deallocate(&sequence);
}

TEST(HierarchyBlockManagerPoolTest, HbmAllocationFailureDoesNotQueueH2d) {
  constexpr size_t kPromptTokens = 20001;
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 11);
  auto& host_leaves =
      HierarchyPoolTestPeer::mutable_host_block_managers(pool).front();
  seed_host_prefix(host_leaves.at(BlockType::SWA).leaf.get(), tokens);
  seed_host_prefix(host_leaves.at(BlockType::C4).leaf.get(), tokens);
  seed_host_prefix(host_leaves.at(BlockType::C128).leaf.get(), tokens);

  CompositeBlockManager* device = HierarchyPoolTestPeer::device_composite(pool);
  std::vector<Block> exhausted_c128 =
      device->allocate_blocks(BlockType::C128, 31);
  ASSERT_EQ(exhausted_c128.size(), 31u);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  EXPECT_FALSE(allocate_with_host_cache_budget(
      &pool,
      &sequence,
      /*num_tokens=*/kPromptTokens,
      /*max_copy_units=*/std::numeric_limits<size_t>::max()));
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_FALSE(sequence.kv_state().has_any_blocks());
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());

  device->deallocate(exhausted_c128);
  exhausted_c128.clear();
}

TEST(HierarchyBlockManagerPoolTest,
     HostAllocationFailureDoesNotRejectHbmAllocation) {
  BlockManagerPool::Options options = make_flat_kv_options();
  // BlockManagerImpl reserves block id 0, leaving only one usable Host block.
  // HBM has enough capacity for both blocks requested below.
  options.host_num_blocks(2);
  HierarchyBlockManagerPool pool(options, /*engine=*/nullptr, /*dp_size=*/1);

  std::vector<int32_t> tokens(257, 71);
  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/256));

  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::KV), 2u);
  EXPECT_GE(sequence.kv_state().current_max_tokens_capacity(), 256u);
  EXPECT_FALSE(sequence.host_kv_state().has_any_blocks());
  EXPECT_FALSE(sequence.has_host_cache_match());
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 0u);

  sequence.kv_state().set_kv_cache_tokens_num(128);
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/256));
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 0u);

  pool.deallocate(&sequence);
}

TEST(HierarchyBlockManagerPoolTest,
     FlatKvChunkGrowthOffloadsCompletedBlocksIncrementally) {
  HierarchyBlockManagerPool pool(make_flat_kv_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(385, 73);
  Sequence sequence = make_test_sequence(/*index=*/0, tokens);

  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/128));
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 0u);

  sequence.kv_state().set_kv_cache_tokens_num(128);
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/256));
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 1u);
  EXPECT_TRUE(sequence.kv_state().blocks(BlockType::KV)[0].is_valid());
  EXPECT_FALSE(sequence.host_kv_state().blocks(BlockType::KV)[0].is_valid());

  sequence.kv_state().set_kv_cache_tokens_num(256);
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/384));
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 2u);
  EXPECT_FALSE(sequence.host_kv_state().blocks(BlockType::KV)[1].is_valid());

  pool.deallocate(&sequence);
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 2u);
}

TEST(HierarchyBlockManagerPoolTest,
     Dsv4ChunkGrowthOffloadsAllCompletedCacheGroups) {
  constexpr size_t kFirstChunkTokens = 16384;
  constexpr size_t kPromptTokens = 20001;
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 79);
  Sequence sequence = make_test_sequence(/*index=*/0, tokens);

  ASSERT_TRUE(pool.allocate(&sequence, kFirstChunkTokens));
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 0u);

  sequence.kv_state().set_kv_cache_tokens_num(kFirstChunkTokens);
  ASSERT_TRUE(pool.allocate(&sequence, kPromptTokens));
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 34u);

  const size_t swa_checkpoint = kFirstChunkTokens / 128 - 1;
  EXPECT_FALSE(sequence.host_kv_state()
                   .blocks(BlockType::SWA)[swa_checkpoint]
                   .is_valid());
  EXPECT_FALSE(sequence.host_kv_state().blocks(BlockType::C4)[31].is_valid());
  EXPECT_TRUE(sequence.host_kv_state().blocks(BlockType::C4)[32].is_valid());
  EXPECT_FALSE(sequence.host_kv_state().blocks(BlockType::C128)[0].is_valid());
  EXPECT_TRUE(sequence.host_kv_state().blocks(BlockType::C128)[1].is_valid());
}

TEST(HierarchyBlockManagerPoolTest,
     H2dRestoreIsPublishedToDevicePrefixCacheDuringAllocation) {
  constexpr size_t kPromptTokens = 20001;
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 13);
  auto& host_leaves =
      HierarchyPoolTestPeer::mutable_host_block_managers(pool).front();
  seed_host_prefix(host_leaves.at(BlockType::SWA).leaf.get(), tokens);
  seed_host_prefix(host_leaves.at(BlockType::C4).leaf.get(), tokens);
  seed_host_prefix(host_leaves.at(BlockType::C128).leaf.get(), tokens);

  Sequence restored = make_test_sequence(/*index=*/0, tokens);
  ASSERT_TRUE(allocate_with_host_cache_budget(
      &pool,
      &restored,
      /*num_tokens=*/kPromptTokens,
      /*max_copy_units=*/std::numeric_limits<size_t>::max()));
  ASSERT_FALSE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());
  HierarchyPoolTestPeer::dispatch_pending_h2d(pool);

  Sequence replay = make_test_sequence(/*index=*/1, tokens);
  ASSERT_TRUE(allocate_with_host_cache_budget(
      &pool,
      &replay,
      /*num_tokens=*/kPromptTokens,
      /*max_copy_units=*/std::numeric_limits<size_t>::max()));
  EXPECT_EQ(replay.kv_state().kv_cache_tokens_num(), 16384u);
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());
}

TEST(HierarchyBlockManagerPoolTest,
     H2dRestoreDescriptionsRemainOwnedByPoolUntilDispatch) {
  constexpr size_t kPromptTokens = 1025;
  HierarchyBlockManagerPool pool(make_flat_kv_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 37);
  auto& host = HierarchyPoolTestPeer::mutable_host_block_managers(pool).front();
  seed_host_prefix(host.at(BlockType::KV).leaf.get(), tokens);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  pool.allocate_shared(&sequence);
  ASSERT_TRUE(allocate_with_host_cache_budget(
      &pool, &sequence, /*num_tokens=*/kPromptTokens, /*max_copy_units=*/4));
  ASSERT_EQ(HierarchyPoolTestPeer::pending_load_infos(pool).size(), 4u);
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::KV), 4u);

  HierarchyPoolTestPeer::dispatch_pending_h2d(pool);
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::KV), 4u);
}

TEST(HierarchyBlockManagerPoolTest,
     SharedHostPrefixRestoresAndGrowsEveryInBatchSequence) {
  constexpr size_t kPromptTokens = 20017;
  constexpr size_t kStepTargetTokens = 17447;
  constexpr size_t kCopyUnits = 257;
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);

  std::vector<int32_t> first_tokens(kPromptTokens, 41);
  std::vector<int32_t> second_tokens = first_tokens;
  second_tokens.back() = 43;
  auto& host_leaves =
      HierarchyPoolTestPeer::mutable_host_block_managers(pool).front();
  seed_host_prefix(host_leaves.at(BlockType::SWA).leaf.get(), first_tokens);
  seed_host_prefix(host_leaves.at(BlockType::C4).leaf.get(), first_tokens);
  seed_host_prefix(host_leaves.at(BlockType::C128).leaf.get(), first_tokens);

  Sequence first = make_test_sequence(/*index=*/0, first_tokens);
  Sequence second = make_test_sequence(/*index=*/1, second_tokens);
  pool.allocate_shared(&first);
  pool.allocate_shared(&second);

  ASSERT_TRUE(allocate_with_host_cache_budget(
      &pool, &first, kStepTargetTokens, /*max_copy_units=*/kCopyUnits));
  pool.cache(&first, kStepTargetTokens);
  ASSERT_TRUE(allocate_with_host_cache_budget(
      &pool, &second, kStepTargetTokens, /*max_copy_units=*/kCopyUnits));

  EXPECT_GE(first.kv_state().current_max_tokens_capacity(), kStepTargetTokens);
  EXPECT_GE(second.kv_state().current_max_tokens_capacity(), kStepTargetTokens);

  pool.deallocate(&first);
  pool.deallocate(&second);
}

TEST(HierarchyBlockManagerPoolTest,
     IncompleteC4C128CombinationHasZeroCopyUnits) {
  constexpr size_t kPromptTokens = 32769;
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 53);
  CompositeBlockManager* device = HierarchyPoolTestPeer::device_composite(pool);
  auto& host = HierarchyPoolTestPeer::mutable_host_block_managers(pool).front();

  seed_host_prefix(
      device->leaf_entries().at(BlockType::SWA).leaf.get(),
      std::vector<int32_t>(tokens.begin(), tokens.begin() + 16384));
  seed_host_prefix(
      device->leaf_entries().at(BlockType::C4).leaf.get(),
      std::vector<int32_t>(tokens.begin(), tokens.begin() + 31 * 512));
  seed_host_prefix(
      device->leaf_entries().at(BlockType::C128).leaf.get(),
      std::vector<int32_t>(tokens.begin(), tokens.begin() + 16384));
  seed_host_prefix(
      host.at(BlockType::SWA).leaf.get(),
      std::vector<int32_t>(tokens.begin(), tokens.begin() + 16384));
  seed_host_prefix(
      host.at(BlockType::C4).leaf.get(),
      std::vector<int32_t>(tokens.begin(), tokens.begin() + 33 * 512));
  seed_host_prefix(
      host.at(BlockType::C128).leaf.get(),
      std::vector<int32_t>(tokens.begin(), tokens.begin() + 32768));

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  pool.allocate_shared(&sequence);

  EXPECT_EQ(sequence.host_kv_state().num_blocks(BlockType::C4), 33u);
  EXPECT_EQ(sequence.host_kv_state().num_blocks(BlockType::C128), 2u);
  EXPECT_EQ(sequence.kv_cache_tokens_num(), 16384u);
  EXPECT_EQ(sequence.host_cache_copy_units(), 0u);
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());
}

TEST(HierarchyBlockManagerPoolTest,
     CompleteC4C128CombinationCountsOneCopyUnit) {
  constexpr size_t kPromptTokens = 33793;
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 59);
  CompositeBlockManager* device = HierarchyPoolTestPeer::device_composite(pool);
  auto& host = HierarchyPoolTestPeer::mutable_host_block_managers(pool).front();

  seed_host_prefix(
      device->leaf_entries().at(BlockType::SWA).leaf.get(),
      std::vector<int32_t>(tokens.begin(), tokens.begin() + 16384));
  seed_host_prefix(
      device->leaf_entries().at(BlockType::C4).leaf.get(),
      std::vector<int32_t>(tokens.begin(), tokens.begin() + 31 * 512));
  seed_host_prefix(
      device->leaf_entries().at(BlockType::C128).leaf.get(),
      std::vector<int32_t>(tokens.begin(), tokens.begin() + 16384));
  seed_host_prefix(
      host.at(BlockType::SWA).leaf.get(),
      std::vector<int32_t>(tokens.begin(), tokens.begin() + 32768));
  seed_host_prefix(
      host.at(BlockType::C4).leaf.get(),
      std::vector<int32_t>(tokens.begin(), tokens.begin() + 65 * 512));
  seed_host_prefix(
      host.at(BlockType::C128).leaf.get(),
      std::vector<int32_t>(tokens.begin(), tokens.begin() + 32768));

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  pool.allocate_shared(&sequence);

  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_EQ(sequence.host_kv_state().kv_cache_tokens_num(), 32768u);
  // Cache publication cursors retain each state/type's own probe reach. The
  // longer Host matches must not be folded into the HBM cursors, and the
  // common restore boundary must not flatten the three block sizes.
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::SWA), 128u);
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::C4), 31u);
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::C128), 1u);
  EXPECT_EQ(sequence.host_kv_state().num_cached_blocks(BlockType::SWA), 256u);
  EXPECT_EQ(sequence.host_kv_state().num_cached_blocks(BlockType::C4), 65u);
  EXPECT_EQ(sequence.host_kv_state().num_cached_blocks(BlockType::C128), 2u);
  EXPECT_EQ(sequence.host_kv_state().num_blocks(BlockType::C4), 65u);
  EXPECT_EQ(sequence.host_kv_state().num_blocks(BlockType::C128), 2u);
  EXPECT_EQ(sequence.kv_cache_tokens_num(), 32768u);
  EXPECT_EQ(sequence.host_cache_copy_units(), 1u);
  const HostCacheRestorePoint zero_budget =
      pool.select_host_cache_restore(&sequence, /*max_copy_units=*/0);
  EXPECT_EQ(zero_budget.restore_target_tokens, 16384u);
  EXPECT_EQ(zero_budget.copy_units, 0u);
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());
}

TEST(HierarchyBlockManagerPoolTest,
     CopyBudgetTrimFallsBackPastMissingSwaWindow) {
  constexpr size_t kRestoreTokens = 65536;
  constexpr size_t kInitialBudgetBoundary = 49152;
  constexpr size_t kExpectedBoundary = 32768;
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kRestoreTokens + 1, 61);
  Sequence sequence = make_test_sequence(/*index=*/0, tokens);

  auto& host = HierarchyPoolTestPeer::mutable_host_block_managers(pool).front();
  BlockManager* swa_leaf = host.at(BlockType::SWA).leaf.get();
  const size_t block_size = swa_leaf->block_size();
  const size_t tail_blocks = swa_leaf->options().swa_blocks_per_seq();
  ASSERT_GT(tail_blocks, 0u);
  ASSERT_EQ(kRestoreTokens % block_size, 0u);
  ASSERT_EQ(kInitialBudgetBoundary % block_size, 0u);
  ASSERT_EQ(kExpectedBoundary % block_size, 0u);

  std::vector<Block> allocated = swa_leaf->allocate(tail_blocks * 2);
  ASSERT_EQ(allocated.size(), tail_blocks * 2);
  std::vector<Block> sparse_swa(kRestoreTokens / block_size);
  const size_t expected_end = kExpectedBoundary / block_size;
  const size_t restore_end = kRestoreTokens / block_size;
  for (size_t i = 0; i < tail_blocks; ++i) {
    sparse_swa[expected_end - tail_blocks + i] = std::move(allocated[i]);
    sparse_swa[restore_end - tail_blocks + i] =
        std::move(allocated[tail_blocks + i]);
  }
  sequence.host_kv_state().replace_composite_blocks(
      BlockType::SWA,
      std::move(sparse_swa),
      /*num_shared_blocks=*/restore_end,
      /*num_cached_blocks=*/restore_end);
  sequence.set_host_cache_match(kRestoreTokens, /*copy_units=*/4);

  const HostCacheRestorePoint selected =
      pool.select_host_cache_restore(&sequence, /*max_copy_units=*/3);
  EXPECT_EQ(selected.restore_target_tokens, kExpectedBoundary);
  EXPECT_EQ(selected.copy_units, 2u);

  pool.deallocate(&sequence);
}

TEST(HierarchyBlockManagerPoolTest, ExistingHostPrefixSkipsDuplicateD2h) {
  constexpr size_t kCachedTokens = 1024;
  HierarchyBlockManagerPool pool(make_flat_kv_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kCachedTokens + 1, 19);
  auto& host = HierarchyPoolTestPeer::mutable_host_block_managers(pool).front();
  seed_host_prefix(host.at(BlockType::KV).leaf.get(), tokens);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  CompositeBlockManager* device = HierarchyPoolTestPeer::device_composite(pool);
  ASSERT_TRUE(device->allocate_sequence(&sequence, kCachedTokens));
  sequence.kv_state().set_kv_cache_tokens_num(kCachedTokens);
  device->cache_for_sequence(&sequence);

  HierarchyPoolTestPeer::collect_offload_pairs(pool, &sequence);
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 0u);

  device->deallocate_for_sequence(&sequence);
  sequence.reset();
}

TEST(HierarchyBlockManagerPoolTest, TypedLayoutRejectsKvStoragePrefetch) {
  BlockManagerPool::Options options = make_typed_cache_options();
  options.enable_kvcache_store(true);
  EXPECT_DEATH(
      {
        HierarchyBlockManagerPool pool(options,
                                       /*engine=*/nullptr,
                                       /*dp_size=*/1);
      },
      "currently supports only a flat KV Host cache layout");
}

TEST(HierarchyBlockManagerPoolTest, RejectsLinearCacheLayout) {
  BlockManagerPool::Options options = make_flat_kv_options();
  options.enable_linear_state(true).linear_state_num_slots(64);

  const int32_t original_chunk_stride =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() = 128;
  EXPECT_DEATH(
      {
        HierarchyBlockManagerPool pool(options,
                                       /*engine=*/nullptr,
                                       /*dp_size=*/1);
      },
      "supports only FLAT_KV and SWA_COMPRESSED cache layouts");
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
      original_chunk_stride;
}

}  // namespace xllm
