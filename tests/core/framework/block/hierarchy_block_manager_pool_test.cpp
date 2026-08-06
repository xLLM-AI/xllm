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
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <thread>
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

  static bool should_probe_prefix_cache(const HierarchyBlockManagerPool& pool,
                                        Sequence* sequence) {
    return pool.should_probe_prefix_cache(sequence);
  }
};

namespace {

class FakePrefetchEngine final : public Engine {
 public:
  explicit FakePrefetchEngine(size_t worker_count)
      : worker_count_(worker_count) {}

  ForwardOutput step(std::vector<Batch>& /*batch*/) override { return {}; }

  void update_last_step_result(std::vector<Batch>& /*batch*/) override {}

  std::vector<int64_t> get_active_activation_memory() const override {
    return {0};
  }

  std::shared_ptr<PrefetchResult> prefetch_from_storage(
      uint32_t dp_rank,
      const std::vector<BlockTransferInfo>& block_transfer_info) override {
    dp_rank_ = dp_rank;
    transfer_infos_ = block_transfer_info;
    result_ =
        std::make_shared<PrefetchResult>(worker_count_, transfer_infos_.size());
    return result_;
  }

  uint32_t dp_rank() const { return dp_rank_; }

  const std::vector<BlockTransferInfo>& transfer_infos() const {
    return transfer_infos_;
  }

  const std::shared_ptr<PrefetchResult>& result() const { return result_; }

 private:
  size_t worker_count_ = 0;
  uint32_t dp_rank_ = 0;
  std::vector<BlockTransferInfo> transfer_infos_;
  std::shared_ptr<PrefetchResult> result_;
};

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

std::shared_ptr<Request> make_test_request(
    const std::vector<int32_t>& prompt_token_ids) {
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(16);
  stopping_checker.set_max_context_len(prompt_token_ids.size() + 16);
  stopping_checker.set_ignore_eos(true);

  RequestState request_state("test",
                             prompt_token_ids,
                             sampling_param,
                             scheduler_param,
                             stopping_checker,
                             prompt_token_ids.size() + 16,
                             /*n=*/1,
                             /*best_of=*/1,
                             /*logprobs=*/false,
                             /*stream=*/false,
                             /*echo=*/false,
                             /*skip_special_tokens=*/true,
                             /*enable_schedule_overlap=*/false,
                             /*output_func=*/nullptr,
                             /*outputs_func=*/nullptr);
  return std::make_shared<Request>(
      "request", "x-request", "time", request_state, "service-request");
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

TEST(HierarchyBlockManagerPoolTest,
     PartialDsv4PrefillDoesNotReprobeAnUnmatchedHostTier) {
  constexpr size_t kPromptTokens = 20001;
  constexpr size_t kSharedTokens = 16384;
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 41);
  CompositeBlockManager* device = HierarchyPoolTestPeer::device_composite(pool);
  Sequence sequence = make_test_sequence(/*index=*/0, tokens);

  // Model an HBM prefix retained after a failed allocation, while the released
  // Host tier still needs a fresh probe. Every typed leaf must cover the same
  // C128-aligned prefix for the shared-token cursor to be meaningful.
  auto& device_leaves = device->leaf_entries();
  sequence.kv_state().mount_composite_shared(
      BlockType::SWA,
      device_leaves.at(BlockType::SWA).leaf->allocate(kSharedTokens / 128));
  sequence.kv_state().mount_composite_shared(
      BlockType::C4,
      device_leaves.at(BlockType::C4).leaf->allocate(kSharedTokens / 512));
  sequence.kv_state().mount_composite_shared(
      BlockType::C128,
      device_leaves.at(BlockType::C128).leaf->allocate(kSharedTokens / 16384));
  sequence.kv_state().set_kv_cache_tokens_num(kSharedTokens);
  sequence.kv_state().set_prefix_cache_matched();
  ASSERT_EQ(sequence.kv_state().kv_cache_tokens_num(), kSharedTokens);
  ASSERT_EQ(sequence.kv_state().shared_tokens_num(), kSharedTokens);

  sequence.host_kv_state().set_prefix_cache_matched(false);
  EXPECT_TRUE(
      HierarchyPoolTestPeer::should_probe_prefix_cache(pool, &sequence));

  sequence.kv_state().set_kv_cache_tokens_num(kSharedTokens + 1);
  EXPECT_FALSE(
      HierarchyPoolTestPeer::should_probe_prefix_cache(pool, &sequence));

  pool.deallocate(&sequence);
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

TEST(PrefetchResultTest, MergesBatchedTpWorkerBitmaps) {
  PrefetchResult result(/*worker_count=*/2, /*block_count=*/5);
  EXPECT_TRUE(result.set_batch_result(
      /*worker_index=*/0, /*offset=*/0, {1, 1}));
  EXPECT_TRUE(result.set_batch_result(
      /*worker_index=*/0, /*offset=*/2, {1, 0, 1}));
  EXPECT_TRUE(result.set_batch_result(
      /*worker_index=*/1, /*offset=*/0, {1, 0, 1, 1, 1}));

  result.mark_worker_completed(/*worker_index=*/0, /*worker_ok=*/true);
  EXPECT_FALSE(result.completed());
  result.mark_worker_completed(/*worker_index=*/1, /*worker_ok=*/true);
  ASSERT_TRUE(result.completed());
  EXPECT_EQ(result.merged_hits(), (std::vector<uint8_t>{1, 0, 1, 0, 1}));
}

TEST(PrefetchResultTest, StopRequestIsIdempotent) {
  PrefetchResult result(
      /*worker_count=*/2, /*block_count=*/5, /*batch_size=*/2);
  EXPECT_EQ(result.batch_size(), 2u);
  EXPECT_FALSE(result.stop_requested());
  EXPECT_TRUE(result.request_stop());
  EXPECT_TRUE(result.stop_requested());
  EXPECT_FALSE(result.request_stop());
}

TEST(PrefetchResultTest, CarriesStreamIdleTimeout) {
  PrefetchResult default_timeout(/*worker_count=*/2, /*block_count=*/5);
  EXPECT_EQ(default_timeout.stream_idle_timeout_ms(), -1);

  PrefetchResult configured_timeout(/*worker_count=*/2,
                                    /*block_count=*/5,
                                    /*batch_size=*/8,
                                    /*stream_idle_timeout_ms=*/30000);
  EXPECT_EQ(configured_timeout.stream_idle_timeout_ms(), 30000);
}

TEST(HierarchyBlockManagerPoolTest,
     StoragePrefetchTimeoutWaitsForInFlightBatch) {
  BlockManagerPool::Options options = make_flat_kv_options();
  options.enable_kvcache_store(true);
  FakePrefetchEngine engine(/*worker_count=*/2);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  std::vector<int32_t> tokens(1025, 83);
  std::shared_ptr<Request> request = make_test_request(tokens);
  Sequence* sequence = request->sequences().front().get();

  pool.prefetch_from_storage(request);
  ASSERT_NE(engine.result(), nullptr);
  ASSERT_EQ(engine.transfer_infos().size(), 8u);
  ASSERT_TRUE(engine.result()->set_batch_result(
      /*worker_index=*/0, /*offset=*/0, {1, 1}));
  ASSERT_TRUE(engine.result()->set_batch_result(
      /*worker_index=*/1, /*offset=*/0, {1, 1}));

  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  EXPECT_FALSE(pool.update_prefetch_result(request, /*timeout=*/1));
  EXPECT_TRUE(engine.result()->stop_requested());
  EXPECT_FALSE(sequence->host_kv_state().has_any_blocks());

  engine.result()->mark_worker_completed(/*worker_index=*/0,
                                         /*worker_ok=*/true);
  EXPECT_FALSE(pool.update_prefetch_result(request, /*timeout=*/1));
  engine.result()->mark_worker_completed(/*worker_index=*/1,
                                         /*worker_ok=*/true);
  EXPECT_TRUE(pool.update_prefetch_result(request, /*timeout=*/1));
  EXPECT_EQ(sequence->host_kv_state().num_blocks(BlockType::KV), 2u);
  EXPECT_TRUE(sequence->host_kv_state().prefix_cache_matched());
  pool.deallocate(sequence);
}

TEST(HierarchyBlockManagerPoolTest,
     StoragePrefetchPublishesOnlyAfterAllTpWorkersComplete) {
  BlockManagerPool::Options options = make_flat_kv_options();
  options.enable_kvcache_store(true);
  FakePrefetchEngine engine(/*worker_count=*/2);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  std::vector<int32_t> tokens(1025, 73);
  std::shared_ptr<Request> request = make_test_request(tokens);
  Sequence* sequence = request->sequences().front().get();

  pool.prefetch_from_storage(request);
  ASSERT_EQ(engine.dp_rank(), 0u);
  ASSERT_EQ(engine.transfer_infos().size(), 8u);
  ASSERT_NE(engine.result(), nullptr);
  EXPECT_FALSE(sequence->kv_state().has_any_blocks());
  EXPECT_FALSE(sequence->host_kv_state().has_any_blocks());
  EXPECT_FALSE(pool.update_prefetch_result(request, /*timeout=*/0));

  const std::vector<uint8_t> hits(engine.transfer_infos().size(), 1);
  ASSERT_TRUE(engine.result()->set_batch_result(
      /*worker_index=*/0, /*offset=*/0, hits));
  engine.result()->mark_worker_completed(/*worker_index=*/0,
                                         /*worker_ok=*/true);
  EXPECT_FALSE(pool.update_prefetch_result(request, /*timeout=*/0));
  ASSERT_TRUE(engine.result()->set_batch_result(
      /*worker_index=*/1, /*offset=*/0, hits));
  engine.result()->mark_worker_completed(/*worker_index=*/1,
                                         /*worker_ok=*/true);
  EXPECT_TRUE(pool.update_prefetch_result(request, /*timeout=*/0));

  // Prefetch publishes the completed Host probe directly. HBM remains
  // unmatched until the normal shared-allocation step.
  EXPECT_FALSE(sequence->kv_state().has_any_blocks());
  EXPECT_TRUE(sequence->host_kv_state().has_any_blocks());
  EXPECT_FALSE(sequence->kv_state().prefix_cache_matched());
  EXPECT_TRUE(sequence->host_kv_state().prefix_cache_matched());
  pool.allocate_shared(sequence);
  EXPECT_FALSE(sequence->kv_state().has_any_blocks());
  EXPECT_EQ(sequence->host_kv_state().num_blocks(BlockType::KV), 8u);
  EXPECT_EQ(sequence->kv_cache_tokens_num(), 1024u);
  pool.deallocate(sequence);
}

TEST(HierarchyBlockManagerPoolTest,
     CancelledStoragePrefetchStopsAndReleasesAfterCompletion) {
  BlockManagerPool::Options options = make_flat_kv_options();
  options.enable_kvcache_store(true);
  FakePrefetchEngine engine(/*worker_count=*/2);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  std::vector<int32_t> tokens(1025);
  for (size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = static_cast<int32_t>(i / 128);
  }
  std::shared_ptr<Request> request = make_test_request(tokens);
  Sequence* sequence = request->sequences().front().get();
  BlockManager* host_leaf =
      HierarchyPoolTestPeer::mutable_host_block_managers(pool)
          .front()
          .at(BlockType::KV)
          .leaf.get();
  const std::vector<int32_t> cached_tokens(tokens.begin(),
                                           tokens.begin() + 512);
  seed_host_prefix(host_leaf, cached_tokens);
  const size_t free_blocks_before = host_leaf->num_free_blocks();
  const size_t cached_blocks_before = host_leaf->num_blocks_in_prefix_cache();

  pool.prefetch_from_storage(request);
  ASSERT_NE(engine.result(), nullptr);
  ASSERT_LT(host_leaf->num_free_blocks(), free_blocks_before);
  request->set_cancel();

  EXPECT_FALSE(pool.update_prefetch_result(request, /*timeout=*/0));
  EXPECT_TRUE(engine.result()->stop_requested());
  EXPECT_FALSE(sequence->host_kv_state().has_any_blocks());
  engine.result()->mark_worker_completed(/*worker_index=*/0,
                                         /*worker_ok=*/true);
  EXPECT_FALSE(pool.update_prefetch_result(request, /*timeout=*/0));
  engine.result()->mark_worker_completed(/*worker_index=*/1,
                                         /*worker_ok=*/true);
  EXPECT_TRUE(pool.update_prefetch_result(request, /*timeout=*/0));

  EXPECT_EQ(host_leaf->num_free_blocks(), free_blocks_before);
  EXPECT_EQ(host_leaf->num_blocks_in_prefix_cache(), cached_blocks_before);
  EXPECT_FALSE(sequence->host_kv_state().has_any_blocks());
  EXPECT_FALSE(sequence->host_kv_state().prefix_cache_matched());
}

TEST(HierarchyBlockManagerPoolTest,
     TypedStoragePrefetchRetainsLongestPrefixWithinHostCapacity) {
  constexpr size_t kPromptTokens = 32769;
  BlockManagerPool::Options options = make_typed_cache_options();
  options.enable_kvcache_store(true).host_num_blocks_by_type(
      {{BlockType::SWA, 129}, {BlockType::C4, 33}, {BlockType::C128, 2}});
  FakePrefetchEngine engine(/*worker_count=*/2);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 79);
  std::shared_ptr<Request> request = make_test_request(tokens);
  Sequence* sequence = request->sequences().front().get();

  pool.prefetch_from_storage(request);
  ASSERT_NE(engine.result(), nullptr);
  EXPECT_EQ(engine.transfer_infos().size(), 128u + 32u + 1u);

  const std::vector<uint8_t> hits(engine.transfer_infos().size(), 1);
  ASSERT_TRUE(engine.result()->set_batch_result(
      /*worker_index=*/0, /*offset=*/0, hits));
  engine.result()->mark_worker_completed(/*worker_index=*/0,
                                         /*worker_ok=*/true);
  ASSERT_TRUE(engine.result()->set_batch_result(
      /*worker_index=*/1, /*offset=*/0, hits));
  engine.result()->mark_worker_completed(/*worker_index=*/1,
                                         /*worker_ok=*/true);
  ASSERT_TRUE(pool.update_prefetch_result(request, /*timeout=*/0));

  pool.allocate_shared(sequence);
  EXPECT_EQ(sequence->host_kv_state().num_blocks(BlockType::SWA), 128u);
  EXPECT_EQ(sequence->host_kv_state().num_blocks(BlockType::C4), 32u);
  EXPECT_EQ(sequence->host_kv_state().num_blocks(BlockType::C128), 1u);
  EXPECT_EQ(sequence->kv_cache_tokens_num(), 16384u);
  pool.deallocate(sequence);
}

TEST(HierarchyBlockManagerPoolTest,
     TypedStoragePrefetchKeepsSwaHitsAfterMiddleMiss) {
  constexpr size_t kPromptTokens = 32769;
  constexpr size_t kMissedSwaOrdinal = 100;
  BlockManagerPool::Options options = make_typed_cache_options();
  options.enable_kvcache_store(true);
  FakePrefetchEngine engine(/*worker_count=*/2);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 79);
  std::shared_ptr<Request> request = make_test_request(tokens);
  Sequence* sequence = request->sequences().front().get();

  pool.prefetch_from_storage(request);
  ASSERT_NE(engine.result(), nullptr);
  ASSERT_FALSE(engine.transfer_infos().empty());
  std::vector<uint8_t> first_worker_hits(engine.transfer_infos().size(), 1);
  size_t swa_ordinal = 0;
  size_t missed_result_index = engine.transfer_infos().size();
  for (size_t i = 0; i < engine.transfer_infos().size(); ++i) {
    if (engine.transfer_infos()[i].block_type != BlockType::SWA) {
      continue;
    }
    if (swa_ordinal == kMissedSwaOrdinal) {
      missed_result_index = i;
      first_worker_hits[i] = 0;
    }
    ++swa_ordinal;
  }
  ASSERT_EQ(swa_ordinal, 256u);
  ASSERT_LT(missed_result_index, engine.transfer_infos().size());

  const size_t split = engine.transfer_infos().size() / 2;
  ASSERT_TRUE(engine.result()->set_batch_result(
      /*worker_index=*/0,
      /*offset=*/0,
      std::vector<uint8_t>(first_worker_hits.begin(),
                           first_worker_hits.begin() + split)));
  ASSERT_TRUE(engine.result()->set_batch_result(
      /*worker_index=*/0,
      /*offset=*/split,
      std::vector<uint8_t>(first_worker_hits.begin() + split,
                           first_worker_hits.end())));
  engine.result()->mark_worker_completed(/*worker_index=*/0,
                                         /*worker_ok=*/true);

  const std::vector<uint8_t> second_worker_hits(engine.transfer_infos().size(),
                                                1);
  ASSERT_TRUE(engine.result()->set_batch_result(
      /*worker_index=*/1, /*offset=*/0, second_worker_hits));
  engine.result()->mark_worker_completed(/*worker_index=*/1,
                                         /*worker_ok=*/true);
  ASSERT_TRUE(pool.update_prefetch_result(request, /*timeout=*/0));
  ASSERT_TRUE(sequence->host_kv_state().has_any_blocks());
  ASSERT_TRUE(sequence->host_kv_state().prefix_cache_matched());
  ASSERT_FALSE(sequence->kv_state().prefix_cache_matched());

  pool.allocate_shared(sequence);
  const Slice<Block> swa_blocks =
      sequence->host_kv_state().blocks(BlockType::SWA);
  ASSERT_EQ(swa_blocks.size(), 256u);
  EXPECT_FALSE(swa_blocks[kMissedSwaOrdinal].is_valid());
  EXPECT_TRUE(swa_blocks[kMissedSwaOrdinal + 1].is_valid());
  EXPECT_TRUE(swa_blocks.back().is_valid());
  EXPECT_EQ(sequence->host_kv_state().num_blocks(BlockType::C4), 64u);
  EXPECT_EQ(sequence->host_kv_state().num_blocks(BlockType::C128), 2u);
  EXPECT_EQ(sequence->kv_cache_tokens_num(), 32768u);
  pool.deallocate(sequence);
}

TEST(HierarchyBlockManagerPoolTest, TypedLayoutSupportsStoragePrefetch) {
  BlockManagerPool::Options options = make_typed_cache_options();
  options.enable_kvcache_store(true);
  HierarchyBlockManagerPool pool(options,
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  EXPECT_EQ(HierarchyPoolTestPeer::host_block_managers(pool).front().size(),
            3u);
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
