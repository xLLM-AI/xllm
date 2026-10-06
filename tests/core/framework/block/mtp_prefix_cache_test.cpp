/* Copyright 2026 The xLLM Authors.

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
#include <memory>
#include <utility>
#include <vector>

#include "core/distributed_runtime/engine.h"
#include "core/framework/batch/batch_group.h"
#include "core/framework/block/composite_block_manager.h"
#include "core/framework/block/hierarchy_block_manager_pool.h"
#include "core/framework/request/request.h"
#include "core/framework/request/sequence.h"

namespace xllm {
namespace {

Sequence make_mtp_sequence(const std::vector<int32_t>& tokens,
                           bool overlap = false) {
  static RequestSamplingParam sampling_param;
  static StoppingChecker stopping_checker;
  SequenceParams params;
  params.seq_capacity = tokens.size() + 128;
  params.enable_schedule_overlap = overlap;
  params.sampling_param = &sampling_param;
  params.stopping_checker = &stopping_checker;
  IncrementalDecoder decoder(/*prompt=*/"",
                             tokens.size(),
                             /*echo=*/false,
                             /*skip_special_tokens=*/true);
  return Sequence(/*index=*/0,
                  tokens,
                  /*input_embedding=*/torch::Tensor(),
                  /*mm_data=*/MMData(),
                  decoder,
                  params);
}

BlockManager::Options mtp_options() {
  BlockManager::Options options;
  options.num_blocks(32).block_size(16).enable_prefix_cache(true).hasher_type(
      BlockHasherType::MTP_TEXT);
  return options;
}

// Replace worker RPCs only; the Sequence, block pools, cache identities, and
// restore/offload decisions under test are real framework objects.
class CacheTransferEngine final : public Engine {
 public:
  std::vector<int64_t> get_active_activation_memory() const override {
    return {0};
  }
  void prefetch_from_storage(
      uint32_t /*dp_rank*/,
      std::shared_ptr<const StoragePrefetchRequest> request,
      PrefetchResult::StopPredicate stop_requested,
      PrefetchResult::DoneCallback done) override {
    request_ = std::move(request);
    queries_ = request_->transfer_infos;
    result_ = std::make_shared<PrefetchResult>(
        /*worker_count=*/1,
        request_->batch_end_unit_offsets,
        /*timeout_ms=*/-1,
        std::move(stop_requested),
        std::move(done));
  }
  void transfer_kv_blocks(
      uint32_t /*dp_rank*/,
      uint64_t /*batch_id*/,
      const std::vector<BlockTransferInfo>& infos) override {
    loads_ = infos;
  }
  std::vector<folly::SemiFuture<uint32_t>> transfer_kv_blocks(
      uint32_t /*dp_rank*/,
      const std::vector<BlockTransferInfo>& infos) override {
    offloads_ = infos;
    std::vector<folly::SemiFuture<uint32_t>> results;
    results.emplace_back(
        folly::makeSemiFuture(static_cast<uint32_t>(infos.size())));
    return results;
  }
  void finish_prefetch() {
    ASSERT_NE(result_, nullptr);
    for (size_t batch_index = 0; batch_index < request_->batch_count();
         ++batch_index) {
      const auto control = result_->record_batch_result(
          /*worker_index=*/0,
          static_cast<uint8_t>(request_->batch_unit_count(batch_index)));
      ASSERT_TRUE(control.has_value());
      if (*control == PrefetchControl::STOP) {
        break;
      }
    }
    result_->mark_worker_ended(
        /*worker_index=*/0, /*worker_ok=*/true, /*worker_quiescent=*/true);
  }
  const std::vector<BlockTransferInfo>& queries() const { return queries_; }
  const std::vector<BlockTransferInfo>& loads() const { return loads_; }
  const std::vector<BlockTransferInfo>& offloads() const { return offloads_; }

 private:
  std::vector<BlockTransferInfo> queries_;
  std::vector<BlockTransferInfo> loads_;
  std::vector<BlockTransferInfo> offloads_;
  std::shared_ptr<PrefetchResult> result_;
  std::shared_ptr<const StoragePrefetchRequest> request_;
};

BlockManagerPool::Options hierarchy_mtp_options() {
  BlockManagerPool::Options options;
  options.num_blocks(32)
      .host_num_blocks(64)
      .block_size(16)
      .enable_prefix_cache(true)
      .enable_host_offload(true)
      .enable_kvcache_store(true)
      .hasher_type(BlockHasherType::MTP_TEXT);
  return options;
}

BlockManagerPool::Options typed_hierarchy_options() {
  BlockManagerPool::Options options = hierarchy_mtp_options();
  options.num_blocks(4096)
      .manager_types({1u, 0u, 0u})
      .compress_ratios({0u, 4u, 128u})
      .swa_num_blocks(256)
      .swa_blocks_per_seq(2)
      .sliding_window_size(16)
      .max_tokens_per_batch(4096)
      .max_seqs_per_batch(1)
      .host_num_blocks_by_type(
          {{BlockType::SWA, 512}, {BlockType::C4, 128}, {BlockType::C128, 16}});
  return options;
}

std::shared_ptr<Request> make_mtp_request(const std::vector<int32_t>& tokens) {
  RequestSamplingParam sampling;
  SchedulerParam scheduler;
  StoppingChecker stopping;
  stopping.set_max_generated_tokens(16);
  stopping.set_max_context_len(tokens.size() + 16);
  stopping.set_ignore_eos(true);
  RequestState state("mtp-cache-test",
                     tokens,
                     sampling,
                     scheduler,
                     stopping,
                     tokens.size() + 16,
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
      "request", "x-request", "time", std::move(state), "mtp-cache-test");
}

}  // namespace

TEST(SequenceMtpPrefixCacheTest, BoundaryRewriteInvalidatesEveryStride) {
  std::vector<int32_t> tokens(65, 7);
  Sequence sequence = make_mtp_sequence(tokens);
  sequence.update_block_hashes(/*block_size=*/16, BlockHasherType::MTP_TEXT);
  ASSERT_EQ(sequence.block_hashes().size(), 4u);
  const XXH3Key first_hash = sequence.block_hashes()[0];
  sequence.update_block_hashes(/*block_size=*/32, BlockHasherType::MTP_TEXT);
  ASSERT_EQ(sequence.block_hashes().size(), 2u);

  sequence.update_token(/*index=*/32, Token(8));
  EXPECT_TRUE(sequence.block_hashes().empty());
  tokens[32] = 8;
  Sequence fresh = make_mtp_sequence(tokens);
  for (uint32_t stride : {16u, 32u}) {
    sequence.update_block_hashes(stride, BlockHasherType::MTP_TEXT);
    fresh.update_block_hashes(stride, BlockHasherType::MTP_TEXT);
    EXPECT_TRUE(sequence.block_hashes() == fresh.block_hashes());
  }
  sequence.update_block_hashes(/*block_size=*/16, BlockHasherType::MTP_TEXT);
  EXPECT_EQ(sequence.block_hashes()[0], first_hash);
}

TEST(SequenceMtpPrefixCacheTest, OverlapPlaceholdersNeverEnterHashChain) {
  Sequence sequence = make_mtp_sequence(std::vector<int32_t>(16, 7),
                                        /*overlap=*/true);
  sequence.kv_state().set_kv_cache_tokens_num(16);
  sequence.append_token(Token(-1));
  sequence.update_block_hashes(/*block_size=*/16, BlockHasherType::MTP_TEXT);
  EXPECT_TRUE(sequence.block_hashes().empty());
  sequence.update_last_step_token(Token(8), /*token_offset=*/0);
  sequence.update_block_hashes(/*block_size=*/16, BlockHasherType::MTP_TEXT);
  EXPECT_EQ(sequence.block_hashes().size(), 1u);
}

TEST(SequenceMtpPrefixCacheTest, CopyRetainsMtpInvalidationPolicy) {
  const std::vector<int32_t> tokens(33, 7);
  Sequence source = make_mtp_sequence(tokens);
  source.update_block_hashes(/*block_size=*/16, BlockHasherType::MTP_TEXT);
  Sequence copy(source);
  copy.update_token(/*index=*/16, Token(8));
  EXPECT_TRUE(copy.block_hashes().empty());
  EXPECT_EQ(source.block_hashes().size(), 2u);

  Sequence ordinary = make_mtp_sequence(tokens);
  ordinary.update_block_hashes(/*block_size=*/16, BlockHasherType::TEXT);
  const XXH3Key first = ordinary.block_hashes()[0];
  ordinary.update_token(/*index=*/16, Token(8));
  ASSERT_EQ(ordinary.block_hashes().size(), 1u);
  EXPECT_EQ(ordinary.block_hashes()[0], first);
}

TEST(CompositeMtpPrefixCacheTest, PublishesCompletedKvWithConfirmedLookahead) {
  CompositeBlockManager manager(build_composite_leaves(mtp_options()),
                                mtp_options());
  const std::vector<int32_t> tokens(33, 7);
  Sequence sequence = make_mtp_sequence(tokens);
  ASSERT_TRUE(manager.allocate_sequence(&sequence, tokens.size()));
  sequence.kv_state().set_kv_cache_tokens_num(15);
  manager.cache_full_blocks_for_sequence(&sequence);
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::KV), 0u);

  sequence.kv_state().set_kv_cache_tokens_num(16);
  manager.cache_full_blocks_for_sequence(&sequence);
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::KV), 1u);
  Sequence repeated = make_mtp_sequence(tokens);
  manager.allocate_shared_for_sequence(&repeated);
  EXPECT_EQ(repeated.kv_cache_tokens_num(), 16u);
  manager.deallocate_for_sequence(&repeated);
  manager.deallocate_for_sequence(&sequence);
}

TEST(CompositeMtpPrefixCacheTest, InBatchBudgetDoesNotPublishUnwrittenKv) {
  CompositeBlockManager manager(build_composite_leaves(mtp_options()),
                                mtp_options());
  const std::vector<int32_t> tokens(33, 7);
  Sequence sequence = make_mtp_sequence(tokens);
  ASSERT_TRUE(manager.allocate_sequence(&sequence, tokens.size()));
  manager.cache_for_sequence(&sequence, tokens.size());
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::KV), 0u);
  EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 0u);
  sequence.kv_state().set_kv_cache_tokens_num(16);
  manager.cache_for_sequence(&sequence, tokens.size());
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::KV), 1u);
  manager.deallocate_for_sequence(&sequence);
}

TEST(CompositeMtpPrefixCacheTest, RealAppendPublishesEverySafeBlock) {
  for (size_t boundary : {16u, 32u}) {
    for (int32_t path : {0, 1, 2}) {
      SCOPED_TRACE(boundary);
      SCOPED_TRACE(path);
      CompositeBlockManager manager(build_composite_leaves(mtp_options()),
                                    mtp_options());
      std::vector<int32_t> tokens(boundary, 7);
      Sequence sequence = make_mtp_sequence(tokens);
      ASSERT_TRUE(manager.allocate_sequence(&sequence, boundary));
      sequence.kv_state().set_kv_cache_tokens_num(boundary);
      sequence.append_token(Token(8));
      ASSERT_EQ(sequence.kv_cache_tokens_num(), boundary);
      ASSERT_EQ(sequence.hash_tokens(BlockHasherType::MTP_TEXT).size(),
                boundary + 1);
      auto publish = [&]() {
        switch (path) {
          case 0:
            manager.cache_full_blocks_for_sequence(&sequence);
            break;
          case 1:
            manager.cache_for_sequence(&sequence);
            break;
          case 2:
            manager.cache_for_sequence(&sequence, boundary);
            break;
        }
      };
      publish();
      EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::KV),
                boundary / 16);
      publish();
      EXPECT_EQ(manager.num_blocks_in_prefix_cache(), boundary / 16);

      tokens.reserve(boundary + 1);
      tokens.emplace_back(8);
      Sequence repeated = make_mtp_sequence(tokens);
      manager.allocate_shared_for_sequence(&repeated);
      EXPECT_EQ(repeated.kv_cache_tokens_num(), boundary);
      manager.deallocate_for_sequence(&repeated);
      tokens.back() = 9;
      Sequence forked = make_mtp_sequence(tokens);
      manager.allocate_shared_for_sequence(&forked);
      EXPECT_EQ(forked.kv_cache_tokens_num(), boundary - 16);
      manager.deallocate_for_sequence(&forked);
      manager.deallocate_for_sequence(&sequence);
    }
  }
}

TEST(CompositeMtpPrefixCacheTest, InBatchBoundsKvBudgetAndCapacitySeparately) {
  for (int32_t constraint : {0, 1, 2}) {
    SCOPED_TRACE(constraint);
    CompositeBlockManager manager(build_composite_leaves(mtp_options()),
                                  mtp_options());
    Sequence sequence = make_mtp_sequence(std::vector<int32_t>(33, 7));
    const size_t capacity = constraint == 2 ? 16 : 32;
    ASSERT_TRUE(manager.allocate_sequence(&sequence, capacity));
    sequence.kv_state().set_kv_cache_tokens_num(constraint == 0 ? 15 : 32);
    manager.cache_for_sequence(&sequence,
                               /*num_tokens=*/constraint == 1 ? 15 : 32);
    EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::KV),
              constraint == 2 ? 1u : 0u);
    manager.deallocate_for_sequence(&sequence);
  }
}

TEST(CompositeMtpPrefixCacheTest, PublicationPathsShareAnIdempotentCursor) {
  CompositeBlockManager manager(build_composite_leaves(mtp_options()),
                                mtp_options());
  const std::vector<int32_t> tokens(49, 7);
  Sequence sequence = make_mtp_sequence(tokens);
  ASSERT_TRUE(manager.allocate_sequence(&sequence, tokens.size()));
  sequence.kv_state().set_kv_cache_tokens_num(33);

  // A mid-step publication respects its limit even with more completed KV.
  manager.cache_for_sequence(&sequence, /*num_tokens=*/17);
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::KV), 1u);
  manager.cache_full_blocks_for_sequence(&sequence);
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::KV), 2u);

  // Neither a repeated publication nor the final flush publishes the tail.
  manager.cache_for_sequence(&sequence, tokens.size());
  manager.cache_for_sequence(&sequence);
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::KV), 2u);
  EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 2u);

  Sequence repeated = make_mtp_sequence(tokens);
  manager.allocate_shared_for_sequence(&repeated);
  EXPECT_EQ(repeated.kv_cache_tokens_num(), 32u);
  manager.deallocate_for_sequence(&repeated);
  manager.deallocate_for_sequence(&sequence);
}

TEST(CompositeMtpPrefixCacheTest, FinalFlushKeepsPublishCursorAtSafeBoundary) {
  CompositeBlockManager manager(build_composite_leaves(mtp_options()),
                                mtp_options());
  const std::vector<int32_t> tokens(32, 7);
  Sequence sequence = make_mtp_sequence(tokens);
  ASSERT_TRUE(manager.allocate_sequence(&sequence, tokens.size()));
  sequence.kv_state().set_kv_cache_tokens_num(32);
  manager.cache_for_sequence(&sequence);
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::KV), 1u);
  manager.deallocate_for_sequence(&sequence);
}

TEST(CompositeMtpPrefixCacheTest, NeverPublishesOverlapPlaceholderIdentity) {
  CompositeBlockManager manager(build_composite_leaves(mtp_options()),
                                mtp_options());
  Sequence sequence = make_mtp_sequence(std::vector<int32_t>(16, 7),
                                        /*overlap=*/true);
  ASSERT_TRUE(manager.allocate_sequence(&sequence, /*num_tokens=*/18));
  sequence.kv_state().set_kv_cache_tokens_num(16);
  sequence.append_token(Token(-1));
  manager.cache_for_sequence(&sequence);
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::KV), 0u);
  EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 0u);

  sequence.update_last_step_token(Token(8), /*token_offset=*/0);
  ASSERT_EQ(sequence.kv_cache_tokens_num(), 16u);
  ASSERT_EQ(sequence.hash_tokens(BlockHasherType::MTP_TEXT).size(), 17u);
  manager.cache_for_sequence(&sequence);
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::KV), 1u);
  manager.deallocate_for_sequence(&sequence);
}

TEST(HierarchyMtpPrefixCacheTest, RestoresEveryBlockProvenByStoreHash) {
  CacheTransferEngine engine;
  HierarchyBlockManagerPool pool(hierarchy_mtp_options(), &engine);
  const std::vector<int32_t> tokens(33, 7);
  auto request = make_mtp_request(tokens);
  Sequence* sequence = request->sequences().front().get();
  bool request_done = false;
  pool.prefetch_from_storage(request, [&](std::shared_ptr<Request> completed) {
    EXPECT_EQ(completed, request);
    request_done = true;
  });
  ASSERT_EQ(engine.queries().size(), 2u);
  EXPECT_FALSE(request_done);
  engine.finish_prefetch();
  pool.drain_prefetch_completions();
  ASSERT_TRUE(request_done);
  pool.allocate_shared(sequence);
  ASSERT_EQ(sequence->kv_cache_tokens_num(), 32u);
  ASSERT_TRUE(pool.allocate(sequence, tokens.size()));
  BatchGroup batches(1);
  pool.transfer_blocks(batches);
  EXPECT_EQ(engine.loads().size(), 2u);
  EXPECT_EQ(sequence->kv_state().num_cached_blocks(BlockType::KV), 2u);
  pool.deallocate(sequence);
  pool.transfer_blocks();
}

TEST(HierarchyMtpPrefixCacheTest, StoreDoesNotQueryWithoutNextToken) {
  CacheTransferEngine engine;
  HierarchyBlockManagerPool pool(hierarchy_mtp_options(), &engine);
  auto request = make_mtp_request(std::vector<int32_t>(16, 7));
  bool request_done = false;
  pool.prefetch_from_storage(request, [&](std::shared_ptr<Request> completed) {
    EXPECT_EQ(completed, request);
    request_done = true;
  });
  EXPECT_TRUE(engine.queries().empty());
  pool.drain_prefetch_completions();
  EXPECT_TRUE(request_done);
  pool.deallocate(request->sequences().front().get());
}

TEST(TypedMtpPrefixCacheTest, CompressedCheckpointRequiresNextToken) {
  BlockManager::Options options = mtp_options();
  options.num_blocks(4096)
      .manager_types({1u, 0u, 0u})
      .compress_ratios({0u, 4u, 128u})
      .swa_num_blocks(256)
      .swa_blocks_per_seq(2)
      .sliding_window_size(16)
      .max_tokens_per_batch(4096)
      .max_seqs_per_batch(1)
      .num_speculative_tokens(1);
  CompositeBlockManager manager(build_composite_leaves(options), options);
  const std::vector<int32_t> tokens(2048, 7);
  Sequence sequence = make_mtp_sequence(tokens);
  ASSERT_TRUE(manager.allocate_sequence(&sequence, tokens.size()));
  sequence.kv_state().set_kv_cache_tokens_num(2048);
  manager.cache_full_blocks_for_sequence(&sequence);
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::C128), 0u);
  sequence.append_token(Token(8));
  ASSERT_EQ(sequence.kv_cache_tokens_num(), 2048u);
  ASSERT_EQ(sequence.hash_tokens(BlockHasherType::MTP_TEXT).size(), 2049u);
  manager.cache_full_blocks_for_sequence(&sequence);
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::C128), 1u);
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::C4), 32u);
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::SWA), 128u);
  std::vector<int32_t> confirmed = tokens;
  confirmed.reserve(tokens.size() + 1);
  confirmed.emplace_back(8);
  Sequence repeated = make_mtp_sequence(confirmed);
  manager.allocate_shared_for_sequence(&repeated);
  EXPECT_EQ(repeated.kv_cache_tokens_num(), 2048u);
  manager.deallocate_for_sequence(&repeated);
  manager.deallocate_for_sequence(&sequence);
}

TEST(TypedMtpPrefixCacheTest, DecodeOffloadWaitsForCompletedLookahead) {
  CacheTransferEngine engine;
  BlockManagerPool::Options options = typed_hierarchy_options();
  options.instance_is_decode(true);
  HierarchyBlockManagerPool pool(options, &engine);
  Sequence sequence = make_mtp_sequence(std::vector<int32_t>(16, 7));
  sequence.kv_state().set_kv_cache_tokens_num(16);
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/16));
  sequence.kv_state().set_kv_cache_tokens_num(16);
  pool.deallocate(&sequence);
  pool.transfer_blocks();
  EXPECT_TRUE(engine.offloads().empty());
}

TEST(TypedMtpPrefixCacheTest, DecodeOffloadUsesConfirmedMtpIdentity) {
  CacheTransferEngine engine;
  BlockManagerPool::Options options = typed_hierarchy_options();
  options.instance_is_decode(true);
  HierarchyBlockManagerPool pool(options, &engine);
  Sequence sequence = make_mtp_sequence(std::vector<int32_t>(16, 7));
  sequence.kv_state().set_kv_cache_tokens_num(16);
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/16));
  sequence.kv_state().set_kv_cache_tokens_num(16);
  sequence.append_token(Token(8));
  ASSERT_EQ(sequence.kv_cache_tokens_num(), 16u);
  ASSERT_EQ(sequence.hash_tokens(BlockHasherType::MTP_TEXT).size(), 17u);
  sequence.update_block_hashes(/*block_size=*/16, BlockHasherType::MTP_TEXT);
  ASSERT_EQ(sequence.block_hashes().size(), 1u);
  const XXH3Key expected_hash = sequence.block_hashes()[0];
  pool.deallocate(&sequence);
  pool.transfer_blocks();
  ASSERT_EQ(engine.offloads().size(), 1u);
  EXPECT_EQ(engine.offloads()[0].block_type, BlockType::SWA);
  EXPECT_EQ(XXH3Key(engine.offloads()[0].hash_key), expected_hash);
}

TEST(TypedPrefixCacheTest, DecodeOffloadUsesContentIdentityWithoutDeviceStamp) {
  CacheTransferEngine engine;
  BlockManagerPool::Options options = typed_hierarchy_options();
  options.instance_is_decode(true).hasher_type(BlockHasherType::TEXT);
  HierarchyBlockManagerPool pool(options, &engine);
  const PrefixHash unstamped_hash{};
  XXH3Key previous_hash(unstamped_hash.data());
  for (int32_t token : {7, 8}) {
    Sequence sequence = make_mtp_sequence(std::vector<int32_t>(16, token));
    // Decode receives the completed prefix from prefill rather than probing
    // the grouped device prefix cache locally.
    sequence.kv_state().set_kv_cache_tokens_num(16);
    ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/16));
    sequence.kv_state().set_kv_cache_tokens_num(16);
    // An offload-only device block has no content stamp. Give it a known
    // unrelated value so copying it fails deterministically.
    sequence.kv_state()
        .mutable_blocks(BlockType::SWA)
        ->at(0)
        .set_hash_value(unstamped_hash.data());
    auto hasher =
        BlockHasher::create(BlockHasherType::TEXT, sequence.mm_data());
    XXH3Key expected_hash;
    hasher->compute(sequence.tokens(),
                    /*start_token_idx=*/0,
                    /*end_token_idx=*/16,
                    /*pre_hash_value=*/nullptr,
                    expected_hash);
    EXPECT_NE(expected_hash, previous_hash);
    pool.deallocate(&sequence);
    pool.transfer_blocks();
    ASSERT_EQ(engine.offloads().size(), 1u);
    EXPECT_EQ(engine.offloads()[0].block_type, BlockType::SWA);
    EXPECT_EQ(XXH3Key(engine.offloads()[0].hash_key), expected_hash);
    previous_hash = expected_hash;
  }
}

TEST(HierarchyMtpPrefixCacheTest,
     OffloadPreservesDeviceStampAndCompletedBoundary) {
  CacheTransferEngine engine;
  HierarchyBlockManagerPool pool(hierarchy_mtp_options(), &engine);
  Sequence sequence = make_mtp_sequence(std::vector<int32_t>(16, 7));
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/16));
  sequence.kv_state().set_kv_cache_tokens_num(16);
  sequence.append_token(Token(8));
  ASSERT_EQ(sequence.kv_cache_tokens_num(), 16u);
  ASSERT_EQ(sequence.hash_tokens(BlockHasherType::MTP_TEXT).size(), 17u);
  pool.cache(&sequence);
  ASSERT_EQ(sequence.kv_state().num_cached_blocks(BlockType::KV), 1u);
  const XXH3Key device_hash(
      sequence.kv_state().blocks(BlockType::KV)[0].get_immutable_hash_value());
  sequence.update_block_hashes(/*block_size=*/16, BlockHasherType::MTP_TEXT);
  EXPECT_EQ(device_hash, sequence.block_hashes()[0]);
  pool.deallocate(&sequence);
  pool.transfer_blocks();
  ASSERT_EQ(engine.offloads().size(), 1u);
  EXPECT_EQ(engine.offloads()[0].block_type, BlockType::KV);
  EXPECT_EQ(XXH3Key(engine.offloads()[0].hash_key), device_hash);
}

TEST(HierarchyMtpPrefixCacheTest, HostMatchRejectsDifferentNextToken) {
  CacheTransferEngine engine;
  HierarchyBlockManagerPool pool(hierarchy_mtp_options(), &engine);
  const std::vector<int32_t> tokens(33, 7);
  auto source = make_mtp_request(tokens);
  bool source_done = false;
  pool.prefetch_from_storage(source, [&](std::shared_ptr<Request> completed) {
    EXPECT_EQ(completed, source);
    source_done = true;
  });
  engine.finish_prefetch();
  pool.drain_prefetch_completions();
  ASSERT_TRUE(source_done);
  pool.deallocate(source->sequences().front().get());

  std::vector<int32_t> forked = tokens;
  forked[16] = 8;
  auto first_fork = make_mtp_request(forked);
  Sequence* first = first_fork->sequences().front().get();
  pool.allocate_shared(first);
  EXPECT_EQ(first->kv_cache_tokens_num(), 0u);
  pool.deallocate(first);

  forked = tokens;
  forked[32] = 8;
  auto second_fork = make_mtp_request(forked);
  Sequence* second = second_fork->sequences().front().get();
  pool.allocate_shared(second);
  EXPECT_EQ(second->kv_cache_tokens_num(), 16u);
  pool.deallocate(second);
}

TEST(TypedMtpPrefixCacheTest, StoreQueriesOnlyCompleteDependencyUnits) {
  CacheTransferEngine engine;
  HierarchyBlockManagerPool pool(typed_hierarchy_options(), &engine);
  auto exact = make_mtp_request(std::vector<int32_t>(2048, 7));
  bool exact_done = false;
  pool.prefetch_from_storage(exact, [&](std::shared_ptr<Request> completed) {
    EXPECT_EQ(completed, exact);
    exact_done = true;
  });
  EXPECT_TRUE(engine.queries().empty());
  pool.drain_prefetch_completions();
  EXPECT_TRUE(exact_done);
  pool.deallocate(exact->sequences().front().get());

  auto ready = make_mtp_request(std::vector<int32_t>(2049, 7));
  bool ready_done = false;
  pool.prefetch_from_storage(ready, [&](std::shared_ptr<Request> completed) {
    EXPECT_EQ(completed, ready);
    ready_done = true;
  });
  // One C128 checkpoint, 32 C4 blocks, and its two-block SWA window.
  ASSERT_EQ(engine.queries().size(), 35u);
  EXPECT_FALSE(ready_done);
  engine.finish_prefetch();
  pool.drain_prefetch_completions();
  ASSERT_TRUE(ready_done);
  Sequence* sequence = ready->sequences().front().get();
  pool.allocate_shared(sequence);
  EXPECT_EQ(sequence->kv_cache_tokens_num(), 2048u);
  pool.deallocate(sequence);
}

}  // namespace xllm
