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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "common/global_flags.h"
#include "core/common/prefill_policy.h"
#include "core/framework/batch/batch.h"
#include "core/framework/block/block_manager_pool.h"
#include "core/framework/block/kv_cache_manager_factory.h"
#include "core/framework/config/scheduler_config.h"
#include "core/kv_cache/block/block_manager_impl.h"
#include "core/kv_cache/prefix/block_hasher.h"
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

XXH3Key make_prefix_hash(uint8_t tag) {
  XXH3Key hash{};
  std::ranges::fill(hash.data, tag);
  return hash;
}

bool allocate_kv_prefix_for_test(BlockManagerPool& pool, Sequence& sequence) {
  int32_t dp_rank = -1;
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
  const size_t chunk_stride = pool.options().linear_state_chunk_stride();
  const size_t target = std::min(sequence.num_tokens(),
                                 sequence.kv_cache_tokens_num() + chunk_stride);
  return pool.allocate(&sequence, target);
}

class BlockManagerPoolTest : public ::testing::Test {
 protected:
  BlockManagerPoolTest() { sampling_param_.is_embeddings = false; }

  Sequence make_sequence(size_t index,
                         const std::vector<int32_t>& prompt_tokens) {
    SequenceParams params;
    params.seq_capacity = prompt_tokens.size() + 8;
    params.echo = false;
    params.skip_special_tokens = true;
    params.streaming = false;
    params.enable_schedule_overlap = false;
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
      .linear_state_chunk_stride(4)
      .linear_state_num_slots(linear_state_num_slots);
  return options;
}

ModelArgs make_shared_cache_model_args() {
  ModelArgs args;
  args.model_type("qwen3_next")
      .n_layers(4)
      .n_heads(1)
      .n_kv_heads(1)
      .head_dim(1)
      .full_attention_interval(2)
      .linear_num_key_heads(1)
      .linear_num_value_heads(1)
      .linear_key_head_dim(1)
      .linear_value_head_dim(1)
      .linear_conv_kernel_dim(2)
      .mamba_ssm_dtype("float32");
  return args;
}

KVCacheCapacity make_shared_cache_capacity() {
  // Two banks mirror 32-byte Full slices: global group = 64 bytes.
  // Logical IDs and physical slabs have independent capacities.
  KVCacheCapacity capacity;
  capacity.n_blocks(53)
      .block_size(8)
      .num_linear_state_blocks(53)
      .linear_conv_state_len(1)
      .linear_ssm_checkpoint_stride(1)
      .shared_slot_bytes(64)
      .shared_slab_count(12)
      .shared_min_slab_count(5)
      .shared_reserved_slab_count(2)
      .shared_linear_state_slots(4);
  return capacity;
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

TEST_F(BlockManagerPoolTest, GroupedFactoryBindsHistoryAndWindowGeometry) {
  ModelArgs args;
  args.model_type("deepseek_v4")
      .n_layers(4)
      .window_size(128)
      .max_seq_len(64)
      .compress_ratios({1, 4, 128, 4});
  KVCacheCapacity capacity;
  capacity.n_blocks(256).block_size(8).swa_count(32).c4_count(64).c128_count(2);
  BlockManagerPool::Options options;
  // Logical stride includes KV split; the SWA ring uses physical rows.
  options.num_blocks(256).block_size(16).model_id("grouped");
  auto result = KVCacheManagerFactory::create(
      capacity,
      args,
      /*world_size=*/1,
      torch::kBFloat16,
      options,
      PrefillPolicy(/*max_batch_tokens=*/512,
                    /*chunk_tokens=*/512,
                    /*enable_chunked_prefill=*/false,
                    /*enable_fair_budget=*/false),
      /*transfer_coordinator=*/nullptr);
  auto& pool = static_cast<BlockManagerPool&>(*result.manager);
  const auto& partitions = pool.options().grouped_partitions();
  ASSERT_EQ(partitions.size(), 3u);
  EXPECT_FALSE(pool.paged_partition().has_value());
  EXPECT_EQ(result.shape.key_cache_shape(), (std::vector<int64_t>{32, 64, 2}));
  EXPECT_EQ(partitions[0].window_size(), 64);
  EXPECT_EQ(partitions[0].token_stride(), 16);
  EXPECT_FALSE(partitions[0].participates_in_admission());
  EXPECT_EQ(pool.options().swa_blocks_per_seq(), 8u);
  EXPECT_EQ(partitions[1].token_stride(), 64);
  EXPECT_EQ(partitions[1].compression_ratio(), 4);
  EXPECT_EQ(partitions[2].token_stride(), 2048);
  EXPECT_EQ(partitions[2].compression_ratio(), 128);
  const auto* window = partitions[0].find_component(CacheComponentRole::KEY);
  const auto* history = partitions[1].find_component(CacheComponentRole::KEY);
  const auto* index = partitions[1].find_component(CacheComponentRole::INDEX);
  const auto* state =
      partitions[0].find_component(CacheComponentRole::COMPRESS_KV_STATE);
  ASSERT_NE(window, nullptr);
  ASSERT_NE(history, nullptr);
  ASSERT_NE(index, nullptr);
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(window->layers, (std::vector<int32_t>{0, 1, 2, 3}));
  EXPECT_EQ(history->layers, (std::vector<int32_t>{1, 3}));
  EXPECT_EQ(index->dtype, CacheScalarType::INT8);
  EXPECT_EQ(state->layers, (std::vector<int32_t>{1, 2, 3}));
  EXPECT_EQ(state->dtype, CacheScalarType::FLOAT32);
  const auto* index_state =
      partitions[0].find_component(CacheComponentRole::COMPRESS_INDEX_KV_STATE);
  ASSERT_NE(index_state, nullptr);
  EXPECT_EQ(index_state->layers, history->layers);
  EXPECT_EQ(partitions[2].find_component(CacheComponentRole::INDEX), nullptr);
  EXPECT_EQ(partitions[1].id().model_id, "grouped");
  EXPECT_NE(partitions[0].id().ordinal, partitions[1].id().ordinal);

  Sequence sequence = make_sequence(/*index=*/0, std::vector<int32_t>(64, 7));
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/64));
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::C4), 1u);
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::C128), 1u);
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::SWA), 4u);
  pool.deallocate_without_cache(&sequence);
}

TEST_F(BlockManagerPoolTest,
       GroupedFactoryKeepsV41SourceOwnershipAndPrefixGate) {
  ModelArgs args;
  args.model_type("deepseek_v41")
      .n_layers(4)
      .window_size(16)
      .compress_ratios({1, 2, 1, 2})
      .kv_source_layer_ids({1, 2});
  KVCacheCapacity capacity;
  capacity.n_blocks(32).block_size(8).swa_count(32).c1_count(32).c2_count(16);
  BlockManagerPool::Options options;
  options.num_blocks(32).block_size(8).enable_prefix_cache(true);
  auto result = KVCacheManagerFactory::create(
      capacity,
      args,
      /*world_size=*/1,
      torch::kBFloat16,
      options,
      PrefillPolicy(/*max_batch_tokens=*/512,
                    /*chunk_tokens=*/512,
                    /*enable_chunked_prefill=*/false,
                    /*enable_fair_budget=*/false),
      /*transfer_coordinator=*/nullptr);
  auto& pool = static_cast<BlockManagerPool&>(*result.manager);
  const auto& partitions = pool.options().grouped_partitions();
  ASSERT_EQ(partitions.size(), 3u);
  const auto* c1 = partitions[1].find_component(CacheComponentRole::KEY);
  const auto* c2 = partitions[2].find_component(CacheComponentRole::INDEX);
  const auto* state =
      partitions[0].find_component(CacheComponentRole::COMPRESS_KV_STATE);
  ASSERT_NE(c1, nullptr);
  ASSERT_NE(c2, nullptr);
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(c1->layers, (std::vector<int32_t>{2}));
  EXPECT_EQ(c2->layers, (std::vector<int32_t>{1}));
  EXPECT_EQ(c2->dtype, CacheScalarType::BFLOAT16);
  EXPECT_EQ(state->layers, (std::vector<int32_t>{1}));
  EXPECT_FALSE(partitions[1].enable_prefix_cache());
  EXPECT_FALSE(partitions[2].enable_prefix_cache());
  Sequence sequence = make_sequence(/*index=*/0, std::vector<int32_t>(32, 7));
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/32));
  sequence.kv_state().set_kv_cache_tokens_num(32);
  pool.publish_prefix(&sequence);
  pool.deallocate(&sequence);
  EXPECT_EQ(pool.num_blocks_in_prefix_cache(), (std::vector<size_t>{0}));
}

TEST_F(BlockManagerPoolTest, GroupedFactoryPreservesWindowOnlyDraft) {
  ModelArgs args;
  args.model_type("deepseek_v4_mtp")
      .n_layers(1)
      .window_size(16)
      .compress_ratios({1});
  KVCacheCapacity capacity;
  capacity.n_blocks(1).block_size(8).swa_count(16);
  BlockManagerPool::Options options;
  options.num_blocks(1).block_size(8).enable_prefix_cache(false);
  auto result = KVCacheManagerFactory::create(
      capacity,
      args,
      /*world_size=*/1,
      torch::kBFloat16,
      options,
      PrefillPolicy(/*max_batch_tokens=*/64,
                    /*chunk_tokens=*/64,
                    /*enable_chunked_prefill=*/false,
                    /*enable_fair_budget=*/false),
      /*transfer_coordinator=*/nullptr);
  auto& pool = static_cast<BlockManagerPool&>(*result.manager);
  ASSERT_EQ(pool.options().grouped_partitions().size(), 1u);
  const auto& window = pool.options().grouped_partitions().front();
  EXPECT_TRUE(window.is_sliding_window());
  EXPECT_EQ(window.components().size(), 1u);
  Sequence sequence = make_sequence(/*index=*/0, std::vector<int32_t>(16, 7));
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/16));
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::SWA), 2u);
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::C4), 0u);
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::C128), 0u);
  pool.deallocate(&sequence);
}

TEST_F(BlockManagerPoolTest,
       SharedFactoryRequiresChunkingAndAlignedStateCheckpoints) {
  BlockManagerPool::Options options;
  // A chunk may align to the capacity block without filling a KV-split block.
  options.block_size(16)
      .enable_linear_state(true)
      .linear_state_chunk_stride(8)
      .enable_prefix_cache(true);
  const auto create_manager = [&options](bool enable_chunked_prefill) {
    return KVCacheManagerFactory::create(
        make_shared_cache_capacity(),
        make_shared_cache_model_args(),
        /*world_size=*/1,
        torch::kBFloat16,
        options,
        PrefillPolicy(/*max_batch_tokens=*/512,
                      options.linear_state_chunk_stride(),
                      enable_chunked_prefill,
                      /*enable_fair_budget=*/false),
        /*transfer_coordinator=*/nullptr);
  };
  auto result = create_manager(/*enable_chunked_prefill=*/true);
  ASSERT_NE(result.manager, nullptr);
  auto& pool = static_cast<BlockManagerPool&>(*result.manager);
  ASSERT_TRUE(pool.paged_partition().has_value());
  ASSERT_TRUE(pool.recurrent_partition().has_value());
  const auto* key =
      pool.paged_partition()->find_component(CacheComponentRole::KEY);
  ASSERT_NE(key, nullptr);
  EXPECT_EQ(key->layers, (std::vector<int32_t>{1, 3}));
  const auto* conv =
      pool.recurrent_partition()->find_component(CacheComponentRole::CONV);
  const auto* ssm =
      pool.recurrent_partition()->find_component(CacheComponentRole::SSM);
  ASSERT_NE(conv, nullptr);
  ASSERT_NE(ssm, nullptr);
  EXPECT_EQ(conv->layers, (std::vector<int32_t>{0, 2}));
  EXPECT_EQ(ssm->layers, conv->layers);
  EXPECT_EQ(conv->dtype, CacheScalarType::BFLOAT16);
  EXPECT_EQ(ssm->dtype, CacheScalarType::FLOAT32);
  EXPECT_TRUE(pool.recurrent_partition()->enable_prefix_cache());
  EXPECT_EQ(pool.recurrent_partition()->ssm_snapshots_per_slot(), 1);
  EXPECT_EQ(
      BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0)->block_size(),
      8);
  EXPECT_DEATH(create_manager(/*enable_chunked_prefill=*/false),
               "requires block-aligned chunked");
  options.linear_state_chunk_stride(12);
  EXPECT_DEATH(create_manager(/*enable_chunked_prefill=*/true),
               "must be a multiple of block_size");
  options.linear_state_chunk_stride(-1);
  auto automatic = create_manager(/*enable_chunked_prefill=*/true);
  ASSERT_NE(automatic.manager, nullptr);
  EXPECT_EQ(static_cast<const BlockManagerPool&>(*automatic.manager)
                .options()
                .linear_state_chunk_stride(),
            512);
}

TEST_F(BlockManagerPoolTest,
       SharedFactoryAllowsUnusedPrefillSettingsWithoutStatePrefixes) {
  for (const bool is_decode : {false, true}) {
    BlockManagerPool::Options options;
    options.block_size(8)
        .enable_linear_state(true)
        .enable_prefix_cache(is_decode)
        .instance_is_decode(is_decode);
    auto result = KVCacheManagerFactory::create(
        make_shared_cache_capacity(),
        make_shared_cache_model_args(),
        /*world_size=*/1,
        torch::kBFloat16,
        options,
        PrefillPolicy(/*max_batch_tokens=*/512,
                      options.linear_state_chunk_stride(),
                      /*enable_chunked_prefill=*/false,
                      /*enable_fair_budget=*/false),
        /*transfer_coordinator=*/nullptr);
    ASSERT_NE(result.manager, nullptr);
  }
}

TEST_F(BlockManagerPoolTest, FactoryAllowsNonChunkedPrefillWithoutLinearState) {
  KVCacheCapacity capacity;
  capacity.n_blocks(64).block_size(4);
  ModelArgs model_args;
  model_args.model_type("test_model")
      .n_layers(2)
      .n_heads(1)
      .n_kv_heads(1)
      .head_dim(8);
  BlockManagerPool::Options options;
  options.num_blocks(64).block_size(4).enable_prefix_cache(true);
  auto result = KVCacheManagerFactory::create(
      capacity,
      model_args,
      /*world_size=*/1,
      torch::kBFloat16,
      options,
      PrefillPolicy(/*max_batch_tokens=*/512,
                    options.linear_state_chunk_stride(),
                    /*enable_chunked_prefill=*/false,
                    /*enable_fair_budget=*/false),
      /*transfer_coordinator=*/nullptr);
  ASSERT_NE(result.manager, nullptr);
}

TEST_F(BlockManagerPoolTest,
       QwenFactoryKeepsLiveStateIndependentOfHistoryGrowth) {
  KVCacheCapacity capacity;
  capacity.n_blocks(16)
      .block_size(4)
      .n_layers(4)
      .num_linear_state_blocks(8)
      .linear_ssm_checkpoint_stride(3)
      .linear_conv_state_len(5);
  ModelArgs args = make_shared_cache_model_args();
  args.model_type("qwen3_5_text")
      .full_attention_interval(4)
      .layer_types({"linear_attention",
                    "full_attention",
                    "linear_attention",
                    "linear_attention"});
  BlockManagerPool::Options options;
  options.num_blocks(16)
      .block_size(4)
      .model_id("target")
      .enable_prefix_cache(false)
      .enable_linear_state(true)
      .num_speculative_tokens(2);
  auto result = KVCacheManagerFactory::create(
      capacity,
      args,
      /*world_size=*/1,
      torch::kBFloat16,
      options,
      PrefillPolicy(/*max_batch_tokens=*/128,
                    /*chunk_tokens=*/8,
                    /*enable_chunked_prefill=*/true,
                    /*enable_fair_budget=*/false),
      /*transfer_coordinator=*/nullptr);
  auto& pool = static_cast<BlockManagerPool&>(*result.manager);
  ASSERT_TRUE(pool.paged_partition().has_value());
  ASSERT_TRUE(pool.recurrent_partition().has_value());
  const auto& state_partition = *pool.recurrent_partition();
  EXPECT_EQ(state_partition.id().model_id,
            pool.paged_partition()->id().model_id);
  EXPECT_NE(state_partition.id().ordinal, pool.paged_partition()->id().ordinal);
  EXPECT_EQ(state_partition.checkpoint_stride(), 8);
  EXPECT_EQ(state_partition.ssm_snapshots_per_slot(), 3);
  const auto* conv = state_partition.find_component(CacheComponentRole::CONV);
  const auto* ssm = state_partition.find_component(CacheComponentRole::SSM);
  ASSERT_NE(conv, nullptr);
  ASSERT_NE(ssm, nullptr);
  EXPECT_EQ(conv->layers, (std::vector<int32_t>{0, 2, 3}));
  EXPECT_EQ(conv->dtype, CacheScalarType::BFLOAT16);
  EXPECT_EQ(ssm->dtype, CacheScalarType::FLOAT32);
  EXPECT_EQ(result.shape.conv_cache_shape().front(), 8);
  EXPECT_EQ(result.shape.ssm_cache_shape().front(), 24);
  const auto* key =
      pool.paged_partition()->find_component(CacheComponentRole::KEY);
  ASSERT_NE(key, nullptr);
  EXPECT_EQ(key->layers, (std::vector<int32_t>{1}));
  auto* state_pool = BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0);
  ASSERT_NE(state_pool, nullptr);
  const size_t free_states = state_pool->num_free_blocks();
  const auto free_history = pool.num_free_blocks();
  Sequence source = make_sequence(/*index=*/0, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  ASSERT_TRUE(pool.allocate(&source));
  ASSERT_TRUE(source.has_linear_state_slot());
  const int32_t live_slot = source.get_linear_state_slot_id();
  EXPECT_EQ(source.kv_state().num_blocks(BlockType::KV), 3u);
  EXPECT_EQ(state_pool->num_free_blocks(), free_states - 1);
  source.kv_state().set_kv_cache_tokens_num(source.num_tokens());
  ASSERT_TRUE(pool.allocate(&source, /*num_tokens=*/13));
  EXPECT_EQ(source.kv_state().num_blocks(BlockType::KV), 4u);
  EXPECT_EQ(source.get_linear_state_slot_id(), live_slot);
  EXPECT_EQ(state_pool->num_free_blocks(), free_states - 1);
  pool.deallocate(&source);
  EXPECT_EQ(state_pool->num_free_blocks(), free_states);
  EXPECT_EQ(pool.num_free_blocks(), free_history);
}

TEST(KVCacheManagerFactoryTest, QwenDraftFullLayerOverridesTargetInterval) {
  KVCacheCapacity capacity;
  capacity.n_blocks(16).block_size(4).n_layers(1);
  ModelArgs args = make_shared_cache_model_args();
  args.model_type("qwen3_5_mtp")
      .n_layers(1)
      .full_attention_interval(4)
      .layer_types({"full_attention"});
  BlockManagerPool::Options options;
  options.num_blocks(16).block_size(4).model_id("target::mtp_draft");
  auto result = KVCacheManagerFactory::create(
      capacity,
      args,
      /*world_size=*/1,
      torch::kBFloat16,
      options,
      PrefillPolicy(/*max_batch_tokens=*/128,
                    /*chunk_tokens=*/8,
                    /*enable_chunked_prefill=*/false,
                    /*enable_fair_budget=*/false),
      /*transfer_coordinator=*/nullptr);
  auto& pool = static_cast<BlockManagerPool&>(*result.manager);
  ASSERT_TRUE(pool.paged_partition().has_value());
  EXPECT_FALSE(pool.recurrent_partition().has_value());
  EXPECT_EQ(pool.paged_partition()->id().model_id, "target::mtp_draft");
  const auto* key =
      pool.paged_partition()->find_component(CacheComponentRole::KEY);
  ASSERT_NE(key, nullptr);
  EXPECT_EQ(key->layers, (std::vector<int32_t>{0}));
  EXPECT_FALSE(result.shape.has_ssm_cache_shape());
  int32_t rank = 0;
  EXPECT_EQ(pool.allocate(/*num_tokens=*/9, rank).size(), 3u);
}

TEST(KVCacheManagerFactoryTest,
     PagedPartitionKeepsSplitStrideAndPoolAccounting) {
  KVCacheCapacity capacity;
  capacity.n_blocks(16).block_size(4).n_layers(2);
  ModelArgs args;
  args.n_layers(2).n_heads(2).n_kv_heads(2).head_dim(16);
  BlockManagerPool::Options options;
  options.num_blocks(16).block_size(8).model_id("target");
  auto result = KVCacheManagerFactory::create(
      capacity,
      args,
      /*world_size=*/1,
      torch::kBFloat16,
      options,
      PrefillPolicy(/*max_batch_tokens=*/512,
                    /*chunk_tokens=*/128,
                    /*enable_chunked_prefill=*/false,
                    /*enable_fair_budget=*/false),
      /*transfer_coordinator=*/nullptr,
      /*dp_size=*/2);
  auto& pool = static_cast<BlockManagerPool&>(*result.manager);
  ASSERT_TRUE(pool.paged_partition().has_value());
  EXPECT_EQ(pool.paged_partition()->id().model_id, "target");
  EXPECT_EQ(pool.paged_partition()->components().size(), 2u);
  const auto* key =
      pool.paged_partition()->find_component(CacheComponentRole::KEY);
  ASSERT_NE(key, nullptr);
  EXPECT_EQ(key->layers, (std::vector<int32_t>{0, 1}));
  EXPECT_EQ(key->dtype, CacheScalarType::BFLOAT16);
  const KVCacheShape expected_shape(capacity, args, /*world_size=*/1);
  EXPECT_EQ(result.shape.key_cache_shape(), expected_shape.key_cache_shape());
  EXPECT_EQ(result.shape.value_cache_shape(),
            expected_shape.value_cache_shape());
  const auto free_before = pool.num_free_blocks();
  {
    int32_t rank = 1;
    const auto blocks = pool.allocate(/*num_tokens=*/9, rank);
    ASSERT_EQ(blocks.size(), 2u);
    EXPECT_EQ(pool.num_free_blocks()[rank], free_before[rank] - 2);
  }
  EXPECT_EQ(pool.num_free_blocks(), free_before);
}

TEST_F(BlockManagerPoolTest,
       PagedPartitionDrivesLeafGrowthAndPrefixCapability) {
  BlockManagerPool::Options options;
  options.num_blocks(16).block_size(4).enable_prefix_cache(true);
  PagedCachePartition partition(
      CachePartitionId{"draft", 0},
      /*token_stride=*/8,
      /*enable_prefix_cache=*/false,
      {{CacheComponentRole::KEY, CacheScalarType::BFLOAT16, {0}},
       {CacheComponentRole::VALUE, CacheScalarType::BFLOAT16, {0}}});
  BlockManagerPool pool(
      options, /*dp_size=*/1, /*page_allocator=*/nullptr, std::move(partition));
  EXPECT_EQ(pool.block_size(), 8);
  EXPECT_FALSE(pool.options().enable_prefix_cache());
  EXPECT_EQ(pool.options().model_id(), "draft");
  int32_t rank = 0;
  EXPECT_EQ(pool.allocate(/*num_tokens=*/9, rank).size(), 2u);
  auto source = make_sequence(/*index=*/0, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  ASSERT_TRUE(pool.allocate(&source));
  source.kv_state().set_kv_cache_tokens_num(source.num_tokens());
  pool.publish_prefix(&source);
  pool.deallocate(&source);
  EXPECT_EQ(pool.num_blocks_in_prefix_cache(), (std::vector<size_t>{0}));
  auto repeated = make_sequence(/*index=*/1, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  pool.allocate_shared(&repeated);
  EXPECT_EQ(repeated.kv_cache_tokens_num(), 0u);
}

#if defined(USE_NPU)
TEST(KVCacheManagerFactoryTest, MlaKeepsComponentsInOneHistoryPartition) {
  for (const bool packed_mla : {false, true}) {
    SCOPED_TRACE(packed_mla);
    KVCacheCapacity capacity;
    capacity.n_blocks(16)
        .block_size(128)
        .n_layers(4)
        .enable_mla_kv_cache_quant(packed_mla)
        .enable_indexer_cache_quant(true);
    ModelArgs args;
    args.model_type("glm_moe_dsa")
        .n_layers(4)
        .enable_mla(true)
        .kv_lora_rank(512)
        .qk_rope_head_dim(64)
        .index_n_heads(1)
        .index_head_dim(128)
        .index_topk(8)
        .index_topk_pattern("FSFS");
    BlockManagerPool::Options options;
    options.num_blocks(16).block_size(128).model_id("target");
    auto result = KVCacheManagerFactory::create(
        capacity,
        args,
        /*world_size=*/1,
        torch::kBFloat16,
        options,
        PrefillPolicy(/*max_batch_tokens=*/512,
                      /*chunk_tokens=*/128,
                      /*enable_chunked_prefill=*/false,
                      /*enable_fair_budget=*/false),
        /*transfer_coordinator=*/nullptr);
    auto& pool = static_cast<BlockManagerPool&>(*result.manager);
    ASSERT_TRUE(pool.paged_partition().has_value());
    const auto& partition = *pool.paged_partition();
    ASSERT_EQ(partition.components().size(), packed_mla ? 5u : 4u);
    EXPECT_EQ(partition.find_component(CacheComponentRole::VALUE), nullptr);
    const auto* latent =
        partition.find_component(CacheComponentRole::MLA_LATENT);
    const auto* rope = partition.find_component(CacheComponentRole::MLA_ROPE);
    const auto* kv_scale =
        partition.find_component(CacheComponentRole::MLA_SCALE);
    ASSERT_NE(latent, nullptr);
    ASSERT_NE(rope, nullptr);
    EXPECT_EQ(kv_scale != nullptr, packed_mla);
    EXPECT_EQ(latent->dtype,
              packed_mla ? CacheScalarType::INT8 : CacheScalarType::BFLOAT16);
    EXPECT_EQ(rope->dtype, CacheScalarType::BFLOAT16);
    if (packed_mla) {
      ASSERT_NE(kv_scale, nullptr);
      EXPECT_EQ(kv_scale->dtype, CacheScalarType::FLOAT32);
    }
    const auto* index = partition.find_component(CacheComponentRole::INDEX);
    const auto* scale =
        partition.find_component(CacheComponentRole::INDEX_SCALE);
    ASSERT_NE(index, nullptr);
    ASSERT_NE(scale, nullptr);
    EXPECT_EQ(index->dtype, CacheScalarType::INT8);
    EXPECT_EQ(index->layers, (std::vector<int32_t>{0, 2}));
    EXPECT_EQ(scale->layers, index->layers);
    EXPECT_EQ(scale->dtype, CacheScalarType::FLOAT16);
    EXPECT_EQ(result.shape.key_cache_shape(),
              (std::vector<int64_t>{16, 128, 1, packed_mla ? 656 : 512}));
    EXPECT_EQ(result.shape.value_cache_shape(),
              (std::vector<int64_t>{16, 128, 1, packed_mla ? 656 : 64}));
    int32_t rank = 0;
    EXPECT_EQ(pool.allocate(/*num_tokens=*/129, rank).size(), 2u);
  }
}

TEST_F(BlockManagerPoolTest, GlmMtpTailOnlyUsesCheckpointStateWithoutSsmRows) {
  ModelArgs args;
  args.model_type("glm5_next_mtp")
      .n_layers(1)
      .n_heads(1)
      .n_kv_heads(1)
      .head_dim(16)
      .enable_mla(true)
      .kv_lora_rank(16)
      .qk_rope_head_dim(8)
      .index_n_heads(1)
      .index_head_dim(8)
      .index_topk(2)
      .index_kpool(4)
      .index_kpool_compress(true)
      .index_kpool_always_select_tail(true)
      .full_attention_interval(4)
      .layer_types({"deepseek_sparse_attention"})
      .indexer_types({"full"});
  KVCacheCapacity capacity;
  capacity.n_blocks(16)
      .block_size(8)
      .n_layers(1)
      .num_linear_state_blocks(8)
      .linear_ssm_checkpoint_stride(5)
      .kpool_layout(KPoolCacheLayout::COMPRESSED_WITH_TAIL)
      .kpool_tail_len(9);
  BlockManagerPool::Options options;
  options.num_blocks(16)
      .block_size(8)
      .model_id("target::mtp_draft")
      .enable_prefix_cache(true)
      .enable_linear_state(true);
  auto result = KVCacheManagerFactory::create(
      capacity,
      args,
      /*world_size=*/1,
      torch::kBFloat16,
      options,
      PrefillPolicy(/*max_batch_tokens=*/128,
                    /*chunk_tokens=*/16,
                    /*enable_chunked_prefill=*/true,
                    /*enable_fair_budget=*/false),
      /*transfer_coordinator=*/nullptr);
  auto& pool = static_cast<BlockManagerPool&>(*result.manager);
  ASSERT_TRUE(pool.paged_partition().has_value());
  ASSERT_TRUE(pool.recurrent_partition().has_value());
  const auto& state = *pool.recurrent_partition();
  EXPECT_EQ(state.id().model_id, "target::mtp_draft");
  EXPECT_NE(state.id().ordinal, pool.paged_partition()->id().ordinal);
  EXPECT_EQ(state.components().size(), 1u);
  EXPECT_EQ(state.ssm_snapshots_per_slot(), 0);
  EXPECT_EQ(state.tail_tokens_per_slot(), 9);
  EXPECT_EQ(state.checkpoint_stride(), 16);
  EXPECT_TRUE(state.enable_prefix_cache());
  EXPECT_EQ(state.find_component(CacheComponentRole::CONV), nullptr);
  EXPECT_EQ(state.find_component(CacheComponentRole::SSM), nullptr);
  const auto* tail = state.find_component(CacheComponentRole::KPOOL_TAIL);
  ASSERT_NE(tail, nullptr);
  EXPECT_EQ(tail->layers, (std::vector<int32_t>{0}));
  EXPECT_FALSE(result.shape.has_conv_cache_shape());
  EXPECT_FALSE(result.shape.has_ssm_cache_shape());
  Sequence source = make_sequence(
      0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
  ASSERT_TRUE(pool.allocate(&source));
  EXPECT_TRUE(source.has_linear_state_slot());
  auto* state_pool = BlockManagerPoolTestPeer::linear_leaf(pool, 0);
  ASSERT_NE(state_pool, nullptr);
  const PrefixHash hash = compute_linear_state_prefix_hash_for_test(
      source.tokens(), state_pool->block_size(), /*boundary_tokens=*/16);
  source.kv_state().set_kv_cache_tokens_num(source.num_tokens());
  pool.publish_prefix(&source);
  pool.deallocate_without_cache(&source);
  ASSERT_GE(insert_linear_state_checkpoint(state_pool, hash), 1);
  Sequence repeated = make_sequence(
      1, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
  pool.allocate_shared(&repeated);
  EXPECT_EQ(repeated.kv_cache_tokens_num(), 16u);
  pool.deallocate_without_cache(&repeated);
}

#endif

TEST(KVCacheManagerFactoryTest,
     ValidatesVirtualMemoryHostOffloadProviderBeforeAllocation) {
  KVCacheCapacity capacity;
  capacity.n_blocks(64).block_size(4);
  ModelArgs model_args;
  model_args.model_type("test_model")
      .n_layers(2)
      .n_heads(1)
      .n_kv_heads(1)
      .head_dim(8);
  BlockManagerPool::Options options;
  options.enable_vmm_kvcache(true).enable_host_offload(true);
  HostCacheValidationOptions host_options;
  host_options.host_blocks_factor = 2.0;

  EXPECT_DEATH(KVCacheManagerFactory::create(
                   capacity,
                   model_args,
                   /*world_size=*/1,
                   torch::kBFloat16,
                   options,
                   PrefillPolicy(/*max_batch_tokens=*/512,
                                 options.linear_state_chunk_stride(),
                                 /*enable_chunked_prefill=*/true,
                                 /*enable_fair_budget=*/false),
                   /*transfer_coordinator=*/nullptr,
                   /*dp_size=*/1,
                   host_options),
               "current platform has no host KV offload");
}

TEST(KVCacheManagerFactoryTest,
     ValidatesVirtualMemoryHostOffloadPrefixCacheBeforeAllocation) {
  KVCacheCapacity capacity;
  capacity.n_blocks(64).block_size(4);
  ModelArgs model_args;
  model_args.model_type("test_model")
      .n_layers(2)
      .n_heads(1)
      .n_kv_heads(1)
      .head_dim(8);
  BlockManagerPool::Options options;
  options.enable_vmm_kvcache(true).enable_host_offload(true);
  HostCacheValidationOptions host_options;
  host_options.host_blocks_factor = 2.0;
  host_options.supports_host_kv_offload = true;
  host_options.enable_prefix_cache = false;

  EXPECT_DEATH(KVCacheManagerFactory::create(
                   capacity,
                   model_args,
                   /*world_size=*/1,
                   torch::kBFloat16,
                   options,
                   PrefillPolicy(/*max_batch_tokens=*/512,
                                 options.linear_state_chunk_stride(),
                                 /*enable_chunked_prefill=*/true,
                                 /*enable_fair_budget=*/false),
                   /*transfer_coordinator=*/nullptr,
                   /*dp_size=*/1,
                   host_options),
               "prefix caching is disabled");
}

TEST_F(BlockManagerPoolTest,
       SharedFactorySeparatesLogicalIdsAndPhysicalBudget) {
  BlockManagerPool::Options options;
  options.block_size(8)
      .enable_linear_state(true)
      .linear_state_chunk_stride(4)
      .enable_prefix_cache(false);
  auto result = KVCacheManagerFactory::create(
      make_shared_cache_capacity(),
      make_shared_cache_model_args(),
      /*world_size=*/1,
      torch::kBFloat16,
      options,
      PrefillPolicy(/*max_batch_tokens=*/512,
                    options.linear_state_chunk_stride(),
                    /*enable_chunked_prefill=*/true,
                    /*enable_fair_budget=*/false),
      /*transfer_coordinator=*/nullptr);
  EXPECT_EQ(result.shape.key_cache_shape().front(), 53);
  EXPECT_EQ(result.shape.conv_cache_shape().front(), 53);
  EXPECT_EQ(result.manager->block_resource_units(BlockType::KV, 2), 128);
  EXPECT_EQ(result.manager->block_resource_units(BlockType::LINEAR, 1), 20);
  ASSERT_EQ(result.manager->num_free_blocks(), std::vector<size_t>{640});
  auto sequence = make_sequence(0, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  ASSERT_TRUE(result.manager->allocate(&sequence));
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::KV), 2);
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1);
  EXPECT_EQ(result.manager->num_free_blocks(), std::vector<size_t>{492});
  result.manager->deallocate(&sequence);
  EXPECT_EQ(result.manager->num_free_blocks(), std::vector<size_t>{640});
}

TEST_F(BlockManagerPoolTest, SharedFactoryAccountsForTypedSlabFragmentation) {
  BlockManagerPool::Options options;
  options.block_size(8)
      .enable_linear_state(true)
      .linear_state_chunk_stride(4)
      .enable_prefix_cache(false);
  KVCacheCapacity capacity = make_shared_cache_capacity();
  capacity.shared_slab_count(5);
  auto result = KVCacheManagerFactory::create(
      capacity,
      make_shared_cache_model_args(),
      /*world_size=*/1,
      torch::kBFloat16,
      options,
      PrefillPolicy(/*max_batch_tokens=*/512,
                    options.linear_state_chunk_stride(),
                    /*enable_chunked_prefill=*/true,
                    /*enable_fair_budget=*/false),
      /*transfer_coordinator=*/nullptr);
  auto first = make_sequence(0, {1, 2, 3, 4, 5, 6, 7, 8});
  auto second = make_sequence(1, {9, 10, 11, 12, 13, 14, 15, 16});
  EXPECT_FALSE(result.manager->exceeds_total_capacity(&first, 8));
  // Two Full blocks plus one state cost 148 B, below the 192 B budget,
  // but require two KV groups and one group for each recurrent role.
  EXPECT_TRUE(result.manager->exceeds_total_capacity(&first, 16));
  EXPECT_TRUE(result.manager->exceeds_total_capacity(
      std::vector<KVCacheManager::KVCacheReservation>{{&first, 8},
                                                      {&second, 8}}));
  ASSERT_TRUE(result.manager->allocate(&first));
  EXPECT_FALSE(result.manager->allocate(&second));
  result.manager->deallocate(&first);
  EXPECT_EQ(result.manager->num_free_blocks(), std::vector<size_t>{192});
}

TEST(KVCacheManagerFactoryTest, RejectsSharedStorageCostMismatch) {
  BlockManagerPool::Options options;
  options.block_size(8)
      .enable_linear_state(true)
      .linear_state_chunk_stride(4)
      .enable_prefix_cache(false);
  auto capacity = make_shared_cache_capacity();
  capacity.shared_linear_state_slots(3);
  EXPECT_DEATH(KVCacheManagerFactory::create(
                   capacity,
                   make_shared_cache_model_args(),
                   /*world_size=*/1,
                   torch::kBFloat16,
                   options,
                   PrefillPolicy(/*max_batch_tokens=*/512,
                                 options.linear_state_chunk_stride(),
                                 /*enable_chunked_prefill=*/true,
                                 /*enable_fair_budget=*/false),
                   /*transfer_coordinator=*/nullptr),
               "num_linear_groups");
}

TEST(BlockManagerTest, ReleasesAllocatedBlocksAfterLastAlias) {
  constexpr uint32_t kNumBlocks = 4;
  constexpr uint32_t kBlockSize = 2;
  BlockManager::Options options;
  options.num_blocks(kNumBlocks).block_size(kBlockSize);
  BlockManagerImpl manager(options);

  auto blocks = manager.allocate(kNumBlocks - 1);
  ASSERT_EQ(blocks.size(), kNumBlocks - 1);
  EXPECT_EQ(blocks.front().id(), 1);
  EXPECT_EQ(blocks.front().size(), kBlockSize);
  EXPECT_TRUE(manager.allocate(/*num_blocks=*/1).empty());

  Block alias = blocks.front();
  EXPECT_EQ(alias.ref_count(), 2);
  Block moved = std::move(alias);
  EXPECT_FALSE(alias.is_valid());
  moved = blocks.back();
  EXPECT_EQ(blocks.front().ref_count(), 1);
  EXPECT_EQ(moved.id(), blocks.back().id());

  blocks.clear();
  EXPECT_EQ(manager.num_free_blocks(), kNumBlocks - 2);
  moved = Block();
  EXPECT_EQ(manager.num_free_blocks(), kNumBlocks - 1);
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
      .linear_state_chunk_stride(4)
      .linear_state_num_slots(64);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seq = make_sequence(0, /*prompt_tokens=*/{1, 2, 3});
  EXPECT_TRUE(pool.allocate(&seq));
  EXPECT_TRUE(seq.get_embedding_block_id() >= 0);
  // id 0 is the reserved padding slot, so a real assignment is strictly
  // positive.
  EXPECT_GT(seq.get_embedding_block_id(), 0);
}

TEST_F(BlockManagerPoolTest, EqualAvailableCapacityRotatesAcrossDpRanks) {
  BlockManagerPool::Options options;
  options.num_blocks(1).block_size(1).enable_prefix_cache(false);
  BlockManagerPool pool(options, /*dp_size=*/3);

  EXPECT_EQ(pool.select_dp_rank(), 0);
  EXPECT_EQ(pool.select_dp_rank(), 1);
  EXPECT_EQ(pool.select_dp_rank(), 2);
  EXPECT_EQ(pool.select_dp_rank(), 0);
}

TEST_F(BlockManagerPoolTest, DpLinearStateAndPrefixCachesAreIsolated) {
  BlockManagerPool::Options options;
  options.num_blocks(8)
      .host_num_blocks(0)
      .block_size(4)
      .enable_prefix_cache(true)
      .enable_linear_state(true)
      .linear_state_chunk_stride(4)
      .linear_state_num_slots(4);
  BlockManagerPoolTestPeer pool(options, /*dp_size=*/2);

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

  BlockManager* first_linear = pool.linear_leaf(/*dp_rank=*/0);
  BlockManager* second_linear = pool.linear_leaf(/*dp_rank=*/1);
  ASSERT_NE(first_linear, nullptr);
  ASSERT_NE(second_linear, nullptr);
  ASSERT_NE(first_linear, second_linear);

  const XXH3Key first_only_hash = make_prefix_hash(/*tag=*/42);
  ASSERT_GE(BlockManagerPoolTestPeer::insert_checkpoint(first_linear,
                                                        first_only_hash),
            1);
  EXPECT_TRUE(
      BlockManagerPoolTestPeer::contains(first_linear, first_only_hash));
  EXPECT_FALSE(
      BlockManagerPoolTestPeer::contains(second_linear, first_only_hash));

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
  EXPECT_EQ(pool.select_dp_rank(), 1);

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
      .linear_state_chunk_stride(4)
      .linear_state_num_slots(64);
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
      .linear_state_chunk_stride(4)
      .linear_state_num_slots(64);
  BlockManagerPool pool(options, /*dp_size=*/1);

  constexpr size_t kSequenceCount = 4;
  std::vector<std::unique_ptr<Sequence>> sequences;
  sequences.reserve(kSequenceCount);
  for (size_t i = 0; i < kSequenceCount; ++i) {
    auto sequence = std::unique_ptr<Sequence>(
        new Sequence(make_sequence(i, /*prompt_tokens=*/{1})));
    EXPECT_TRUE(pool.allocate(sequence.get()));
    EXPECT_TRUE(sequence->get_embedding_block_id() >= 0);
    sequences.push_back(std::move(sequence));
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
      .linear_state_chunk_stride(4)
      .linear_state_num_slots(64);
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
      .linear_state_chunk_stride(4)
      .linear_state_num_slots(64);
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
      .linear_state_chunk_stride(4)
      .linear_state_num_slots(64);
  BlockManagerPool pool(options, /*dp_size=*/2);

  Sequence seq0 = make_sequence(0, /*prompt_tokens=*/{1});
  ASSERT_TRUE(pool.try_allocate(&seq0));
  EXPECT_EQ(seq0.dp_rank(), 0);

  Sequence seq1 = make_sequence(1, /*prompt_tokens=*/{2});
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
      .linear_state_chunk_stride(4)
      .linear_state_num_slots(64);
  BlockManagerPool pool(options, /*dp_size=*/2);

  Sequence seq0 = make_sequence(0, /*prompt_tokens=*/{1});
  Sequence seq1 = make_sequence(1, /*prompt_tokens=*/{2});
  ASSERT_TRUE(pool.try_allocate(&seq0));
  ASSERT_TRUE(pool.try_allocate(&seq1));

  Sequence retry = make_sequence(2, /*prompt_tokens=*/{3});
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
      .linear_state_chunk_stride(8)
      .linear_state_num_slots(kLinearStateSlots);
  BlockManagerPool pool(options, /*dp_size=*/1);

  // Each leaf reserves slot 0 as padding, so a fresh KV pool reports
  // kKvBlocks - 1 free. The key invariant: capacity tracks the KV pool, never
  // the (much larger) linear slot pool.
  const std::vector<size_t> free = pool.num_free_blocks();
  ASSERT_EQ(free.size(), 1u);
  EXPECT_EQ(free[0], kKvBlocks - 1);
  EXPECT_LT(free[0], static_cast<size_t>(kLinearStateSlots));
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
  pool.publish_prefix(&cached_seq);
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
  pool.publish_prefix(&source);
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
      .linear_state_chunk_stride(4)
      .linear_state_num_slots(64);
  BlockManagerPoolTestPeer pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(allocate_kv_prefix_for_test(pool, cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.publish_prefix(&cached_seq);
  cached_seq.update_linear_state_hashes(options.linear_state_chunk_stride());
  const XXH3Key checkpoint_hash = cached_seq.linear_state_hashes()[1];
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
  BlockManager* prefix_cache = pool.linear_leaf(/*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  EXPECT_GE(BlockManagerPoolTestPeer::insert_checkpoint(prefix_cache,
                                                        checkpoint_hash),
            1);
  EXPECT_TRUE(
      BlockManagerPoolTestPeer::contains(prefix_cache, checkpoint_hash));

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
      .linear_state_chunk_stride(4)
      .linear_state_num_slots(64);
  BlockManagerPoolTestPeer pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(allocate_kv_prefix_for_test(pool, cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.publish_prefix(&cached_seq);
  cached_seq.update_linear_state_hashes(options.linear_state_chunk_stride());
  const XXH3Key tail_hash = cached_seq.linear_state_hashes()[1];
  pool.deallocate_without_cache(&cached_seq);

  BlockManager* prefix_cache = pool.linear_leaf(/*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  EXPECT_GE(
      BlockManagerPoolTestPeer::insert_checkpoint(prefix_cache, tail_hash), 1);

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
      .linear_state_chunk_stride(4)
      .linear_state_num_slots(64);
  BlockManagerPoolTestPeer pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(allocate_kv_prefix_for_test(pool, cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.publish_prefix(&cached_seq);
  cached_seq.update_linear_state_hashes(options.linear_state_chunk_stride());
  const XXH3Key tail_hash = cached_seq.linear_state_hashes()[1];
  pool.deallocate_without_cache(&cached_seq);

  BlockManager* prefix_cache = pool.linear_leaf(/*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  EXPECT_GE(
      BlockManagerPoolTestPeer::insert_checkpoint(prefix_cache, tail_hash), 1);

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
  BlockManagerPoolTestPeer pool(
      make_linear_state_pool_options(/*linear_state_num_slots=*/4),
      /*dp_size=*/1);
  BlockManager* leaf = pool.linear_leaf(/*dp_rank=*/0);
  ASSERT_NE(leaf, nullptr);
  Sequence sequence = make_sequence(0, {1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&sequence, 4));
  Batch batch(&sequence);
  sequence.kv_state().set_kv_cache_tokens_num(4);
  const int32_t source_id = sequence.get_linear_state_slot_id();
  const Block retained_old_read =
      sequence.kv_state().blocks(BlockType::LINEAR)[0];
  const XXH3Key evictable_hash = make_prefix_hash(/*tag=*/2);
  const XXH3Key pinned_hash = make_prefix_hash(/*tag=*/1);
  ASSERT_GE(BlockManagerPoolTestPeer::insert_checkpoint(leaf, evictable_hash),
            1);
  ASSERT_GE(BlockManagerPoolTestPeer::insert_checkpoint(leaf, pinned_hash), 1);
  auto pinned = BlockManagerPoolTestPeer::match(leaf, pinned_hash);
  ASSERT_FALSE(pinned.empty());
  ASSERT_TRUE(pool.allocate(&sequence, 8));
  EXPECT_NE(sequence.get_linear_state_slot_id(), source_id);
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_EQ(sequence.kv_state().blocks(BlockType::LINEAR)[0].id(), source_id);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(leaf, pinned_hash));
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(leaf, evictable_hash));
  sequence.update_linear_state_hashes(4);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(
      leaf, sequence.linear_state_hashes()[0]));
  Block::release(std::move(pinned));
  pool.deallocate_without_cache(&sequence);
}

TEST_F(BlockManagerPoolTest,
       PrefixMountSurvivesOutputAllocationFailureAndRetry) {
  BlockManagerPoolTestPeer pool(
      make_linear_state_pool_options(/*linear_state_num_slots=*/3),
      /*dp_size=*/1);
  BlockManager* leaf = pool.linear_leaf(/*dp_rank=*/0);
  ASSERT_NE(leaf, nullptr);
  Sequence cached = make_sequence(0, {1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(allocate_kv_prefix_for_test(pool, cached));
  cached.kv_state().set_kv_cache_tokens_num(8);
  pool.publish_prefix(&cached);
  cached.update_linear_state_hashes(4);
  const int32_t source_id = BlockManagerPoolTestPeer::insert_checkpoint(
      leaf, cached.linear_state_hashes()[1]);
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
        .linear_state_chunk_stride(kChunkStride)
        .linear_state_num_slots(16);
    BlockManagerPoolTestPeer pool(options, /*dp_size=*/1);

    const std::vector<int32_t> prompt_tokens(513, 1);
    Sequence cached_seq = make_sequence(0, prompt_tokens);
    ASSERT_TRUE(allocate_kv_prefix_for_test(pool, cached_seq));
    cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
    pool.publish_prefix(&cached_seq);

    BlockManager* prefix_cache = pool.linear_leaf(/*dp_rank=*/0);
    ASSERT_NE(prefix_cache, nullptr);
    cached_seq.update_linear_state_hashes(kChunkStride);
    int32_t expected_slot = -1;
    for (size_t checkpoint : checkpoints) {
      const XXH3Key hash =
          cached_seq.linear_state_hashes()[checkpoint / kChunkStride - 1];
      const int32_t slot =
          BlockManagerPoolTestPeer::insert_checkpoint(prefix_cache, hash);
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
      .linear_state_chunk_stride(4)
      .linear_state_num_slots(64);
  BlockManagerPoolTestPeer pool(options, /*dp_size=*/1);

  Sequence short_cached_seq =
      make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(allocate_kv_prefix_for_test(pool, short_cached_seq));
  short_cached_seq.kv_state().set_kv_cache_tokens_num(
      short_cached_seq.num_tokens());
  pool.publish_prefix(&short_cached_seq);
  pool.deallocate_without_cache(&short_cached_seq);

  Sequence long_cached_seq =
      make_sequence(1,
                    /*prompt_tokens=*/{
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_TRUE(allocate_kv_prefix_for_test(pool, long_cached_seq));
  long_cached_seq.update_linear_state_hashes(
      options.linear_state_chunk_stride());
  const XXH3Key kv_boundary_hash = long_cached_seq.linear_state_hashes()[1];
  const XXH3Key long_checkpoint_hash = long_cached_seq.linear_state_hashes()[3];
  pool.deallocate_without_cache(&long_cached_seq);

  BlockManager* prefix_cache = pool.linear_leaf(/*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  EXPECT_GE(BlockManagerPoolTestPeer::insert_checkpoint(prefix_cache,
                                                        kv_boundary_hash),
            1);
  EXPECT_GE(BlockManagerPoolTestPeer::insert_checkpoint(prefix_cache,
                                                        long_checkpoint_hash),
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
      .linear_state_chunk_stride(4)
      .linear_state_num_slots(64);
  BlockManagerPoolTestPeer pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      make_sequence(0,
                    /*prompt_tokens=*/{
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_TRUE(allocate_kv_prefix_for_test(pool, cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.publish_prefix(&cached_seq);
  cached_seq.update_linear_state_hashes(options.linear_state_chunk_stride());
  const XXH3Key inner_hash = cached_seq.linear_state_hashes()[1];
  const XXH3Key tail_hash = cached_seq.linear_state_hashes()[3];
  pool.deallocate_without_cache(&cached_seq);

  BlockManager* prefix_cache = pool.linear_leaf(/*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  EXPECT_GE(
      BlockManagerPoolTestPeer::insert_checkpoint(prefix_cache, inner_hash), 1);
  EXPECT_GE(
      BlockManagerPoolTestPeer::insert_checkpoint(prefix_cache, tail_hash), 1);

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

TEST_F(BlockManagerPoolTest, SequenceUpdateBlockHashes) {
  const uint32_t block_size = 4;
  const uint32_t n_blocks = 5;
  std::vector<int32_t> prompt;
  prompt.reserve(n_blocks * block_size);
  for (uint32_t i = 0; i < n_blocks * block_size; ++i) {
    prompt.push_back(static_cast<int32_t>(i * 7 + 1));
  }

  Sequence seq = make_sequence(/*index=*/0, /*prompt_tokens=*/prompt);
  const std::vector<XXH3Key> expected = ExpectedChain(prompt, block_size);

  seq.update_block_hashes(block_size, BlockHasherType::TEXT);
  EXPECT_TRUE(seq.block_hashes() == expected);

  seq.update_block_hashes(/*block_size=*/2 * block_size,
                          /*hasher_type=*/BlockHasherType::TEXT);
  EXPECT_EQ(seq.block_hashes().size(), n_blocks / 2);
  seq.update_block_hashes(block_size, BlockHasherType::TEXT);
  EXPECT_TRUE(seq.block_hashes() == expected);
}

// In-batch prefix cache publishes only the full blocks covered by the given
// token budget. When the budget is overestimated, publish_prefix() clamps to
// the sequence's own tokens and registers just the complete blocks.
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

  // num_tokens (8) is larger than the 7 real tokens. Publication clamps to the
  // sequence tokens, so only the single full block (tokens [0, 4)) is cached.
  EXPECT_NO_FATAL_FAILURE(pool->publish_prefix(&seq, /*num_tokens=*/8));
  EXPECT_EQ(pool->num_blocks_in_prefix_cache()[0], 1u);
}

}  // namespace xllm
