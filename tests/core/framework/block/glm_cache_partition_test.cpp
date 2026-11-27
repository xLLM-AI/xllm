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
#include <string>
#include <utility>
#include <vector>

#include "core/common/prefill_policy.h"
#include "core/framework/block/block_manager_pool.h"
#include "core/framework/block/kv_cache_manager_factory.h"
#include "core/framework/model/model_args.h"
#include "core/framework/request/sequence.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "core/kv_cache/storage/kv_cache_create_options.h"
#include "core/kv_cache/storage/linear_state_restore.h"
#include "framework/block/block_manager_pool_test_peer.h"
#include "platform/device.h"

namespace xllm {
namespace {

class GlmCachePartitionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    Device device(/*device_index=*/0);
    device.set_device();
    device.init_device_context();
    sampling_param_.is_embeddings = false;
  }

  Sequence make_sequence(const std::vector<int32_t>& prompt_tokens) {
    SequenceParams params;
    params.seq_capacity = prompt_tokens.size() + 8;
    params.enable_schedule_overlap = false;
    params.sampling_param = &sampling_param_;
    params.stopping_checker = &stopping_checker_;
    IncrementalDecoder decoder(/*prompt=*/"prompt",
                               prompt_tokens.size(),
                               /*echo=*/false,
                               /*skip_special_tokens=*/true);
    return Sequence(/*index=*/0,
                    prompt_tokens,
                    /*input_embedding=*/torch::Tensor(),
                    /*mm_data=*/MMData(),
                    std::move(decoder),
                    params);
  }

  RequestSamplingParam sampling_param_;
  StoppingChecker stopping_checker_;
};

TEST_F(GlmCachePartitionTest, HybridPartitionsMatchStorageAndStateGrowth) {
  for (const std::string& model_type :
       {"glm5_next", "glm5_next_text", "glm5_3_flash"}) {
    for (const int32_t layout : {0, 1, 2}) {
      SCOPED_TRACE(::testing::Message() << model_type << " layout=" << layout);
      const bool packed_kpool = layout == 1;
      const bool compressed_kpool = layout == 2;
      ModelArgs args;
      args.model_type(model_type)
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
          .mamba_ssm_dtype("float32")
          .enable_mla(true)
          .kv_lora_rank(16)
          .qk_rope_head_dim(8)
          .index_n_heads(1)
          .index_head_dim(8)
          .index_topk(2)
          .indexer_types({"full", "full", "full", "shared"})
          .layer_types({"linear_attention",
                        "deepseek_sparse_attention",
                        "linear_attention",
                        "deepseek_sparse_attention"})
          .index_kpool(4)
          .index_kpool_compress(packed_kpool || compressed_kpool)
          .index_kpool_always_select_tail(compressed_kpool);
      KVCacheCapacity capacity;
      capacity.n_blocks(16)
          .block_size(8)
          .n_layers(4)
          .num_linear_state_blocks(8)
          .linear_ssm_checkpoint_stride(3)
          .linear_conv_state_len(3)
          .kpool_layout(compressed_kpool
                            ? KPoolCacheLayout::COMPRESSED_WITH_TAIL
                            : KPoolCacheLayout::PACKED)
          .kpool_tail_len(compressed_kpool ? 6 : 0);
      BlockManagerPool::Options options;
      options.num_blocks(16)
          .block_size(8)
          .model_id("glm_target")
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
                        /*chunk_tokens=*/16,
                        /*enable_chunked_prefill=*/true,
                        /*enable_fair_budget=*/false),
          /*transfer_coordinator=*/nullptr);
      auto& pool = static_cast<BlockManagerPool&>(*result.manager);
      ASSERT_TRUE(pool.paged_partition().has_value());
      ASSERT_TRUE(pool.recurrent_partition().has_value());
      const auto& history = *pool.paged_partition();
      const auto& state = *pool.recurrent_partition();
      EXPECT_EQ(history.token_stride(), 8);
      EXPECT_EQ(state.checkpoint_stride(), 16);
      EXPECT_EQ(state.ssm_snapshots_per_slot(), 3);
      EXPECT_EQ(state.tail_tokens_per_slot(), compressed_kpool ? 6 : 0);
      const auto* main = history.find_component(CacheComponentRole::MLA_LATENT);
      const auto* index = history.find_component(CacheComponentRole::INDEX);
      const auto* conv = state.find_component(CacheComponentRole::CONV);
      const auto* ssm = state.find_component(CacheComponentRole::SSM);
      const auto* tail = state.find_component(CacheComponentRole::KPOOL_TAIL);
      ASSERT_NE(main, nullptr);
      ASSERT_NE(index, nullptr);
      ASSERT_NE(conv, nullptr);
      ASSERT_NE(ssm, nullptr);
      EXPECT_EQ(main->layers, (std::vector<int32_t>{1, 3}));
      EXPECT_EQ(index->layers, (std::vector<int32_t>{1}));
      EXPECT_EQ(conv->layers, (std::vector<int32_t>{0, 2}));
      EXPECT_EQ(ssm->layers, conv->layers);
      EXPECT_EQ(ssm->dtype, CacheScalarType::FLOAT32);
      EXPECT_EQ(tail != nullptr, compressed_kpool);
      if (tail != nullptr) {
        EXPECT_EQ(tail->layers, index->layers);
        EXPECT_EQ(tail->dtype, CacheScalarType::BFLOAT16);
      }
      EXPECT_EQ(
          history.find_component(CacheComponentRole::INDEX_GATE) != nullptr,
          packed_kpool);
      EXPECT_EQ(
          history.find_component(CacheComponentRole::INDEX_VALID) != nullptr,
          packed_kpool);
      EXPECT_EQ(result.shape.index_cache_shape(),
                (std::vector<int64_t>{
                    16, compressed_kpool ? 2 : 8, 1, packed_kpool ? 17 : 8}));
      auto create_options = get_kv_cache_create_options(args);
      create_options.device(torch::Device("npu:0"))
          .num_layers(4)
          .model_type(model_type)
          .dtype(torch::kBFloat16)
          .ssm_dtype(torch::kFloat32)
          .enable_lighting_indexer(true)
          .indexer_cache_enabled_layers({false, true, false, false});
      KVCache linear(result.shape, create_options, /*layer_id=*/0);
      KVCache indexed(result.shape, create_options, /*layer_id=*/1);
      KVCache shared_index(result.shape, create_options, /*layer_id=*/3);
      EXPECT_EQ(linear.get_ssm_cache().size(0), 24);
      EXPECT_EQ(linear.get_ssm_cache().scalar_type(), torch::kFloat32);
      EXPECT_EQ(indexed.get_index_cache().sizes().vec(),
                result.shape.index_cache_shape());
      EXPECT_FALSE(shared_index.get_index_cache().defined());
      EXPECT_EQ(indexed.get_kpool_tail().defined(), compressed_kpool);
      if (compressed_kpool) {
        EXPECT_EQ(indexed.get_kpool_tail().sizes().vec(),
                  (std::vector<int64_t>{8, 2, 6, 8}));
      }
      linear.get_conv_cache()[1].fill_(11);
      const torch::Tensor conv_checkpoint = linear.get_conv_cache()[1].clone();
      linear.get_ssm_cache().narrow(0, 3, 3).copy_(
          torch::arange(3, torch::kFloat32).reshape({3, 1, 1, 1}));
      const torch::Tensor ssm_checkpoint =
          linear.get_ssm_cache().narrow(0, 3, 3).clone();
      torch::Tensor tail_checkpoint;
      torch::Tensor tail_padding;
      if (compressed_kpool) {
        indexed.get_kpool_tail()[0].fill_(23);
        indexed.get_kpool_tail()[1].fill_(17);
        tail_checkpoint = indexed.get_kpool_tail()[1].clone();
        tail_padding = indexed.get_kpool_tail()[0].clone();
      }
      std::vector<KVCache> caches;
      caches.reserve(3);
      caches.emplace_back(std::move(linear));
      caches.emplace_back(std::move(indexed));
      caches.emplace_back(std::move(shared_index));
      std::vector<int64_t> validity = {0};
      restore_linear_state_slots(caches,
                                 {2},
                                 {1},
                                 validity,
                                 /*reads_distinct_state=*/false);
      EXPECT_EQ(validity, (std::vector<int64_t>{1}));
      EXPECT_TRUE(torch::equal(caches[0].get_conv_cache()[2], conv_checkpoint));
      EXPECT_TRUE(torch::equal(caches[0].get_ssm_cache().narrow(0, 6, 3),
                               ssm_checkpoint));
      if (compressed_kpool) {
        EXPECT_TRUE(
            torch::equal(caches[1].get_kpool_tail()[2], tail_checkpoint));
        validity = {0};
        restore_linear_state_slots(caches,
                                   {2},
                                   {2},
                                   validity,
                                   /*reads_distinct_state=*/false);
        EXPECT_EQ(
            caches[1].get_kpool_tail()[2][0].count_nonzero().item<int64_t>(),
            0);
        EXPECT_TRUE(torch::isneginf(caches[1].get_kpool_tail()[2][1])
                        .all()
                        .item<bool>());
        EXPECT_TRUE(
            torch::equal(caches[1].get_kpool_tail()[1], tail_checkpoint));
        EXPECT_TRUE(torch::equal(caches[1].get_kpool_tail()[0], tail_padding));
      }
      auto* state_pool = BlockManagerPoolTestPeer::linear_leaf(pool, 0);
      ASSERT_NE(state_pool, nullptr);
      const size_t free_states = state_pool->num_free_blocks();
      const auto free_history = pool.num_free_blocks();
      Sequence source = make_sequence({1, 2, 3, 4, 5, 6, 7, 8, 9});
      ASSERT_TRUE(pool.allocate(&source));
      const int32_t live_slot = source.get_linear_state_slot_id();
      EXPECT_EQ(state_pool->num_free_blocks(), free_states - 1);
      source.kv_state().set_kv_cache_tokens_num(source.num_tokens());
      ASSERT_TRUE(pool.allocate(&source, /*num_tokens=*/17));
      EXPECT_EQ(source.get_linear_state_slot_id(), live_slot);
      EXPECT_EQ(state_pool->num_free_blocks(), free_states - 1);
      pool.deallocate(&source);
      EXPECT_EQ(state_pool->num_free_blocks(), free_states);
      EXPECT_EQ(pool.num_free_blocks(), free_history);
    }
  }
}

}  // namespace
}  // namespace xllm
