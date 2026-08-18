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

#include "kv_cache_estimation.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "framework/model/model_args.h"

namespace xllm {
namespace {

ModelArgs make_standard_args() {
  ModelArgs model_args;
  model_args.n_layers(4).head_dim(16);
  return model_args;
}

KVCacheEstimateOptions make_estimate_options() {
  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat16;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 1024 * 1024;
  options.block_size = 16;
  options.world_size = 1;
  options.n_local_kv_heads = 2;
  options.max_seqs_per_batch = 8;
  return options;
}

}  // namespace

TEST(KVCacheEstimationTest, EstimatesStandardAttentionBlocks) {
  ModelArgs model_args = make_standard_args();
  KVCacheEstimateOptions options = make_estimate_options();

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.cache_size_in_bytes(), 1024 * 1024);
  EXPECT_EQ(capacity.block_size(), 16);
  EXPECT_EQ(capacity.slot_size(), 128);
  EXPECT_EQ(capacity.n_layers(), 4);
  EXPECT_EQ(capacity.num_full_attention_layers(), 4);
  EXPECT_EQ(capacity.num_linear_attention_layers(), 0);
  EXPECT_EQ(capacity.n_blocks(), 128);
}

TEST(KVCacheEstimationTest, IgnoresLinearStateSlotsWithoutLinearAttention) {
  ModelArgs model_args = make_standard_args();
  KVCacheEstimateOptions options = make_estimate_options();
  options.max_linear_state_cache_slots = 32;

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.num_linear_attention_layers(), 0);
  EXPECT_EQ(capacity.num_linear_state_blocks(), 2);
  EXPECT_EQ(capacity.linear_cache_size_in_bytes(), 0);
}

TEST(KVCacheEstimationTest, UserIndexerCacheDtypeDirectlyControlsQuantization) {
  ModelArgs model_args = make_standard_args();
  model_args.model_type("unsupported_model")
      .index_n_heads(1)
      .index_head_dim(16);
  KVCacheEstimateOptions options = make_estimate_options();

  options.indexer_cache_dtype = "auto";
  KVCacheCapacity auto_capacity =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(auto_capacity.index_slot_size(), 32);
  EXPECT_FALSE(auto_capacity.enable_indexer_cache_quant());

  options.indexer_cache_dtype = "int8";
  KVCacheCapacity int8_capacity =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(int8_capacity.index_slot_size(), 20);
  EXPECT_TRUE(int8_capacity.enable_indexer_cache_quant());
}

TEST(KVCacheEstimationTest, IndexerScaleUsesLogicalCacheCapacity) {
  ModelArgs model_args = make_standard_args();
  model_args.model_type("deepseek_v32").index_n_heads(1).index_head_dim(16);
  KVCacheEstimateOptions options = make_estimate_options();
  options.indexer_cache_dtype = "int8";
  options.cache_size_in_bytes = 10 * 1024 * 1024;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.n_blocks(), 1107);
}

#if defined(USE_MLU)
TEST(KVCacheEstimationTest, SharedDsaLayersDoNotConsumeIndexerCacheBudget) {
  ModelArgs model_args = make_standard_args();
  model_args.model_type("glm_moe_dsa")
      .index_n_heads(1)
      .index_head_dim(16)
      .index_topk(8)
      .index_topk_pattern("FSFS");
  KVCacheEstimateOptions options = make_estimate_options();

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.num_full_attention_layers(), 4);
  EXPECT_EQ(capacity.num_indexer_layers(), 2);
  EXPECT_EQ(capacity.n_blocks(), 113);
}

#endif

namespace {

ModelArgs make_linear_attention_args(int64_t head_dim = 16) {
  ModelArgs model_args = make_standard_args();
  model_args.head_dim(head_dim)
      .full_attention_interval(2)
      .linear_num_key_heads(2)
      .linear_num_value_heads(2)
      .linear_key_head_dim(4)
      .linear_value_head_dim(8)
      .linear_conv_kernel_dim(3);
  return model_args;
}

KVCacheEstimateOptions make_linear_attention_options() {
  KVCacheEstimateOptions options = make_estimate_options();
  options.n_local_linear_k_heads = 2;
  options.n_local_linear_v_heads = 2;
  return options;
}

}  // namespace

TEST(KVCacheEstimationTest, ReservesLinearAttentionState) {
  ModelArgs model_args = make_linear_attention_args();
  KVCacheEstimateOptions options = make_linear_attention_options();

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.num_full_attention_layers(), 2);
  EXPECT_EQ(capacity.num_linear_attention_layers(), 2);
  EXPECT_EQ(capacity.num_linear_state_blocks(), 10);
  EXPECT_EQ(capacity.linear_slot_size(), 256);
  EXPECT_EQ(capacity.linear_cache_size_in_bytes(), 5120);
  EXPECT_EQ(capacity.n_blocks(), 254);
}

TEST(KVCacheEstimationTest, LinearStateCapacityVariants) {
  struct TestCase {
    const char* name;
    int64_t head_dim;
    int64_t cache_size_in_bytes;
    int64_t block_size;
    int64_t n_local_kv_heads;
    int64_t max_seqs_per_batch;
    bool enable_prefix_cache;
    int64_t max_linear_state_cache_slots;
    int64_t expected_num_linear_state_blocks;
    int64_t min_num_linear_state_blocks;
  };

  const std::vector<TestCase> test_cases = {
      {"PrefixCacheGrowsLinearStateCheckpointPool",
       /*head_dim=*/16,
       /*cache_size_in_bytes=*/64LL << 30,
       /*block_size=*/16,
       /*n_local_kv_heads=*/2,
       /*max_seqs_per_batch=*/200,
       /*enable_prefix_cache=*/true,
       /*max_linear_state_cache_slots=*/0,
       /*expected_num_linear_state_blocks=*/-1,
       /*min_num_linear_state_blocks=*/202},
      {"PrefixCacheUsesLinearStateMemoryRatio",
       /*head_dim=*/1,
       /*cache_size_in_bytes=*/1024 * 1024,
       /*block_size=*/1,
       /*n_local_kv_heads=*/1,
       /*max_seqs_per_batch=*/8,
       /*enable_prefix_cache=*/true,
       /*max_linear_state_cache_slots=*/0,
       /*expected_num_linear_state_blocks=*/970,
       /*min_num_linear_state_blocks=*/-1},
      {"NoPrefixCacheCapsLinearStateBlocksByBudget",
       /*head_dim=*/16,
       /*cache_size_in_bytes=*/1024 * 1024,
       /*block_size=*/16,
       /*n_local_kv_heads=*/2,
       /*max_seqs_per_batch=*/100000,
       /*enable_prefix_cache=*/false,
       /*max_linear_state_cache_slots=*/0,
       /*expected_num_linear_state_blocks=*/229,
       /*min_num_linear_state_blocks=*/-1},
      {"UnlimitedConcurrencyFallsBackToPaddingSlots",
       /*head_dim=*/16,
       /*cache_size_in_bytes=*/1024 * 1024,
       /*block_size=*/16,
       /*n_local_kv_heads=*/2,
       /*max_seqs_per_batch=*/0,
       /*enable_prefix_cache=*/false,
       /*max_linear_state_cache_slots=*/0,
       /*expected_num_linear_state_blocks=*/2,
       /*min_num_linear_state_blocks=*/-1},
      {"ExplicitLinearStateSlotsOverrideAutoSizing",
       /*head_dim=*/16,
       /*cache_size_in_bytes=*/64LL << 30,
       /*block_size=*/16,
       /*n_local_kv_heads=*/2,
       /*max_seqs_per_batch=*/200,
       /*enable_prefix_cache=*/true,
       /*max_linear_state_cache_slots=*/32,
       /*expected_num_linear_state_blocks=*/34,
       /*min_num_linear_state_blocks=*/-1},
  };

  for (const TestCase& test_case : test_cases) {
    SCOPED_TRACE(test_case.name);
    ModelArgs model_args = make_linear_attention_args(test_case.head_dim);
    KVCacheEstimateOptions options = make_linear_attention_options();
    options.cache_size_in_bytes = test_case.cache_size_in_bytes;
    options.block_size = test_case.block_size;
    options.n_local_kv_heads = test_case.n_local_kv_heads;
    options.max_seqs_per_batch = test_case.max_seqs_per_batch;
    options.enable_prefix_cache = test_case.enable_prefix_cache;
    options.max_linear_state_cache_slots =
        test_case.max_linear_state_cache_slots;

    KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

    if (test_case.expected_num_linear_state_blocks >= 0) {
      EXPECT_EQ(capacity.num_linear_state_blocks(),
                test_case.expected_num_linear_state_blocks);
    }
    if (test_case.min_num_linear_state_blocks >= 0) {
      EXPECT_GT(capacity.num_linear_state_blocks(),
                test_case.min_num_linear_state_blocks);
    }
  }
}

TEST(KVCacheEstimationTest, Qwen35MtpExpandsConvStateLen) {
  ModelArgs model_args = make_standard_args();
  model_args.model_type("qwen3_5")
      .full_attention_interval(2)
      .linear_num_key_heads(2)
      .linear_num_value_heads(2)
      .linear_key_head_dim(4)
      .linear_value_head_dim(8)
      .linear_conv_kernel_dim(3);
  KVCacheEstimateOptions options = make_estimate_options();
  options.n_local_linear_k_heads = 2;
  options.n_local_linear_v_heads = 2;
  options.num_speculative_tokens = 1;

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.linear_conv_state_len(), 3);
  EXPECT_EQ(capacity.linear_ssm_checkpoint_stride(), 2);
  EXPECT_EQ(capacity.linear_slot_size(), 448);
  EXPECT_EQ(capacity.linear_cache_size_in_bytes(), 8960);
}

TEST(KVCacheEstimationTest, Qwen35TextMtpUsesSsmCheckpointStride) {
  ModelArgs model_args = make_standard_args();
  model_args.model_type("qwen3_5_text")
      .full_attention_interval(2)
      .linear_num_key_heads(2)
      .linear_num_value_heads(2)
      .linear_key_head_dim(4)
      .linear_value_head_dim(8)
      .linear_conv_kernel_dim(3);
  KVCacheEstimateOptions options = make_estimate_options();
  options.n_local_linear_k_heads = 2;
  options.n_local_linear_v_heads = 2;
  options.num_speculative_tokens = 1;

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.linear_conv_state_len(), 3);
  EXPECT_EQ(capacity.linear_ssm_checkpoint_stride(), 2);
  EXPECT_EQ(capacity.linear_slot_size(), 448);
  EXPECT_EQ(capacity.linear_cache_size_in_bytes(), 8960);
}

TEST(KVCacheEstimationTest, EstimatesDeepSeekV4Pools) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 4, 128});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 2818048;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 19);
#if defined(USE_MLU)
  EXPECT_EQ(capacity.c4_count(), 64);
  EXPECT_EQ(capacity.c128_count(), 2);
  EXPECT_EQ(capacity.n_blocks(), 256);
#else
  EXPECT_EQ(capacity.c4_count(), 96);
  EXPECT_EQ(capacity.c128_count(), 3);
  EXPECT_EQ(capacity.n_blocks(), 384);
#endif
}

TEST(KVCacheEstimationTest, EstimatesDeepSeekV4DSparkSwaPool) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4_dspark")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 1, 1});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 2818048;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 19);
  EXPECT_EQ(capacity.c4_count(), 0);
  EXPECT_EQ(capacity.c128_count(), 0);
  EXPECT_EQ(capacity.n_blocks(), 1);
}

TEST(KVCacheEstimationTest,
     DeepSeekV4SpeculativeBudgetIncludesDSparkDraftSwaPool) {
  ModelArgs target_args;
  target_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 4, 128});
  ModelArgs draft_args;
  draft_args.model_type("deepseek_v4_dspark")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 1, 1});

  KVCacheEstimateOptions target_options;
  target_options.dtype = torch::kFloat32;
  target_options.kv_cache_dtype = "auto";
  target_options.cache_size_in_bytes = 2818048;
  target_options.block_size = 128;
  target_options.max_seqs_per_batch = 4;
  KVCacheEstimateOptions draft_options = target_options;
  draft_options.is_draft_engine = true;
  target_options.draft_model_args = &draft_args;
  target_options.draft_options = &draft_options;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(target_args, target_options);

  constexpr int64_t kDraftSwaBytes =
      /*layers=*/3 * /*swa_count=*/19 * /*block_size=*/128 *
      /*head_dim=*/16 * /*float32_bytes=*/4;
  EXPECT_LE(capacity.cache_size_in_bytes() + kDraftSwaBytes,
            target_options.cache_size_in_bytes);
  EXPECT_EQ(capacity.swa_count(), 19);
  EXPECT_EQ(capacity.c4_count(), 64);
  EXPECT_EQ(capacity.c128_count(), 2);
}

TEST(KVCacheEstimationTest,
     SpeculativeDecodePreservesDeepSeekV4PoolBlockCount) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 4, 128});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 2818048;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;

  KVCacheCapacity target_capacity =
      estimate_kv_cache_capacity(model_args, options);
  KVCacheCapacity draft_capacity = target_capacity;
  draft_capacity.n_blocks(target_capacity.n_blocks() * 4);

  EXPECT_EQ(estimate_speculative_kv_cache_blocks(
                target_capacity, draft_capacity, /*share_device=*/true),
            target_capacity.n_blocks());
}

TEST(KVCacheEstimationTest, SpeculativeDecodeCombinesStandardCacheCosts) {
  KVCacheCapacity target_capacity;
  target_capacity.cache_size_in_bytes(100000)
      .block_size(10)
      .slot_size(2)
      .num_full_attention_layers(1);
  KVCacheCapacity draft_capacity;
  draft_capacity.cache_size_in_bytes(100000)
      .block_size(10)
      .slot_size(1)
      .n_layers(2);

  EXPECT_EQ(estimate_speculative_kv_cache_blocks(
                target_capacity, draft_capacity, /*share_device=*/true),
            1666);
}

TEST(KVCacheEstimationTest, SpeculativeDecodeUsesDraftShapeForTp1Body) {
  KVCacheCapacity target_capacity;
  target_capacity.cache_size_in_bytes(100000)
      .block_size(10)
      .slot_size(2)
      .scale_slot_size(1)
      .index_slot_size(3)
      .num_full_attention_layers(2)
      .num_indexer_layers(1);
  KVCacheCapacity draft_capacity;
  draft_capacity.cache_size_in_bytes(100000)
      .block_size(10)
      .slot_size(4)
      .scale_slot_size(1)
      .index_slot_size(2)
      .num_indexer_layers(1)
      .n_layers(2);

  EXPECT_EQ(estimate_speculative_kv_cache_blocks(target_capacity,
                                                 draft_capacity,
                                                 /*share_device=*/true,
                                                 /*draft_body_uses_tp1=*/true),
            476);
}

}  // namespace xllm
