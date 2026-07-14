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
  options.max_concurrent_requests = 8;
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

#if defined(USE_MLU)
TEST(KVCacheEstimationTest,
     RdmaScalePaddingReducesCapacityToLargestAllocatableBlockCount) {
  ModelArgs model_args = make_standard_args();
  model_args.model_type("deepseek_v32").index_n_heads(1).index_head_dim(16);
  KVCacheEstimateOptions options = make_estimate_options();
  options.indexer_cache_dtype = "int8";
  options.cache_size_in_bytes = 10 * 1024 * 1024;

  options.enable_rdma_scale_padding = false;
  const KVCacheCapacity logical_capacity =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(logical_capacity.n_blocks(), 1107);

  options.enable_rdma_scale_padding = true;
  const KVCacheCapacity rdma_capacity =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(rdma_capacity.n_blocks(), 227);
}

TEST(KVCacheEstimationTest, RdmaScalePaddingFailsWhenOneBlockExceedsBudget) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ModelArgs model_args = make_standard_args();
  model_args.model_type("deepseek_v32").index_n_heads(1).index_head_dim(16);
  KVCacheEstimateOptions options = make_estimate_options();
  options.indexer_cache_dtype = "int8";
  options.cache_size_in_bytes = 8 * 1024 * 1024;
  options.enable_rdma_scale_padding = true;

  EXPECT_DEATH(
      { (void)estimate_kv_cache_capacity(model_args, options); },
      "no memory for one KV cache block with RDMA-registerable indexer scales");
}
#endif

TEST(KVCacheEstimationTest, ReservesLinearAttentionState) {
  ModelArgs model_args = make_standard_args();
  model_args.full_attention_interval(2)
      .linear_num_key_heads(2)
      .linear_num_value_heads(2)
      .linear_key_head_dim(4)
      .linear_value_head_dim(8)
      .linear_conv_kernel_dim(3);
  KVCacheEstimateOptions options = make_estimate_options();
  options.n_local_linear_k_heads = 2;
  options.n_local_linear_v_heads = 2;

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.num_full_attention_layers(), 2);
  EXPECT_EQ(capacity.num_linear_attention_layers(), 2);
  EXPECT_EQ(capacity.num_linear_state_blocks(), 10);
  EXPECT_EQ(capacity.linear_slot_size(), 256);
  EXPECT_EQ(capacity.linear_cache_size_in_bytes(), 5120);
  EXPECT_EQ(capacity.n_blocks(), 254);
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

}  // namespace xllm
