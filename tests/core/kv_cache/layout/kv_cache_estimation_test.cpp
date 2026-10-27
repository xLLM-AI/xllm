/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/kv_cache/layout/kv_cache_estimation.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/config/kv_cache_config.h"
#include "core/kv_cache/layout/deepseek_v4_cache_policy.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/kv_cache/layout/static_kv_cache_layout.h"
#include "framework/model/model_args.h"
#include "runtime/options.h"
#include "tests/core/kv_cache/dcp_indexer_test_fixture.h"

#if defined(USE_NPU)
#include "models/model_registry.h"
#endif

namespace xllm {
namespace {

class DcpIndexerEstimationTest : public test::DcpIndexerConfigTest {};

class VirtualMemoryPrefixEstimationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    KVCacheConfig::get_instance().enable_vmm_kvcache(true);
  }

  void TearDown() override { KVCacheConfig::get_instance() = saved_config_; }

 private:
  KVCacheConfig saved_config_ = KVCacheConfig::get_instance();
};

class SharedStaticRuntimeEstimationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    KVCacheConfig::get_instance().enable_shared_kv_cache(true);
  }

  void TearDown() override { KVCacheConfig::get_instance() = saved_config_; }

 private:
  KVCacheConfig saved_config_ = KVCacheConfig::get_instance();
};

ModelArgs make_standard_args() {
  ModelArgs model_args;
  model_args.n_layers(4).head_dim(16).n_heads(4).n_kv_heads(4);
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

TEST(KVCacheEstimationTest, RuntimeBudgetUsesMinimumAcrossWorkers) {
  ModelArgs model_args = make_standard_args();
  runtime::Options runtime_options;
  runtime_options.max_memory_utilization(0.8).block_size(16);

  KVCacheEstimateContext context;
  context.dtype = torch::kFloat16;
  context.world_size = 1;
  context.worker_memory = {{8 * 1024 * 1024, 10 * 1024 * 1024},
                           {16 * 1024 * 1024, 20 * 1024 * 1024}};

  const KVCacheCapacity capacity =
      KVCacheEstimator(model_args).estimate(runtime_options, context);

  // The workers have effective budgets of 6 MiB and 12 MiB respectively.
  EXPECT_EQ(capacity.cache_size_in_bytes(), 6 * 1024 * 1024);
}

TEST(KVCacheEstimationTest, RuntimeBudgetCapsBeforeVlmEncoderReservation) {
  ModelArgs model_args = make_standard_args();
  runtime::Options runtime_options;
  runtime_options.max_memory_utilization(0.8)
      .max_cache_size(5 * 1024 * 1024)
      .max_encoder_cache_size(1)
      .block_size(16);

  KVCacheEstimateContext context;
  context.dtype = torch::kFloat16;
  context.worker_memory = {
      {8 * 1024 * 1024, 10 * 1024 * 1024},
  };
  context.is_multimodal = true;

  const KVCacheCapacity capacity =
      KVCacheEstimator(model_args).estimate(runtime_options, context);

  // min(8 MiB - 20%, 5 MiB) - 1 MiB = 4 MiB.
  EXPECT_EQ(capacity.cache_size_in_bytes(), 4 * 1024 * 1024);
}

TEST(KVCacheEstimationTest, SleepingWeightsKeepTheirFutureDeviceCapacity) {
  ModelArgs model_args = make_standard_args();
  runtime::Options runtime_options;
  runtime_options.max_memory_utilization(0.9).block_size(16);

  constexpr int64_t kGiB = 1024LL * 1024 * 1024;
  KVCacheEstimateContext context;
  context.dtype = torch::kFloat16;
  context.worker_memory = {{80 * kGiB, 80 * kGiB}};
  // The residency coordinator includes TP rounding and its safety pages.
  context.reserved_memory_bytes = 10 * kGiB;

  const KVCacheCapacity capacity =
      KVCacheEstimator(model_args).estimate(runtime_options, context);

  // Keep both the 10 GiB weight demand and the 8 GiB utilization headroom.
  EXPECT_EQ(capacity.cache_size_in_bytes(), 62 * kGiB);
}

TEST(KVCacheEstimationTest, SleepingWeightReservationPrecedesExplicitKvCap) {
  ModelArgs model_args = make_standard_args();
  runtime::Options runtime_options;
  runtime_options.max_memory_utilization(1.0)
      .max_cache_size(4 * 1024 * 1024)
      .block_size(16);

  KVCacheEstimateContext context;
  context.dtype = torch::kFloat16;
  context.worker_memory = {{16 * 1024 * 1024, 16 * 1024 * 1024}};
  context.reserved_memory_bytes = 8 * 1024 * 1024;

  const KVCacheCapacity capacity =
      KVCacheEstimator(model_args).estimate(runtime_options, context);

  EXPECT_EQ(capacity.cache_size_in_bytes(), 4 * 1024 * 1024);
}

TEST(KVCacheEstimationTest, VirtualMemoryBudgetBypassesWorkerAndRuntimeCaps) {
  ModelArgs model_args = make_standard_args();
  runtime::Options runtime_options;
  runtime_options.max_memory_utilization(0.1).max_cache_size(1024).block_size(
      16);

  KVCacheEstimateContext context;
  context.dtype = torch::kFloat16;
  context.virtual_memory_cache_size = 7 * 1024 * 1024;

  const KVCacheCapacity capacity =
      KVCacheEstimator(model_args).estimate(runtime_options, context);

  EXPECT_EQ(capacity.cache_size_in_bytes(), 7 * 1024 * 1024);
}

TEST(KVCacheEstimationTest, RuntimeEstimatorDerivesTensorParallelHeads) {
  ModelArgs model_args = make_standard_args();
  model_args.n_heads(8).n_kv_heads(4);
  runtime::Options runtime_options;
  runtime_options.block_size(16);

  KVCacheEstimateContext context;
  context.dtype = torch::kFloat16;
  context.world_size = 2;
  context.virtual_memory_cache_size = 1024 * 1024;

  const KVCacheCapacity capacity =
      KVCacheEstimator(model_args).estimate(runtime_options, context);

  // 2 local KV heads: 2 (K/V) * 2-byte dtype * 16 head dim * 2 heads.
  EXPECT_EQ(capacity.slot_size(), 128);
}

TEST(KVCacheEstimationTest, DraftTp1OverridesTensorParallelHeadDerivation) {
  ModelArgs model_args = make_standard_args();
  model_args.n_heads(8).n_kv_heads(4);
  runtime::Options runtime_options;
  runtime_options.block_size(16).enable_mtp_draft_body_tp1(true).is_draft_model(
      true);

  KVCacheEstimateContext context;
  context.dtype = torch::kFloat16;
  context.world_size = 2;
  context.virtual_memory_cache_size = 1024 * 1024;

  const KVCacheCapacity capacity =
      KVCacheEstimator(model_args).estimate(runtime_options, context);

  // TP1 draft uses all four KV heads instead of the target's local two.
  EXPECT_EQ(capacity.slot_size(), 256);
}

TEST(KVCacheEstimationTest, RuntimeEstimatorRejectsMissingWorkerBudget) {
  ModelArgs model_args = make_standard_args();
  runtime::Options runtime_options;
  KVCacheEstimateContext context;

  EXPECT_DEATH(
      (void)KVCacheEstimator(model_args).estimate(runtime_options, context),
      "requires worker memory snapshots");
}

TEST(KVCacheEstimationTest, RuntimeEstimatorRejectsNonPositiveWorkerBudget) {
  ModelArgs model_args = make_standard_args();
  runtime::Options runtime_options;
  KVCacheEstimateContext context;
  context.worker_memory = {{0, 0}};

  EXPECT_DEATH(
      (void)KVCacheEstimator(model_args).estimate(runtime_options, context),
      "Available kv cache size must be greater than 0");
}

TEST(KVCacheEstimationTest, RuntimeEstimatorDerivesLinearHeadCounts) {
  ModelArgs model_args = make_standard_args();
  model_args.full_attention_interval(2)
      .linear_num_key_heads(2)
      .linear_num_value_heads(2)
      .linear_key_head_dim(4)
      .linear_value_head_dim(8)
      .linear_conv_kernel_dim(3);
  runtime::Options runtime_options;
  runtime_options.block_size(16);

  KVCacheEstimateContext context;
  context.dtype = torch::kFloat16;
  context.world_size = 2;
  context.virtual_memory_cache_size = 2 * 1024 * 1024;

  const KVCacheCapacity capacity =
      KVCacheEstimator(model_args).estimate(runtime_options, context);

  // One local key/value head per linear-attention layer after TP sharding.
  EXPECT_EQ(capacity.linear_slot_size(), 128);
  EXPECT_EQ(capacity.num_linear_attention_layers(), 2);
  EXPECT_GT(capacity.num_linear_state_blocks(), 0);
}

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

TEST(KVCacheEstimationTest, EmbeddingContextSharesTheAvailableMemoryBudget) {
  ModelArgs model_args = make_standard_args();
  KVCacheEstimateOptions options = make_estimate_options();
  options.embedding_context_bytes_per_block = 2048;
  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(capacity.n_blocks(), 102);
  EXPECT_LE(capacity.n_blocks() * (8192 + 2048), options.cache_size_in_bytes);
  EXPECT_GT((capacity.n_blocks() + 1) * (8192 + 2048),
            options.cache_size_in_bytes);
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

#if defined(USE_NPU)
  ModelArgs kpool_args = make_standard_args();
  kpool_args.model_type("glm5_next")
      .index_n_heads(32)
      .index_head_dim(128)
      .index_kpool(4)
      .index_kpool_compress(true);
  KVCacheEstimateOptions kpool_options = make_estimate_options();
  kpool_options.block_size = 128;
  kpool_options.dtype = torch::kBFloat16;

  const KVCacheCapacity packed_capacity =
      estimate_kv_cache_capacity(kpool_args, kpool_options);
  EXPECT_EQ(packed_capacity.kpool_layout(), KPoolCacheLayout::PACKED);

  kpool_args.index_kpool_always_select_tail(true);
  kpool_options.num_speculative_tokens = 3;
  const KVCacheCapacity compressed_capacity =
      estimate_kv_cache_capacity(kpool_args, kpool_options);
  EXPECT_EQ(compressed_capacity.kpool_layout(),
            KPoolCacheLayout::COMPRESSED_WITH_TAIL);
  EXPECT_EQ(compressed_capacity.kpool_tail_len(), 7);

  kpool_args.model_type("deepseek_v32").index_n_heads(1).index_head_dim(16);
  const KVCacheCapacity non_target_capacity =
      estimate_kv_cache_capacity(kpool_args, kpool_options);
  EXPECT_EQ(non_target_capacity.kpool_layout(), KPoolCacheLayout::PACKED);

  kpool_args.model_type("glm5_next")
      .index_n_heads(32)
      .index_head_dim(128)
      .index_kpool(4)
      .index_kpool_compress(true)
      .index_kpool_always_select_tail(true);
  kpool_options.indexer_cache_dtype = "int8";

  EXPECT_DEATH((void)estimate_kv_cache_capacity(kpool_args, kpool_options),
               "KPool cache layouts do not support.*int8");
#endif
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

#if defined(USE_MLU) || defined(USE_NPU)
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

TEST(KVCacheEstimationTest, GlmSharedIndexerFreqAllocatesOnlyFullLayers) {
  ModelArgs model_args = make_standard_args();
  model_args.model_type("glm_moe_dsa")
      .n_layers(78)
      .index_n_heads(1)
      .index_head_dim(16)
      .index_topk(8)
      .index_topk_freq(4)
      .index_skip_topk_offset(3);

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, make_estimate_options());

  EXPECT_EQ(capacity.num_full_attention_layers(), 78);
  EXPECT_EQ(capacity.num_indexer_layers(), 21);
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

TEST(KVCacheEstimationTest, SharedStaticCapacityUsesTheWholeArenaBudget) {
  ModelArgs model_args = make_linear_attention_args();
  model_args.linear_value_head_dim(4).mamba_ssm_dtype("float32");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.enable_shared_kv_cache = true;
  // Each of two banks mirrors a 2 KiB Full bundle and 128-byte SSM rows.
  // Conv rows align within the 2 KiB slice instead of inflating its LCM.
  constexpr int64_t kSlabBytes = 4096;
  options.cache_size_in_bytes = 25 * kSlabBytes + kSlabBytes - 1;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.shared_slot_bytes(), kSlabBytes);
  EXPECT_EQ(capacity.shared_slab_count(), 25);
  EXPECT_EQ(capacity.n_blocks(), 211);
  EXPECT_EQ(capacity.num_linear_state_blocks(), 211);
  EXPECT_EQ(capacity.cache_size_in_bytes(), 25 * kSlabBytes);
  EXPECT_EQ(capacity.linear_cache_size_in_bytes(), 0);
}

TEST_F(SharedStaticRuntimeEstimationTest, VlmTp4UsesOnlyDecoderCacheGeometry) {
  ModelArgs args;
  args.model_type("qwen3_5")
      .hidden_size(4096)
      .n_layers(4)
      .n_heads(32)
      .n_kv_heads(8)
      .head_dim(128)
      .layer_types({"linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention"})
      .linear_num_key_heads(16)
      .linear_num_value_heads(32)
      .linear_key_head_dim(128)
      .linear_value_head_dim(128)
      .linear_conv_kernel_dim(4)
      .mamba_ssm_dtype("float32")
      .mm_num_hidden_layers(27)
      .mm_hidden_size(1152)
      .mm_projection_dim(4096);
  runtime::Options options;
  options.backend("vlm")
      .block_size(128)
      .num_speculative_tokens(4)
      .max_seqs_per_batch(4)
      .max_tokens_per_chunk_for_prefill(512);
  KVCacheEstimateContext context;
  context.dtype = torch::kBFloat16;
  context.world_size = 4;
  context.is_multimodal = true;
  context.virtual_memory_cache_size = 256 * 1024 * 1024;
  const KVCacheCapacity vlm = KVCacheEstimator(args).estimate(options, context);
  const KVCacheShape vlm_shape(vlm, args, /*world_size=*/4);

  args.model_type("qwen3_5_text").mm_num_hidden_layers(0).mm_hidden_size(0);
  options.backend("llm");
  context.is_multimodal = false;
  const KVCacheCapacity text =
      KVCacheEstimator(args).estimate(options, context);
  const KVCacheShape text_shape(text, args, /*world_size=*/4);

  EXPECT_EQ(vlm.n_layers(), 4);
  EXPECT_EQ(vlm.num_full_attention_layers(), 1);
  EXPECT_EQ(vlm.num_linear_attention_layers(), 3);
  EXPECT_EQ(vlm.linear_ssm_checkpoint_stride(), 5);
  EXPECT_EQ(vlm.shared_slot_bytes(), text.shared_slot_bytes());
  EXPECT_EQ(vlm.shared_slab_count(), text.shared_slab_count());
  EXPECT_EQ(vlm_shape.shared_slab_count(), vlm.shared_slab_count());
  EXPECT_EQ(vlm.shared_linear_state_slots(), text.shared_linear_state_slots());
  EXPECT_EQ(vlm.n_blocks(), text.n_blocks());
  EXPECT_EQ(vlm_shape.key_cache_shape(), text_shape.key_cache_shape());
  EXPECT_EQ(vlm_shape.value_cache_shape(), text_shape.value_cache_shape());
  EXPECT_EQ(vlm_shape.conv_cache_shape(), text_shape.conv_cache_shape());
  EXPECT_EQ(vlm_shape.ssm_cache_shape(), text_shape.ssm_cache_shape());
}

TEST(KVCacheEstimationTest, SharedStaticIgnoresSeparateMegaGdnSlotLimit) {
  ModelArgs model_args = make_linear_attention_args();
  model_args.model_type("qwen3_5_text")
      .linear_value_head_dim(4)
      .mamba_ssm_dtype("float32");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.enable_shared_kv_cache = true;
  options.linear_state_cache_block_limit = 1024;
  options.cache_size_in_bytes = 4096 * 4096;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.shared_slab_count(), 4096);
  EXPECT_EQ(capacity.n_blocks(), 37431);
  EXPECT_EQ(capacity.num_linear_state_blocks(), 37431);
  EXPECT_EQ(capacity.cache_size_in_bytes(), options.cache_size_in_bytes);
}

TEST(KVCacheEstimationTest, SharedStaticCostsMatchTheStorageLayout) {
  ModelArgs model_args = make_linear_attention_args();
  model_args.linear_value_head_dim(4).mamba_ssm_dtype("float32");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.enable_shared_kv_cache = true;

  StaticKVCacheLayoutSpec spec;
  spec.full_row_bytes = 1024;
  spec.conv_row_bytes = 96;
  spec.ssm_row_bytes = 128;
  spec.dtype = torch::kFloat16;
  spec.ssm_dtype = torch::kFloat32;
  spec.full_layer_ids = {1, 3};
  spec.linear_layer_ids = {0, 2};
  const StaticKVCacheLayout layout(std::move(spec));
  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.shared_slot_bytes(), layout.slab_bytes());
  EXPECT_EQ(capacity.shared_reserved_slab_count(), layout.num_reserved_slabs());
  EXPECT_EQ(capacity.shared_linear_state_slots(), layout.num_linear_groups());
  EXPECT_EQ(capacity.shared_linear_state_slots(), 4);
  EXPECT_EQ(capacity.cache_size_in_bytes(),
            capacity.shared_slab_count() * layout.slab_bytes());
}

TEST(KVCacheEstimationTest, SharedStaticDoesNotReserveLinearMemoryTwice) {
  ModelArgs model_args = make_linear_attention_args();
  model_args.linear_value_head_dim(4).mamba_ssm_dtype("float32");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.enable_shared_kv_cache = true;
  options.cache_size_in_bytes = 25 * 4096;
  const KVCacheCapacity base = estimate_kv_cache_capacity(model_args, options);

  options.max_seqs_per_batch = 64;
  options.enable_prefix_cache = true;
  const KVCacheCapacity cached =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(base.shared_slab_count(), 25);
  EXPECT_EQ(base.n_blocks(), 211);
  EXPECT_EQ(cached.n_blocks(), base.n_blocks());
  EXPECT_EQ(cached.cache_size_in_bytes(), options.cache_size_in_bytes);
  EXPECT_EQ(cached.linear_cache_size_in_bytes(), 0);
}

TEST(KVCacheEstimationTest, SharedStaticRequiresOneHybridWorkingSet) {
  ModelArgs model_args = make_linear_attention_args();
  model_args.linear_value_head_dim(4).mamba_ssm_dtype("float32");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.enable_shared_kv_cache = true;
  options.cache_size_in_bytes = 4 * 4096;

  EXPECT_DEATH((void)estimate_kv_cache_capacity(model_args, options),
               "cannot fit padding and one hybrid sequence");

  // Scratch and padding plus Full, Conv and SSM mirrored groups.
  options.cache_size_in_bytes = 5 * 4096;
  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(capacity.shared_min_slab_count(), 5);
  EXPECT_EQ(capacity.shared_slab_count(), 5);
  EXPECT_EQ(capacity.n_blocks(), 28);
}

TEST(KVCacheEstimationTest,
     SharedStaticRequiresSeparateReadPaddingWhenFullScratchFillsASlab) {
  ModelArgs model_args = make_linear_attention_args();
  model_args.linear_value_head_dim(4).linear_num_value_heads(4).mamba_ssm_dtype(
      "float32");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.enable_shared_kv_cache = true;
  options.n_local_linear_v_heads = 4;
  // Each bank's 2 KiB Full bundle fills one slice, so writable scratch
  // and immutable padding require two reserved mirrored groups.
  constexpr int64_t kSlabBytes = 4096;
  options.cache_size_in_bytes = 4 * kSlabBytes;
  EXPECT_DEATH((void)estimate_kv_cache_capacity(model_args, options),
               "cannot fit padding and one hybrid sequence");

  // The minimum needs one mirrored group each for Full, Conv and SSM.
  options.cache_size_in_bytes = 5 * kSlabBytes;
  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(capacity.shared_slot_bytes(), kSlabBytes);
  EXPECT_EQ(capacity.shared_reserved_slab_count(), 2);
  EXPECT_EQ(capacity.shared_min_slab_count(), 5);
  EXPECT_EQ(capacity.shared_slab_count(), 5);
  // Three groups hold 12 KiB; each two-layer Linear state consumes 768 bytes.
  EXPECT_EQ(capacity.n_blocks(), 17);
  EXPECT_EQ(capacity.num_linear_state_blocks(), 17);

  StaticKVCacheLayoutSpec spec;
  spec.num_slots = capacity.n_blocks();
  spec.num_slabs = capacity.shared_slab_count();
  spec.full_row_bytes = 1024;
  spec.conv_row_bytes = 128;
  spec.ssm_row_bytes = 256;
  spec.dtype = torch::kFloat16;
  spec.ssm_dtype = torch::kFloat32;
  spec.full_layer_ids = {1, 3};
  spec.linear_layer_ids = {0, 2};
  const StaticKVCacheLayout layout(std::move(spec));
  EXPECT_EQ(layout.num_reserved_slabs(), 2u);
  EXPECT_EQ(layout.full_read_padding_row(), 2u);
  EXPECT_TRUE(layout.can_fit(/*full_blocks=*/1, /*linear_states=*/1));
  EXPECT_FALSE(layout.can_fit(/*full_blocks=*/2, /*linear_states=*/1));
}

TEST(KVCacheEstimationTest, SharedStaticRejectsIndependentLinearSlotOverride) {
  ModelArgs model_args = make_linear_attention_args();
  model_args.linear_value_head_dim(4).mamba_ssm_dtype("float32");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.enable_shared_kv_cache = true;
  options.max_linear_state_cache_slots = 2;

  EXPECT_DEATH((void)estimate_kv_cache_capacity(model_args, options),
               "derives linear capacity from its arena");
}

TEST(KVCacheEstimationTest, SharedStaticBoundsDenseInt32TokenSlots) {
  ModelArgs model_args = make_linear_attention_args(/*head_dim=*/1);
  model_args.linear_value_head_dim(4).mamba_ssm_dtype("float32");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.enable_shared_kv_cache = true;
  options.n_local_kv_heads = 1;
  options.cache_size_in_bytes = int64_t{256} << 30;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  // Eight 32-byte Full rows occupy each bank-local slice. One element per
  // token makes the int32 token slot limit stricter than the FIA byte span.
  constexpr int64_t kSlabBytes = 512;
  const int64_t tokens_per_slab = 8 * options.block_size;
  const int64_t max_index = std::numeric_limits<int32_t>::max();
  const int64_t expected_slabs = max_index / tokens_per_slab;
  EXPECT_EQ(capacity.shared_slot_bytes(), kSlabBytes);
  EXPECT_EQ(capacity.shared_slab_count(), expected_slabs);
  EXPECT_LE(expected_slabs * tokens_per_slab, max_index);
  EXPECT_GT((expected_slabs + 1) * tokens_per_slab, max_index);
  EXPECT_EQ(capacity.cache_size_in_bytes(), expected_slabs * kSlabBytes);
  EXPECT_GT(capacity.n_blocks(), capacity.shared_slab_count());
  EXPECT_LT(capacity.cache_size_in_bytes(), options.cache_size_in_bytes);
}

TEST(KVCacheEstimationTest, SharedStaticBoundsFullBankFiaElements) {
  ModelArgs model_args = make_linear_attention_args();
  model_args.linear_value_head_dim(4).mamba_ssm_dtype("float32");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.enable_shared_kv_cache = true;
  options.cache_size_in_bytes = int64_t{256} << 30;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  constexpr int64_t kElementsPerFullRow = 512;
  constexpr int64_t kFullRowsPerSlice = 2;
  constexpr int64_t kFiaElementSpan = int64_t{1} << 32;
  constexpr int64_t kSlabBytes = 4096;
  const int64_t expected_slabs =
      kFiaElementSpan / kElementsPerFullRow / kFullRowsPerSlice;
  EXPECT_EQ(capacity.shared_slab_count(), expected_slabs);
  EXPECT_LE(expected_slabs * kFullRowsPerSlice * kElementsPerFullRow,
            kFiaElementSpan);
  EXPECT_GT((expected_slabs + 1) * kFullRowsPerSlice * kElementsPerFullRow,
            kFiaElementSpan);
  EXPECT_EQ(capacity.cache_size_in_bytes(), expected_slabs * kSlabBytes);
  EXPECT_LT(capacity.cache_size_in_bytes(), options.cache_size_in_bytes);
}

TEST(KVCacheEstimationTest, SharedStaticBoundsDenseConvIndices) {
  ModelArgs model_args = make_linear_attention_args(/*head_dim=*/128);
  model_args.n_layers(64)
      .linear_num_key_heads(1)
      .linear_num_value_heads(1)
      .linear_key_head_dim(8)
      .linear_value_head_dim(8)
      .mamba_ssm_dtype("float32");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.enable_shared_kv_cache = true;
  options.n_local_linear_k_heads = 1;
  options.n_local_linear_v_heads = 1;
  options.cache_size_in_bytes = int64_t{256} << 40;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  // Thirty-two banks each hold a 16 KiB slice. Conv rows span the arena
  // with ceil(512 KiB / 96) native rows per mirrored group.
  constexpr int64_t kSlabBytes = 524288;
  constexpr int64_t kConvRowsPerSlab = 5462;
  const int64_t max_index = std::numeric_limits<int32_t>::max();
  const int64_t expected_slabs = max_index / kConvRowsPerSlab;
  EXPECT_EQ(capacity.shared_slot_bytes(), kSlabBytes);
  EXPECT_EQ(capacity.shared_slab_count(), expected_slabs);
  EXPECT_LE(expected_slabs * kConvRowsPerSlab, max_index);
  EXPECT_GT((expected_slabs + 1) * kConvRowsPerSlab, max_index);
  EXPECT_EQ(capacity.cache_size_in_bytes(), expected_slabs * kSlabBytes);
}

TEST(KVCacheEstimationTest, SharedStaticQwen38MtpSubdividesTypedSlabs) {
  ModelArgs model_args = make_linear_attention_args(/*head_dim=*/256);
  model_args.model_type("qwen3_5_text")
      .n_layers(64)
      .n_heads(24)
      .n_kv_heads(4)
      .full_attention_interval(4)
      .linear_num_key_heads(16)
      .linear_num_value_heads(48)
      .linear_key_head_dim(128)
      .linear_value_head_dim(128)
      .linear_conv_kernel_dim(4)
      .mamba_ssm_dtype("float32");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.enable_shared_kv_cache = true;
  options.dtype = torch::kBFloat16;
  options.world_size = 2;
  options.block_size = 128;
  options.n_local_kv_heads = 2;
  options.n_local_linear_k_heads = 8;
  options.n_local_linear_v_heads = 24;
  options.num_speculative_tokens = 4;
  // Sixteen banks mirror 7.5 MiB slices; Conv alignment does not inflate
  // the group beyond 120 MiB. Each slice contains thirty Full bundles.
  constexpr int64_t kSlabBytes = 125829120;
  options.cache_size_in_bytes = 25 * kSlabBytes + kSlabBytes - 1;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.shared_slot_bytes(), kSlabBytes);
  EXPECT_EQ(capacity.shared_slab_count(), 25);
  EXPECT_EQ(capacity.shared_min_slab_count(), 7);
  EXPECT_EQ(capacity.shared_linear_state_slots(), 96);
  EXPECT_EQ(capacity.linear_conv_state_len(), 7);
  EXPECT_EQ(capacity.linear_ssm_checkpoint_stride(), 5);
  EXPECT_EQ(capacity.n_blocks(), 691);
  EXPECT_EQ(capacity.num_linear_state_blocks(), 691);
  EXPECT_EQ(capacity.cache_size_in_bytes(), 25 * kSlabBytes);
}

TEST(KVCacheEstimationTest, SharedStaticQwen38MtpBoundsFiaBankAddressSpan) {
  ModelArgs model_args = make_linear_attention_args(/*head_dim=*/256);
  model_args.model_type("qwen3_5_text")
      .n_layers(64)
      .n_heads(24)
      .n_kv_heads(4)
      .full_attention_interval(4)
      .linear_num_key_heads(16)
      .linear_num_value_heads(48)
      .linear_key_head_dim(128)
      .linear_value_head_dim(128)
      .linear_conv_kernel_dim(4)
      .mamba_ssm_dtype("float32");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.enable_shared_kv_cache = true;
  options.dtype = torch::kBFloat16;
  options.world_size = 2;
  options.block_size = 128;
  options.n_local_kv_heads = 2;
  options.n_local_linear_k_heads = 8;
  options.n_local_linear_v_heads = 24;
  options.num_speculative_tokens = 4;
  constexpr int64_t kSliceBytes = 7864320;
  constexpr int64_t kSlabBytes = 16 * kSliceBytes;
  constexpr int64_t kElementsPerFullRow = 65536;
  constexpr int64_t kFullPagesPerSlice = 60;
  constexpr int64_t kFiaElementSpan = int64_t{1} << 32;
  constexpr int64_t kExpectedSlabs =
      kFiaElementSpan / kElementsPerFullRow / kFullPagesPerSlice;
  // The pre-bank implementation logged this Qwen run as 319 slabs at
  // 55,050,240 bytes each. Preserve that byte budget when checking the
  // mirrored layout: one new group is 16 bank slices, so 139 groups fit.
  constexpr int64_t kHistoricalArenaBytes = 17561026560;
  constexpr int64_t kExpectedProductionSlabs =
      kHistoricalArenaBytes / kSlabBytes;
  options.cache_size_in_bytes = kHistoricalArenaBytes;

  const KVCacheCapacity production_capacity =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(kExpectedProductionSlabs, 139);
  EXPECT_EQ(production_capacity.shared_slab_count(), kExpectedProductionSlabs);
  EXPECT_EQ(production_capacity.cache_size_in_bytes(),
            kExpectedProductionSlabs * kSlabBytes);
  EXPECT_LE(kExpectedProductionSlabs * kFullPagesPerSlice * kElementsPerFullRow,
            kFiaElementSpan);

  options.cache_size_in_bytes = int64_t{256} << 30;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  // The 16 Full layers form 16 mirrored banks. Each bank exposes 60 native
  // Full pages per 7.5 MiB slice, so 1092 groups are the largest FIA-safe
  // arena before a 32-bit element offset reaches 2^32.
  EXPECT_EQ(kExpectedSlabs, 1092);
  EXPECT_EQ(capacity.shared_slab_count(), kExpectedSlabs);
  EXPECT_EQ(capacity.shared_slot_bytes(), kSlabBytes);
  EXPECT_EQ(capacity.shared_reserved_slab_count(), 2);
  EXPECT_EQ(capacity.shared_min_slab_count(), 7);
  EXPECT_EQ(capacity.cache_size_in_bytes(), kExpectedSlabs * kSlabBytes);
  EXPECT_EQ(capacity.num_full_attention_layers(), 16);
  EXPECT_EQ(capacity.num_linear_attention_layers(), 48);

  StaticKVCacheLayoutSpec spec;
  spec.num_slabs = static_cast<size_t>(kExpectedSlabs);
  spec.full_row_bytes = 131072;
  spec.conv_row_bytes = 71680;
  spec.ssm_row_bytes = 1572864;
  spec.checkpoint_stride = 5;
  spec.dtype = torch::kBFloat16;
  spec.ssm_dtype = torch::kFloat32;
  for (int64_t layer = 0; layer < 64; ++layer) {
    if ((layer + 1) % 4 == 0) {
      spec.full_layer_ids.emplace_back(layer);
    } else {
      spec.linear_layer_ids.emplace_back(layer);
    }
  }
  const StaticKVCacheLayout layout(std::move(spec));
  EXPECT_EQ(layout.bank_count(), 16u);
  EXPECT_EQ(layout.full_layers_per_bank(), 1u);
  EXPECT_EQ(layout.linear_layers_per_bank(), 3u);
  EXPECT_EQ(layout.slice_bytes(), static_cast<size_t>(kSliceBytes));
  EXPECT_EQ(layout.slab_bytes(), static_cast<size_t>(kSlabBytes));
  EXPECT_EQ(layout.full_pages_per_slab(),
            static_cast<size_t>(kFullPagesPerSlice));
  EXPECT_EQ(layout.max_slab_count(), static_cast<size_t>(kExpectedSlabs));
}

TEST(KVCacheEstimationTest, SharedStaticRejectsQuantizedFullPageGeometry) {
  ModelArgs model_args = make_linear_attention_args();
  model_args.linear_value_head_dim(4).mamba_ssm_dtype("float32");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.enable_shared_kv_cache = true;
  options.kv_cache_dtype = "fp8_e4m3";

  EXPECT_DEATH((void)estimate_kv_cache_capacity(model_args, options),
               "requires model-dtype Full KV rows");
}

TEST(KVCacheEstimationTest, SharedStaticFlagPreservesDefaultSeparateBudgets) {
  ModelArgs model_args = make_linear_attention_args();
  KVCacheEstimateOptions options = make_linear_attention_options();
  ASSERT_FALSE(options.enable_shared_kv_cache);

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.shared_slot_bytes(), 0);
  EXPECT_EQ(capacity.num_linear_state_blocks(), 10);
  EXPECT_EQ(capacity.linear_cache_size_in_bytes(), 5120);
  EXPECT_EQ(capacity.n_blocks(), 254);
}

TEST(KVCacheEstimationTest, SharedStaticReservesEveryMtpCheckpoint) {
  ModelArgs model_args = make_linear_attention_args();
  model_args.model_type("qwen3_5_text")
      .linear_value_head_dim(4)
      .mamba_ssm_dtype("float32");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.enable_shared_kv_cache = true;
  options.num_speculative_tokens = 4;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  StaticKVCacheLayoutSpec spec;
  spec.dtype = torch::kFloat16;
  spec.ssm_dtype = torch::kFloat32;
  spec.full_row_bytes = 1024;
  spec.conv_row_bytes = 288;
  spec.ssm_row_bytes = 128;
  spec.checkpoint_stride = 5;
  spec.full_layer_ids = {1, 3};
  spec.linear_layer_ids = {0, 2};
  const StaticKVCacheLayout layout(std::move(spec));
  EXPECT_EQ(capacity.linear_conv_state_len(), 6);
  EXPECT_EQ(capacity.linear_ssm_checkpoint_stride(), 5);
  EXPECT_EQ(capacity.shared_slot_bytes(), layout.slab_bytes());
  EXPECT_EQ(capacity.shared_reserved_slab_count(), layout.num_reserved_slabs());
  EXPECT_EQ(capacity.n_blocks(), capacity.num_linear_state_blocks());
  EXPECT_LE(capacity.cache_size_in_bytes(), options.cache_size_in_bytes);
}

TEST_F(SharedStaticRuntimeEstimationTest, FullOnlyDraftKeepsDenseKvRows) {
  ModelArgs model_args = make_standard_args();
  model_args.model_type("qwen3_5_mtp")
      .n_layers(1)
      .num_nextn_predict_layers(1)
      .layer_types({"full_attention"});
  runtime::Options options;
  options.block_size(16)
      .is_draft_model(true)
      .speculative_algorithm("mtp")
      .num_speculative_tokens(4);
  KVCacheEstimateContext context;
  context.dtype = torch::kFloat16;
  context.virtual_memory_cache_size = 1024 * 1024;

  const KVCacheCapacity capacity =
      KVCacheEstimator(model_args).estimate(options, context);

  EXPECT_EQ(capacity.shared_slot_bytes(), 0);
  EXPECT_EQ(capacity.num_full_attention_layers(), 1);
  EXPECT_EQ(capacity.num_linear_attention_layers(), 0);
  EXPECT_EQ(capacity.linear_cache_size_in_bytes(), 0);
  EXPECT_EQ(capacity.n_blocks(), 256);
}
TEST(KVCacheEstimationTest, MultimodalRuntimeReservesMtpCheckpoints) {
  ModelArgs model_args = make_linear_attention_args();
  model_args.model_type("qwen3_5");
  runtime::Options runtime_options;
  runtime_options.block_size(16)
      .max_seqs_per_batch(8)
      .max_linear_state_cache_slots(16)
      .num_speculative_tokens(4);

  KVCacheEstimateContext context;
  context.dtype = torch::kFloat16;
  context.world_size = 1;
  context.virtual_memory_cache_size = 1024 * 1024;
  context.is_multimodal = true;

  const KVCacheCapacity capacity =
      KVCacheEstimator(model_args).estimate(runtime_options, context);

  EXPECT_EQ(capacity.linear_ssm_checkpoint_stride(), 5);
  EXPECT_EQ(capacity.linear_conv_state_len(), 6);

  runtime_options.num_speculative_tokens(0);
  const KVCacheCapacity without_mtp =
      KVCacheEstimator(model_args).estimate(runtime_options, context);
  EXPECT_EQ(without_mtp.linear_ssm_checkpoint_stride(), 1);
  EXPECT_EQ(without_mtp.linear_conv_state_len(), 2);
}

#if defined(USE_MLU)
TEST(KVCacheEstimationTest, KPoolReservesSpeculativeTailWithRequestState) {
  ModelArgs args = make_linear_attention_args();
  args.index_n_heads(1).index_head_dim(16).index_kpool(4).index_kpool_compress(
      true);
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.dtype = torch::kBFloat16;
  options.num_speculative_tokens = 5;
  const KVCacheCapacity capacity = estimate_kv_cache_capacity(args, options);
  // Two recurrent layers and two indexer layers. Each tail keeps K + W
  // BF16 key/gate rows; W includes the verify base token and prelaunch room.
  EXPECT_EQ(capacity.num_linear_state_blocks(), 10);
  EXPECT_EQ(capacity.linear_cache_size_in_bytes(), 25600);
  EXPECT_EQ(
      capacity.num_linear_attention_layers() * capacity.linear_slot_size(),
      512);
  EXPECT_EQ(capacity.num_indexer_layers() * capacity.kpool_tail_slot_size(),
            2048);
  EXPECT_EQ(capacity.kpool_tail_len(), 16);
  EXPECT_LE(capacity.n_blocks() * 16 * (2 * 128 + 2 * 8) +
                capacity.linear_cache_size_in_bytes(),
            options.cache_size_in_bytes);
  args.num_nextn_predict_layers(2);
  const KVCacheCapacity speculative = estimate_kv_cache_capacity(args, options);
  EXPECT_EQ(speculative.num_full_attention_layers(), 2);
  EXPECT_EQ(speculative.num_indexer_layers(), 2);
  EXPECT_EQ(speculative.linear_cache_size_in_bytes(),
            capacity.linear_cache_size_in_bytes());
  EXPECT_EQ(speculative.n_blocks(), capacity.n_blocks());
}
#endif

TEST(KVCacheEstimationTest, SharedKPoolPlanAlignsStateBeforeBudgeting) {
  // Supply the compressed-tail layout directly: backend estimators select
  // different layouts, while the shared planner operates on capacities.
  KVCacheCapacity target_cap;
  target_cap.kpool_layout(KPoolCacheLayout::COMPRESSED_WITH_TAIL)
      .block_size(32)
      .n_layers(2)
      .num_full_attention_layers(2)
      .num_indexer_layers(2)
      .slot_size(64)
      .index_slot_size(16)
      .index_block_bytes(512)
      .num_linear_state_blocks(4)
      .kpool_tail_len(16)
      .kpool_tail_slot_size(2048)
      .linear_cache_size_in_bytes(16384)
      .cache_size_in_bytes(65536);

  for (const int32_t draft_kpool : {4, 32}) {
    for (const int64_t draft_slots : {3, 5}) {
      SCOPED_TRACE(::testing::Message() << "draft_kpool=" << draft_kpool
                                        << ", draft_slots=" << draft_slots);
      ModelArgs draft_args;
      draft_args.index_head_dim(64).index_kpool(draft_kpool);
      KVCacheCapacity draft_cap;
      draft_cap.kpool_layout(KPoolCacheLayout::COMPRESSED_WITH_TAIL)
          .block_size(32)
          .n_layers(1)
          .num_full_attention_layers(1)
          .num_indexer_layers(1)
          .slot_size(128)
          .index_slot_size(draft_kpool == 4 ? 32 : 4)
          .index_block_bytes(draft_kpool == 4 ? 1024 : 128)
          .num_linear_state_blocks(draft_slots)
          .kpool_tail_len(draft_kpool)
          .kpool_tail_slot_size(draft_kpool == 4 ? 1024 : 8192)
          .linear_cache_size_in_bytes(draft_slots *
                                      (draft_kpool == 4 ? 1024 : 8192))
          .cache_size_in_bytes(131072);
      const SharedKPoolPlan plan =
          plan_shared_kpool(target_cap, draft_cap, draft_args);

      EXPECT_EQ(plan.target.num_linear_state_blocks(), 4);
      EXPECT_EQ(plan.draft.num_linear_state_blocks(), 4);
      EXPECT_EQ(plan.draft.kpool_tail_len(), draft_kpool == 4 ? 16 : 32);
      EXPECT_EQ(plan.draft.kpool_tail_slot_size(),
                draft_kpool == 4 ? 4096 : 8192);
      EXPECT_EQ(plan.draft.linear_cache_size_in_bytes(),
                draft_kpool == 4 ? 16384 : 32768);
      EXPECT_EQ(plan.target.n_blocks(), draft_kpool == 4 ? 3 : 1);
      EXPECT_EQ(plan.draft.n_blocks(), plan.target.n_blocks());

      EXPECT_EQ(plan.target.cache_size_in_bytes(), 65536);
      EXPECT_EQ(plan.draft.cache_size_in_bytes(), 65536);
      EXPECT_EQ(plan.target.kpool_tail_len(), target_cap.kpool_tail_len());
    }
  }
}

TEST(KVCacheEstimationTest, SharedKPoolPlanRejectsInsufficientMemory) {
  ModelArgs draft_args;
  draft_args.index_kpool(4).index_head_dim(16);
  KVCacheCapacity capacity;
  capacity.block_size(16)
      .slot_size(64)
      .index_slot_size(8)
      .index_block_bytes(128)
      .num_full_attention_layers(1)
      .num_indexer_layers(1)
      .num_linear_state_blocks(5)
      .kpool_tail_len(16)
      .kpool_tail_slot_size(1024)
      .linear_cache_size_in_bytes(5120)
      .cache_size_in_bytes(10240);
  EXPECT_DEATH(plan_shared_kpool(capacity, capacity, draft_args),
               "target and draft KPool request state");
  capacity.cache_size_in_bytes(10241);
  EXPECT_DEATH(plan_shared_kpool(capacity, capacity, draft_args),
               "no memory for a shared KPool block");
}

TEST(KVCacheEstimationTest, LinearStateCapacityVariants) {
  struct TestCase {
    const char* name;
    int64_t head_dim;
    int64_t cache_size_in_bytes;
    int64_t block_size;
    int64_t n_local_kv_heads;
    int64_t max_seqs_per_batch;
    int64_t max_concurrent_requests;
    InstanceRole instance_role;
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
       /*max_concurrent_requests=*/0,
       /*instance_role=*/InstanceRole::DEFAULT,
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
       /*max_concurrent_requests=*/0,
       /*instance_role=*/InstanceRole::DEFAULT,
       /*enable_prefix_cache=*/true,
       /*max_linear_state_cache_slots=*/0,
       /*expected_num_linear_state_blocks=*/970,
       /*min_num_linear_state_blocks=*/-1},
      {"NoPrefixCacheBoundsLinearStateByConcurrentSequences",
       /*head_dim=*/16,
       /*cache_size_in_bytes=*/1024 * 1024,
       /*block_size=*/16,
       /*n_local_kv_heads=*/2,
       /*max_seqs_per_batch=*/8,
       /*max_concurrent_requests=*/0,
       /*instance_role=*/InstanceRole::DEFAULT,
       /*enable_prefix_cache=*/false,
       /*max_linear_state_cache_slots=*/0,
       /*expected_num_linear_state_blocks=*/10,
       /*min_num_linear_state_blocks=*/-1},
      {"NoPrefixCacheDecodeUsesSingleSlotPerSequence",
       /*head_dim=*/16,
       /*cache_size_in_bytes=*/1024 * 1024,
       /*block_size=*/16,
       /*n_local_kv_heads=*/2,
       /*max_seqs_per_batch=*/8,
       /*max_concurrent_requests=*/0,
       /*instance_role=*/InstanceRole::DECODE,
       /*enable_prefix_cache=*/false,
       /*max_linear_state_cache_slots=*/0,
       /*expected_num_linear_state_blocks=*/10,
       /*min_num_linear_state_blocks=*/-1},
      {"ConcurrencyLimitBoundsLinearStateSeats",
       /*head_dim=*/16,
       /*cache_size_in_bytes=*/1024 * 1024,
       /*block_size=*/16,
       /*n_local_kv_heads=*/2,
       /*max_seqs_per_batch=*/64,
       /*max_concurrent_requests=*/4,
       /*instance_role=*/InstanceRole::DEFAULT,
       /*enable_prefix_cache=*/false,
       /*max_linear_state_cache_slots=*/0,
       /*expected_num_linear_state_blocks=*/6,
       /*min_num_linear_state_blocks=*/-1},
      {"NoPrefixCacheCapsLinearStateBlocksByBudget",
       /*head_dim=*/16,
       /*cache_size_in_bytes=*/1024 * 1024,
       /*block_size=*/16,
       /*n_local_kv_heads=*/2,
       /*max_seqs_per_batch=*/100000,
       /*max_concurrent_requests=*/0,
       /*instance_role=*/InstanceRole::DEFAULT,
       /*enable_prefix_cache=*/false,
       /*max_linear_state_cache_slots=*/0,
       /*expected_num_linear_state_blocks=*/229,
       /*min_num_linear_state_blocks=*/-1},
      {"UnboundedConcurrencyUsesPhysicalLinearStateCap",
       /*head_dim=*/16,
       /*cache_size_in_bytes=*/1024 * 1024,
       /*block_size=*/16,
       /*n_local_kv_heads=*/2,
       /*max_seqs_per_batch=*/0,
       /*max_concurrent_requests=*/0,
       /*instance_role=*/InstanceRole::DEFAULT,
       /*enable_prefix_cache=*/false,
       /*max_linear_state_cache_slots=*/0,
       /*expected_num_linear_state_blocks=*/229,
       /*min_num_linear_state_blocks=*/-1},
      {"ExplicitLinearStateSlotsOverrideAutoSizing",
       /*head_dim=*/16,
       /*cache_size_in_bytes=*/64LL << 30,
       /*block_size=*/16,
       /*n_local_kv_heads=*/2,
       /*max_seqs_per_batch=*/200,
       /*max_concurrent_requests=*/0,
       /*instance_role=*/InstanceRole::DEFAULT,
       /*enable_prefix_cache=*/true,
       /*max_linear_state_cache_slots=*/32,
       /*expected_num_linear_state_blocks=*/34,
       /*min_num_linear_state_blocks=*/-1},
      {"ExplicitLinearStateSlotsOverrideBoundedSizing",
       /*head_dim=*/16,
       /*cache_size_in_bytes=*/64LL << 30,
       /*block_size=*/16,
       /*n_local_kv_heads=*/2,
       /*max_seqs_per_batch=*/200,
       /*max_concurrent_requests=*/0,
       /*instance_role=*/InstanceRole::DEFAULT,
       /*enable_prefix_cache=*/false,
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
    options.max_concurrent_requests = test_case.max_concurrent_requests;
    options.instance_role = test_case.instance_role;
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
    EXPECT_GT(capacity.n_blocks(), 0);
    EXPECT_LE(capacity.linear_cache_size_in_bytes(),
              test_case.cache_size_in_bytes);
  }
}

TEST(KVCacheEstimationTest, CapsPhysicalLinearStateSlotsForSelectedBackend) {
  ModelArgs model_args = make_linear_attention_args(/*head_dim=*/1);
  model_args.model_type("qwen3_5");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.cache_size_in_bytes = 64LL << 30;
  options.enable_prefix_cache = true;

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_GT(capacity.num_linear_state_blocks(), 1024);
  options.linear_state_cache_block_limit = 1024;
  capacity = estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(capacity.num_linear_state_blocks(), 1024);
}

TEST(KVCacheEstimationTest, RejectsExplicitSlotsBeyondSelectedBackendLimit) {
  ModelArgs model_args = make_linear_attention_args(/*head_dim=*/1);
  model_args.model_type("qwen3_5");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.cache_size_in_bytes = 64LL << 30;
  options.enable_prefix_cache = true;
  options.max_linear_state_cache_slots = 1023;
  options.linear_state_cache_block_limit = 1024;

  EXPECT_DEATH(
      (void)estimate_kv_cache_capacity(model_args, options),
      "Selected backend supports at most 1024 physical linear-state slots");
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

TEST(KVCacheEstimationTest, MultimodalMtpReservesPerTokenRecurrentCheckpoints) {
  ModelArgs model_args = make_standard_args();
  model_args.model_type("qwen3_5")
      .full_attention_interval(2)
      .linear_num_key_heads(2)
      .linear_num_value_heads(2)
      .linear_key_head_dim(4)
      .linear_value_head_dim(8)
      .linear_conv_kernel_dim(3);
  runtime::Options options;
  options.max_memory_utilization(1.0)
      .block_size(16)
      .max_seqs_per_batch(4)
      .max_tokens_per_batch(64)
      .max_linear_state_cache_slots(16)
      .enable_prefix_cache(false)
      .num_speculative_tokens(0);
  KVCacheEstimateContext context;
  context.dtype = torch::kFloat16;
  context.world_size = 1;
  context.worker_memory = {{256LL << 20, 256LL << 20}};
  context.is_multimodal = true;
  const KVCacheCapacity ordinary =
      KVCacheEstimator(model_args).estimate(options, context);
  options.num_speculative_tokens(4);
  const KVCacheCapacity speculative =
      KVCacheEstimator(model_args).estimate(options, context);

  EXPECT_EQ(ordinary.linear_ssm_checkpoint_stride(), 1);
  EXPECT_EQ(speculative.linear_conv_state_len(), 6);
  EXPECT_EQ(speculative.linear_ssm_checkpoint_stride(), 5);
  EXPECT_GT(speculative.linear_cache_size_in_bytes(),
            ordinary.linear_cache_size_in_bytes());
  EXPECT_LT(speculative.n_blocks(), ordinary.n_blocks());
}

TEST_F(VirtualMemoryPrefixEstimationTest, ReservesReusableLinearStateSlots) {
  ModelArgs model_args = make_standard_args();
  model_args.model_type("qwen3_5")
      .full_attention_interval(2)
      .linear_num_key_heads(2)
      .linear_num_value_heads(2)
      .linear_key_head_dim(4)
      .linear_value_head_dim(8)
      .linear_conv_kernel_dim(3);
  runtime::Options options;
  options.block_size(16)
      .max_seqs_per_batch(4)
      .max_tokens_per_batch(64)
      .max_linear_state_cache_slots(0)
      .enable_prefix_cache(false)
      .num_speculative_tokens(0);
  KVCacheEstimateContext context;
  context.dtype = torch::kFloat16;
  context.world_size = 1;
  context.virtual_memory_cache_size = 1LL << 20;
  const KVCacheCapacity without_prefix =
      KVCacheEstimator(model_args).estimate(options, context);
  options.enable_prefix_cache(true);
  const KVCacheCapacity with_prefix =
      KVCacheEstimator(model_args).estimate(options, context);

  EXPECT_GT(with_prefix.num_linear_state_blocks(),
            without_prefix.num_linear_state_blocks());
  EXPECT_GT(with_prefix.linear_cache_size_in_bytes(),
            without_prefix.linear_cache_size_in_bytes());
  EXPECT_LT(with_prefix.n_blocks(), without_prefix.n_blocks());
}

#if defined(USE_NPU)
TEST(KVCacheEstimationTest, Qwen35NativeMtpDraftUsesFullAttentionCacheBudget) {
  for (const std::string& model_type :
       {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}) {
    for (const int64_t draft_layers : {1, 2}) {
      SCOPED_TRACE(model_type);
      SCOPED_TRACE(draft_layers);
      ModelArgs target_args = make_linear_attention_args();
      target_args.model_type(model_type)
          .num_nextn_predict_layers(draft_layers)
          .layer_types({"linear_attention",
                        "full_attention",
                        "linear_attention",
                        "full_attention"});
      ModelRegistry::configure_mtp_args(target_args,
                                        /*speculative_algorithm=*/"MTP",
                                        /*is_draft_model=*/false,
                                        /*is_python_model=*/false);
      ModelArgs draft_args = target_args;
      ModelRegistry::configure_mtp_args(draft_args,
                                        /*speculative_algorithm=*/"MTP",
                                        /*is_draft_model=*/true,
                                        /*is_python_model=*/false);

      KVCacheEstimateOptions target_options = make_linear_attention_options();
      target_options.num_speculative_tokens = 3;
      const KVCacheCapacity target_capacity =
          estimate_kv_cache_capacity(target_args, target_options);
      EXPECT_EQ(target_capacity.n_layers(), 4);
      EXPECT_EQ(target_capacity.num_full_attention_layers(), 2);
      EXPECT_EQ(target_capacity.num_linear_attention_layers(), 2);
      EXPECT_GT(target_capacity.linear_cache_size_in_bytes(), 0);

      KVCacheEstimateOptions draft_options = target_options;
      draft_options.is_draft_model = true;
      const KVCacheCapacity draft_capacity =
          estimate_kv_cache_capacity(draft_args, draft_options);
      EXPECT_EQ(draft_capacity.n_layers(), draft_layers);
      EXPECT_EQ(draft_capacity.num_full_attention_layers(), draft_layers);
      EXPECT_EQ(draft_capacity.num_linear_attention_layers(), 0);
      EXPECT_EQ(draft_capacity.linear_cache_size_in_bytes(), 0);
      EXPECT_EQ(draft_capacity.slot_size(), 128);
      EXPECT_EQ(draft_capacity.block_size(), 16);
      EXPECT_EQ(draft_capacity.cache_size_in_bytes(), 1024 * 1024);
      EXPECT_EQ(draft_capacity.n_blocks(), 512 / draft_layers);
    }
  }
}
#endif

TEST(KVCacheEstimationTest, GlmMtpAllocatesFrameworkCheckpointsOnlyForDecode) {
  for (const std::string& model_type : {"glm5_3_flash", "glm5_next"}) {
    ModelArgs model_args = make_standard_args();
    model_args.model_type(model_type)
        .full_attention_interval(2)
        .linear_num_key_heads(2)
        .linear_num_value_heads(2)
        .linear_key_head_dim(4)
        .linear_value_head_dim(8)
        .linear_conv_kernel_dim(3);
    KVCacheEstimateOptions options = make_estimate_options();
    options.n_local_linear_k_heads = 2;
    options.n_local_linear_v_heads = 2;
    options.num_speculative_tokens = 3;
    const KVCacheCapacity decode =
        estimate_kv_cache_capacity(model_args, options);
    EXPECT_EQ(decode.linear_conv_state_len(), 5);
    EXPECT_EQ(decode.linear_ssm_checkpoint_stride(), 4);
    options.instance_role = InstanceRole::PREFILL;
    const KVCacheCapacity prefill =
        estimate_kv_cache_capacity(model_args, options);
    EXPECT_EQ(prefill.linear_conv_state_len(), 2);
    EXPECT_EQ(prefill.linear_ssm_checkpoint_stride(), 1);
    EXPECT_GT(decode.linear_slot_size(), prefill.linear_slot_size());
  }
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
  options.cache_size_in_bytes =
      2818048 + /*sixteen_additional_swa_blocks=*/16 * 90112;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 2176;

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 35);
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

TEST(KVCacheEstimationTest, DeepSeekV4RejectsBudgetWithoutCompressedCacheUnit) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(128)
      .compress_ratios({1, 4, 128});

  constexpr int64_t kSwaCount = 7;
  constexpr int64_t kSwaBytesPerBlock =
      /*c1=*/128 * 16 * 4 +
      /*c4=*/128 * (16 * 4 + 2 * 16 * 4 * 2 + 2 * 8 * 4 * 2) +
      /*c128=*/128 * (16 * 4 + 16 * 4 * 2);
  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes =
      kSwaCount * kSwaBytesPerBlock + /*remaining_bytes=*/1;
  options.block_size = 128;
  options.max_seqs_per_batch = 1;
  options.max_tokens_per_batch = 384;
  options.max_tokens_per_chunk_for_prefill = 128;

  EXPECT_DEATH(
      estimate_kv_cache_capacity(model_args, options),
      "minimum DSV4 SWA cache leaves insufficient memory for one compressed "
      "cache unit");
}

TEST(KVCacheEstimationTest, DeepSeekV4PdPrefillUsesBatchTokenCapacity) {
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
  options.cache_size_in_bytes = 16 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 385;
  options.max_tokens_per_chunk_for_prefill = 385;
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::PREFILL;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  // Four sequences retain three window blocks each, plus four burst blocks,
  // four per-sequence tail blocks, and two guard blocks.
  EXPECT_EQ(capacity.swa_count(), 22);
}

TEST(KVCacheEstimationTest, DeepSeekV4MixUsesBatchTokenCapacity) {
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
  options.cache_size_in_bytes = 16 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 16384;
  options.max_tokens_per_chunk_for_prefill = 385;
  options.instance_role = InstanceRole::MIX;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 146);
}

TEST(KVCacheEstimationTest,
     DeepSeekV4PdPrefillWithoutChunkingUsesBatchCapacity) {
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
  options.cache_size_in_bytes = 16 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 385;
  options.max_tokens_per_chunk_for_prefill = 129;
  options.enable_chunked_prefill = false;
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::PREFILL;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  // Chunking does not affect SWA sizing.
  EXPECT_EQ(capacity.swa_count(), 22);
}

TEST(KVCacheEstimationTest, DeepSeekV4PdDecodeUsesBatchTokenCapacity) {
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
  options.cache_size_in_bytes = 16 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 385;
  options.instance_role = InstanceRole::DECODE;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 22);
}

TEST(KVCacheEstimationTest,
     DeepSeekV4PdDecodeSwaCapacityIgnoresScheduleOverlap) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(128)
      .compress_ratios({1, 4, 128});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 16 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 1;
  options.max_tokens_per_batch = 385;
  options.num_speculative_tokens = 65;
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::DECODE;

  options.enable_schedule_overlap = false;
  const KVCacheCapacity non_overlap_capacity =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(non_overlap_capacity.swa_count(), 8);

  options.enable_schedule_overlap = true;
  const KVCacheCapacity overlap_capacity =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(overlap_capacity.swa_count(), 8);
}

TEST(KVCacheEstimationTest, DeepSeekV4PrefixCacheKeepsOperationalSwaPool) {
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
  options.cache_size_in_bytes = 128 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 385;
  options.max_tokens_per_chunk_for_prefill = 385;
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::PREFILL;
  options.enable_prefix_cache = true;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  ASSERT_GT(capacity.c128_count(), 0);
  EXPECT_EQ(capacity.c4_count(), 32 * capacity.c128_count());
  EXPECT_EQ(capacity.swa_count(), 22);
}

TEST(KVCacheEstimationTest,
     DeepSeekV4RealisticMixBudgetRetainsCompressedPools) {
  std::vector<int32_t> compress_ratios{0, 0};
  compress_ratios.reserve(43);
  for (int32_t layer_id = 2; layer_id < 43; ++layer_id) {
    compress_ratios.emplace_back(layer_id % 2 == 0 ? 4 : 128);
  }

  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(43)
      .head_dim(512)
      .index_head_dim(128)
      .window_size(128)
      .max_seq_len(1048576)
      .compress_ratios(std::move(compress_ratios));

  KVCacheEstimateOptions options;
  options.dtype = torch::kBFloat16;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = int64_t{16} * 1024 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 10;
  options.max_tokens_per_batch = 10240;
  options.max_tokens_per_chunk_for_prefill = 2048;
  options.enable_prefix_cache = true;
  options.instance_role = InstanceRole::MIX;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 102);
  EXPECT_GT(capacity.c4_count(), 0);
  EXPECT_GT(capacity.c128_count(), 0);
  EXPECT_EQ(capacity.c4_count(), 32 * capacity.c128_count());
}

TEST(KVCacheEstimationTest,
     DeepSeekV4PdPrefillFairBudgetKeepsPrefixCachePublishUnit) {
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
  options.cache_size_in_bytes = 64 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 16384;
  options.max_tokens_per_chunk_for_prefill = 4096;
  options.enable_chunked_prefill = true;
  options.enable_prefix_cache = true;
  options.enable_dp_fair_token_budget = true;
  options.dp_size = 4;
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::PREFILL;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 146);

  KVCacheEstimateOptions no_prefix_options = options;
  no_prefix_options.enable_prefix_cache = false;
  const KVCacheCapacity no_prefix_capacity =
      estimate_kv_cache_capacity(model_args, no_prefix_options);
  EXPECT_EQ(no_prefix_capacity.swa_count(), 50);
}

TEST(KVCacheEstimationTest, DeepSeekV4DecodeKeepsOperationalSwaPool) {
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
  options.cache_size_in_bytes = 128 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 385;
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::DECODE;
  options.enable_prefix_cache = true;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 22);
  EXPECT_GT(capacity.c128_count(), 0);
}

TEST(KVCacheEstimationTest, DeepSeekV4SwaOnlyPrefixKeepsOperationalSwaPool) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4_dspark")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 1, 1});

  constexpr int64_t kSwaBytesPerBlock = 3 * 128 * 16 * 4;
  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 43 * kSwaBytesPerBlock;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 385;
  options.max_tokens_per_chunk_for_prefill = 385;
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::PREFILL;
  options.enable_prefix_cache = true;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 22);
  EXPECT_EQ(capacity.c4_count(), 0);
  EXPECT_EQ(capacity.c128_count(), 0);
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
  options.max_tokens_per_batch = 2176;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 35);
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
  const DeepSeekV4CachePolicy cache_policy =
      get_dsv4_cache_policy(target_options.dtype);
  const int64_t scale_bytes =
      cache_policy.has_indexer_cache_scale ? cache_policy.scale_dtype_size : 0;
  const int64_t c4_block_bytes =
      128 * (16 * 4 + 8 * cache_policy.index_dtype_size + scale_bytes);
  const int64_t c128_block_bytes = 128 * 16 * 4;
  const int64_t compressed_unit_bytes = 32 * c4_block_bytes + c128_block_bytes;
  constexpr int64_t kTargetSwaBytes = 35 * 90112;
  constexpr int64_t kDraftSwaBytes =
      /*layers=*/3 * /*swa_count=*/35 * /*block_size=*/128 *
      /*head_dim=*/16 * /*float32_bytes=*/4;
  // Reserve both SWA pools and exactly two compressed units for either
  // the model-dtype indexer cache or the quantized indexer cache with scales.
  target_options.cache_size_in_bytes =
      kTargetSwaBytes + kDraftSwaBytes + 2 * compressed_unit_bytes;
  target_options.block_size = 128;
  target_options.max_seqs_per_batch = 4;
  target_options.max_tokens_per_batch = 2176;
  KVCacheEstimateOptions draft_options = target_options;
  draft_options.is_draft_model = true;

  const KVCacheCapacity target_only_capacity =
      estimate_kv_cache_capacity(target_args, target_options);

  target_options.draft_model_args = &draft_args;
  target_options.draft_options = &draft_options;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(target_args, target_options);

  EXPECT_LE(capacity.cache_size_in_bytes() + kDraftSwaBytes,
            target_options.cache_size_in_bytes);
  EXPECT_EQ(capacity.swa_count(), 35);
  EXPECT_EQ(capacity.c4_count(), 64);
  EXPECT_EQ(capacity.c128_count(), 2);
  EXPECT_EQ(capacity.swa_count(), target_only_capacity.swa_count());
  EXPECT_EQ(capacity.c4_count() + 64, target_only_capacity.c4_count());
  EXPECT_EQ(capacity.c128_count() + 2, target_only_capacity.c128_count());
  EXPECT_LT(capacity.n_blocks(), target_only_capacity.n_blocks());
}

TEST(KVCacheEstimationTest, PhysicalHeadWidthControlsCacheBudget) {
  ModelArgs model_args = make_standard_args();
  model_args.head_dim(96).kv_cache_storage_head_dim(128).n_heads(8).n_kv_heads(
      4);
  runtime::Options runtime_options;
  runtime_options.block_size(16);
  KVCacheEstimateContext context;
  context.dtype = torch::kBFloat16;
  context.world_size = 2;
  context.virtual_memory_cache_size = 1024 * 1024;

  const KVCacheCapacity capacity =
      KVCacheEstimator(model_args).estimate(runtime_options, context);

  // Two local KV heads, with physical padding in both K and V.
  EXPECT_EQ(capacity.slot_size(), 2 * 2 * 128 * 2);
  EXPECT_EQ(model_args.head_dim(), 96);
}

TEST_P(DcpIndexerEstimationTest, IncludesAllIndexerBytes) {
  ModelArgs model_args = make_standard_args();
  model_args.index_n_heads(1).index_head_dim(16);
  KVCacheEstimateOptions options = make_estimate_options();
  options.indexer_cache_dtype = "auto";
  const KVCacheCapacity ordinary =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(ordinary.index_slot_size(), 32 * expected_factor());

  options.indexer_cache_dtype = "int8";
  const KVCacheCapacity quantized =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(quantized.index_slot_size(), 20 * expected_factor());
  EXPECT_EQ(quantized.slot_size(), ordinary.slot_size());

  model_args.index_n_heads(0);
  const KVCacheCapacity absent =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(absent.index_slot_size(), 0);
}

INSTANTIATE_TEST_SUITE_P(KvSplits,
                         DcpIndexerEstimationTest,
                         ::testing::Values(1, 2, 4, 0));

#if defined(USE_NPU)
TEST(KVCacheIndexCacheBytesTest, CompressedTailChargesWholePools) {
  ModelArgs args;
  args.n_layers(4).head_dim(16).n_heads(4).n_kv_heads(4);
  args.model_type("glm5_next")
      .index_n_heads(1)
      .index_head_dim(128)
      .index_kpool(3)
      .index_kpool_compress(true)
      .index_kpool_always_select_tail(true);
  KVCacheEstimateOptions options;
  options.dtype = torch::kBFloat16;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 1024 * 1024;
  options.block_size = 96;
  options.world_size = 1;
  options.n_local_kv_heads = 2;
  options.max_seqs_per_batch = 8;
  const KVCacheCapacity capacity = estimate_kv_cache_capacity(args, options);
  const int64_t pool_bytes = 2 * 128;
  EXPECT_EQ(capacity.index_slot_size(), (pool_bytes + 3 - 1) / 3);
  EXPECT_EQ(capacity.index_block_bytes(), 96 / 3 * pool_bytes);
  EXPECT_NE(capacity.index_block_bytes(),
            capacity.index_slot_size() * options.block_size);
}
#endif

}  // namespace xllm
