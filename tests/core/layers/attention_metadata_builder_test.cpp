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

#include "core/layers/common/attention_metadata_builder.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>

#include "core/framework/batch/batch_forward_type.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/model/model_params_view.h"
#include "core/framework/model/rec_model_params.h"
#include "core/layers/common/attention_metadata.h"

namespace xllm::layer {
namespace {

ModelParamsView make_params() {
  ModelParamsView params = ModelInputSnapshot(LlmModelParams()).view();
  params.meta.batch_forward_type = BatchForwardType::PREFILL;
  params.meta.num_sequences = 3;
  params.meta.q_max_seq_len = 1;
  params.meta.kv_max_seq_len = 8;
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  params.attention.device.q_seq_lens = torch::tensor({1, 1, 1}, options);
  params.attention.device.kv_seq_lens = torch::tensor({1, 4, 8}, options);
  params.attention.device.q_cu_seq_lens = torch::tensor({1, 2, 3}, options);
  params.attention.device.kv_cache_tokens_nums =
      torch::tensor({7, 0, 5}, options);
  params.embedding.linear_state_ids = {4, 2, 9};
  params.embedding.linear_state_indices = torch::tensor({4, 2, 9}, options);
  params.linear_state_validity_mask = {0, 1, 0};
  return params;
}

RecModelParams make_rec_params() {
  RecModelParams params;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.meta.num_sequences = 2;
  params.meta.q_max_seq_len = 1;
  params.meta.kv_max_seq_len = 5;
  params.attention.host.q_seq_lens = {1, 1};
  params.attention.host.kv_seq_lens = {3, 5};
  params.attention.host.q_cu_seq_lens = {1, 2};
  params.attention.device.q_seq_lens =
      torch::tensor(params.attention.host.q_seq_lens, torch::kInt32);
  params.attention.device.kv_seq_lens =
      torch::tensor(params.attention.host.kv_seq_lens, torch::kInt32);
  params.attention.device.q_cu_seq_lens =
      torch::tensor(params.attention.host.q_cu_seq_lens, torch::kInt32);
  params.attention.device.new_cache_slots =
      torch::tensor({1, 2}, torch::kInt32);
  params.attention.device.block_tables =
      torch::tensor({{0}, {1}}, torch::kInt32);
  return params;
}

TEST(AttentionMetadataBuilderTest, RecRoundMetadataDoesNotEnableSpecVerify) {
  auto params = make_rec_params();
  auto& rec = params.mutable_llmrec_params();
  rec.beam_width = 2;
  rec.current_round_tensor = torch::tensor({3}, torch::kInt32);

  const auto metadata =
      AttentionMetadataBuilder::build(params, /*enable_mla=*/false);

  EXPECT_TRUE(metadata.step_tensor.is_same(rec.current_round_tensor));
  EXPECT_FALSE(metadata.is_spec_verify);
  EXPECT_FALSE(metadata.expanded_decode.enabled);
  EXPECT_EQ(metadata.total_kv_len, 8);
  EXPECT_TRUE(
      metadata.block_table.is_same(params.attention.device.block_tables));
}

TEST(AttentionMetadataBuilderTest, LlmSpecVerifyDoesNotAcquireRecRoundState) {
  auto params = make_params();
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.is_spec_verify = true;
  params.graph.use_expanded_decode_for_spec_verify_attention = true;
  params.graph.expanded_kv_seq_lens = torch::tensor({2, 3}, torch::kInt32);
  params.graph.expanded_block_tables = torch::tensor({{0}, {1}}, torch::kInt32);
  params.graph.expanded_paged_kv_indptr =
      torch::tensor({0, 1, 2}, torch::kInt32);
  params.graph.expanded_paged_kv_indices = torch::tensor({0, 1}, torch::kInt32);
  params.graph.expanded_paged_kv_last_page_len =
      torch::tensor({2, 3}, torch::kInt32);

  const auto metadata =
      AttentionMetadataBuilder::build(params, /*enable_mla=*/false);

  EXPECT_TRUE(metadata.is_spec_verify);
  EXPECT_TRUE(metadata.expanded_decode.enabled);
  EXPECT_TRUE(metadata.expanded_decode.kv_seq_lens.is_same(
      params.graph.expanded_kv_seq_lens));
  EXPECT_FALSE(metadata.step_tensor.defined());
#if defined(USE_CUDA) || defined(USE_MUSA)
  EXPECT_FALSE(metadata.xattention_two_stage_decode_cache.has_value());
#endif
}

#if defined(USE_CUDA) || defined(USE_MUSA)
class ScopedXAttentionMode final {
 public:
  explicit ScopedXAttentionMode(bool one_stage)
      : previous_(RecConfig::get_instance().enable_xattention_one_stage()) {
    RecConfig::get_instance().enable_xattention_one_stage(one_stage);
  }
  ~ScopedXAttentionMode() {
    RecConfig::get_instance().enable_xattention_one_stage(previous_);
  }

 private:
  bool previous_;
};

TEST(AttentionMetadataBuilderTest,
     RecTwoStageCacheKeepsBeamAndRoundDimensions) {
  const ScopedXAttentionMode mode(/*one_stage=*/false);
  auto params = make_rec_params();
  auto& rec = params.mutable_llmrec_params();
  rec.beam_width = 2;
  rec.current_round_tensor = torch::tensor({3}, torch::kInt32);
  rec.unshared_k_caches = {torch::zeros({2, 2, 7, 1, 4})};
  rec.two_stage_shared_o = torch::zeros({4, 2, 4});
  rec.two_stage_shared_lse = torch::zeros({4, 2, 1});
  rec.two_stage_unshared_o = torch::zeros({4, 2, 4});
  rec.two_stage_unshared_lse = torch::zeros({4, 2, 1});
  rec.two_stage_q_cu_seq_lens_shared = torch::tensor({0, 2, 4}, torch::kInt32);
  rec.two_stage_qo_indptr_expanded =
      torch::tensor({0, 1, 2, 3, 4}, torch::kInt32);

  const auto metadata =
      AttentionMetadataBuilder::build(params, /*enable_mla=*/false);

  ASSERT_TRUE(metadata.xattention_two_stage_decode_cache.has_value());
  const auto& cache = metadata.xattention_two_stage_decode_cache.value();
  EXPECT_EQ(cache.cached_batch_size, 2);
  EXPECT_EQ(cache.cached_beam_size, 2);
  EXPECT_EQ(cache.cached_max_decode_step, 7);
  EXPECT_EQ(cache.cached_step, 3);
  EXPECT_EQ(cache.cached_num_heads, 2);
  EXPECT_EQ(cache.cached_head_size, 4);
  EXPECT_TRUE(cache.shared_o.is_same(rec.two_stage_shared_o));
  EXPECT_TRUE(cache.unshared_o.is_same(rec.two_stage_unshared_o));
  EXPECT_TRUE(
      cache.qo_indptr_expanded.is_same(rec.two_stage_qo_indptr_expanded));
}

TEST(AttentionMetadataBuilderTest, RecOneStageDoesNotBuildTwoStageCache) {
  const ScopedXAttentionMode mode(/*one_stage=*/true);
  auto params = make_rec_params();
  auto& rec = params.mutable_llmrec_params();
  rec.current_round_tensor = torch::tensor({1}, torch::kInt32);

  const auto metadata =
      AttentionMetadataBuilder::build(params, /*enable_mla=*/false);

  EXPECT_TRUE(metadata.step_tensor.is_same(rec.current_round_tensor));
  EXPECT_FALSE(metadata.xattention_two_stage_decode_cache.has_value());
}
#endif

TEST(AttentionMetadataBuilderTest, LinearPrefillSchedulesRespectChunkSize) {
  for (const int32_t block_size : {16, 64}) {
    for (const bool chunked : {false, true}) {
      AttentionMetadata metadata;
      metadata.is_prefill = !chunked;
      metadata.is_chunked_prefill = chunked;
      metadata.q_cu_seq_lens =
          torch::tensor({0, block_size, 2 * block_size + 1}, torch::kInt32);
      AttentionMetadataBuilder::build_linear_prefill(metadata, block_size);
      EXPECT_EQ(metadata.tot, 3);
      EXPECT_TRUE(torch::equal(metadata.batch.narrow(0, 0, 3),
                               torch::tensor({0, 1, 1}, torch::kInt32)));
      EXPECT_TRUE(torch::equal(metadata.token_block_offset.narrow(0, 0, 3),
                               torch::tensor({0, 0, 1}, torch::kInt32)));
      EXPECT_TRUE(
          torch::equal(metadata.chunk_indices,
                       torch::tensor({{0, 0}, {1, 0}, {1, 1}}, torch::kInt32)));
      EXPECT_TRUE(metadata.batch.slice(0, 3).eq(-1).all().item<bool>());
      EXPECT_TRUE(
          metadata.token_block_offset.slice(0, 3).eq(-1).all().item<bool>());
    }
  }
}

TEST(AttentionMetadataBuilderTest, LinearDecodeDoesNotRequirePrefillLengths) {
  AttentionMetadata metadata;
  metadata.is_prefill = false;
  metadata.is_chunked_prefill = false;
  AttentionMetadataBuilder::build_linear_prefill(metadata, /*block_size=*/64);
  EXPECT_FALSE(metadata.batch.defined());
  EXPECT_FALSE(metadata.chunk_indices.defined());
}

TEST(AttentionMetadataBuilderTest, MaterializesCanonicalInitialStateMask) {
  ModelParamsView params = make_params();

  AttentionMetadata metadata =
      AttentionMetadataBuilder::build(params, /*enable_mla=*/false);

  ASSERT_TRUE(metadata.has_initial_states.defined());
  EXPECT_EQ(metadata.has_initial_states.scalar_type(), torch::kBool);
  EXPECT_EQ(metadata.has_initial_states.device(), torch::Device(torch::kCPU));
  EXPECT_TRUE(torch::equal(metadata.has_initial_states,
                           torch::tensor({false, true, false}, torch::kBool)));
}

TEST(AttentionMetadataBuilderTest, DoesNotDeriveValidityFromContextOrSlot) {
  ModelParamsView params = make_params();
  params.embedding.linear_state_indices = torch::tensor({0, 0, 0}, torch::kInt);
  params.attention.device.kv_cache_tokens_nums =
      torch::tensor({9, 0, 11}, torch::kInt);
  params.linear_state_validity_mask = {0, 1, 0};

  AttentionMetadata metadata =
      AttentionMetadataBuilder::build(params, /*enable_mla=*/false);

  EXPECT_TRUE(torch::equal(metadata.has_initial_states,
                           torch::tensor({false, true, false}, torch::kBool)));
}

TEST(AttentionMetadataBuilderTest, MaterializesChunkedPrefillMask) {
  ModelParamsView params = make_params();
  params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;

  AttentionMetadata metadata =
      AttentionMetadataBuilder::build(params, /*enable_mla=*/false);

  ASSERT_TRUE(metadata.has_initial_states.defined());
  EXPECT_TRUE(torch::equal(metadata.has_initial_states,
                           torch::tensor({false, true, false}, torch::kBool)));
}

TEST(AttentionMetadataBuilderTest, SkipsInitialStateMaterializationOnRequest) {
  ModelParamsView params = make_params();
  AttentionMetadataBuildOptions build_options;
  build_options.materialize_linear_state_validity = false;

  AttentionMetadata metadata = AttentionMetadataBuilder::build(
      params, /*enable_mla=*/false, {}, std::nullopt, build_options);

  EXPECT_FALSE(metadata.has_initial_states.defined());
  EXPECT_EQ(params.linear_state_validity_mask, std::vector<int64_t>({0, 1, 0}));
  EXPECT_EQ(params.embedding.linear_state_ids, std::vector<int32_t>({4, 2, 9}));
}

TEST(AttentionMetadataBuilderTest,
     SkippingInitialStateMaterializationKeepsValidation) {
  ModelParamsView params = make_params();
  params.linear_state_validity_mask.pop_back();
  AttentionMetadataBuildOptions build_options;
  build_options.materialize_linear_state_validity = false;

  EXPECT_DEATH(
      AttentionMetadataBuilder::build(
          params, /*enable_mla=*/false, {}, std::nullopt, build_options),
      "linear state mask row count mismatch");
}

TEST(AttentionMetadataBuilderTest, DecodeDoesNotMaterializeInitialStateMask) {
  ModelParamsView params = make_params();
  params.meta.batch_forward_type = BatchForwardType::DECODE;

  AttentionMetadata metadata =
      AttentionMetadataBuilder::build(params, /*enable_mla=*/false);

  EXPECT_FALSE(metadata.has_initial_states.defined());
}

TEST(AttentionMetadataBuilderTest, IgnoresTransportIdsWithoutLinearStateOps) {
  ModelParamsView params = make_params();
  params.embedding.linear_state_ids = {-1, -1, -1};
  params.embedding.linear_state_indices =
      torch::tensor({-1, -1, -1}, torch::kInt);
  params.linear_state_validity_mask.clear();

  AttentionMetadata metadata =
      AttentionMetadataBuilder::build(params, /*enable_mla=*/false);

  EXPECT_FALSE(metadata.has_initial_states.defined());
}

TEST(AttentionMetadataBuilderTest, MaterializesColdMaskForDummyShard) {
  ModelParamsView params = ModelInputSnapshot(LlmModelParams()).view();
  params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  params.meta.num_sequences = 0;
  params.meta.q_max_seq_len = 0;
  params.meta.kv_max_seq_len = 0;

  AttentionMetadata metadata =
      AttentionMetadataBuilder::build(params,
                                      /*enable_mla=*/false,
                                      /*attn_mask=*/{},
                                      torch::Device(torch::kCPU));

  ASSERT_TRUE(metadata.is_dummy);
#if defined(USE_NPU)
  EXPECT_EQ(metadata.q_seq_lens_vec, std::vector<int32_t>({1}));
  EXPECT_EQ(metadata.kv_seq_lens_vec, std::vector<int32_t>({1}));
#else
  EXPECT_EQ(metadata.q_seq_lens_vec, std::vector<int32_t>({0, 1}));
  EXPECT_EQ(metadata.kv_seq_lens_vec, std::vector<int32_t>({0, 1}));
#endif
  EXPECT_TRUE(
      torch::equal(metadata.q_seq_lens, torch::tensor({1}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(metadata.kv_seq_lens, torch::tensor({1}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(metadata.q_cu_seq_lens,
                           torch::tensor({0, 1}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(metadata.kv_cu_seq_lens,
                           torch::tensor({0, 1}, torch::kInt32)));
  ASSERT_TRUE(metadata.has_initial_states.defined());
  EXPECT_TRUE(
      torch::equal(metadata.has_initial_states, torch::tensor({false})));
}

#if defined(USE_MUSA)
TEST(AttentionMetadataBuilderTest, BuildsMusaMetadataWithCommonBuilder) {
  ModelParamsView params = ModelInputSnapshot(LlmModelParams()).view();
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.meta.num_sequences = 2;
  params.meta.q_max_seq_len = 1;
  params.meta.kv_max_seq_len = 7;
  params.attention.host.q_seq_lens = {1, 1};
  params.attention.host.kv_seq_lens = {3, 7};

  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  params.attention.device.q_seq_lens = torch::tensor({1, 1}, options);
  params.attention.device.kv_seq_lens = torch::tensor({3, 7}, options);
  params.attention.device.q_cu_seq_lens = torch::tensor({1, 2}, options);
  params.attention.device.block_tables =
      torch::tensor({{0, -1}, {1, 2}}, options);
  params.attention.device.paged_kv_indptr = torch::tensor({0, 1, 3}, options);
  params.attention.device.paged_kv_indices = torch::tensor({0, 1, 2}, options);
  params.attention.device.paged_kv_last_page_len =
      torch::tensor({3, 7}, options);

  params.attn_metadata = std::make_shared<AttentionMetadata>();
  params.attn_metadata->fa3_metadata.share_fa3_scheduler_metadata = true;
  params.attn_metadata->fa3_metadata.fa3_scheduler_metadata =
      torch::tensor({4, 3, 2, 1}, options);

  AttentionMetadata metadata =
      AttentionMetadataBuilder::build(params, /*enable_mla=*/false);

  EXPECT_TRUE(torch::equal(metadata.q_cu_seq_lens,
                           torch::tensor({0, 1, 2}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(metadata.kv_cu_seq_lens,
                           torch::tensor({0, 3, 10}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(metadata.q_seq_lens, torch::tensor({1, 1}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(metadata.kv_seq_lens, torch::tensor({3, 7}, torch::kInt32)));
  EXPECT_EQ(metadata.block_table.scalar_type(), torch::kInt32);
  EXPECT_EQ(metadata.fa3_metadata.paged_kv_indptr_host.device(),
            torch::Device(torch::kCPU));
  EXPECT_EQ(metadata.fa3_metadata.paged_kv_indptr_host.scalar_type(),
            torch::kInt32);
  EXPECT_EQ(metadata.fa3_metadata.paged_kv_indices_host.scalar_type(),
            torch::kInt32);
  EXPECT_EQ(metadata.fa3_metadata.paged_kv_last_page_len_host.scalar_type(),
            torch::kInt32);
  EXPECT_TRUE(metadata.fa3_metadata.share_fa3_scheduler_metadata);
  EXPECT_TRUE(
      torch::equal(metadata.fa3_metadata.fa3_scheduler_metadata,
                   params.attn_metadata->fa3_metadata.fa3_scheduler_metadata));
}
#endif

}  // namespace
}  // namespace xllm::layer
