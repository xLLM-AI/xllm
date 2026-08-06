/* Copyright 2026 The xLLM Authors.

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

#include "core/layers/common/attention_metadata_builder.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>

#include "core/framework/batch/batch_forward_type.h"
#include "core/framework/model/model_input_params.h"
#include "core/layers/common/attention_metadata.h"

namespace xllm::layer {
namespace {

ModelInputParams make_params() {
  ModelInputParams params;
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

TEST(AttentionMetadataBuilderTest, MaterializesCanonicalInitialStateMask) {
  ModelInputParams params = make_params();

  AttentionMetadata metadata =
      AttentionMetadataBuilder::build(params, /*enable_mla=*/false);

  ASSERT_TRUE(metadata.has_initial_states.defined());
  EXPECT_EQ(metadata.has_initial_states.scalar_type(), torch::kBool);
  EXPECT_EQ(metadata.has_initial_states.device(), torch::Device(torch::kCPU));
  EXPECT_TRUE(torch::equal(metadata.has_initial_states,
                           torch::tensor({false, true, false}, torch::kBool)));
}

TEST(AttentionMetadataBuilderTest, DoesNotDeriveValidityFromContextOrSlot) {
  ModelInputParams params = make_params();
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
  ModelInputParams params = make_params();
  params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;

  AttentionMetadata metadata =
      AttentionMetadataBuilder::build(params, /*enable_mla=*/false);

  ASSERT_TRUE(metadata.has_initial_states.defined());
  EXPECT_TRUE(torch::equal(metadata.has_initial_states,
                           torch::tensor({false, true, false}, torch::kBool)));
}

TEST(AttentionMetadataBuilderTest, SkipsInitialStateMaterializationOnRequest) {
  ModelInputParams params = make_params();
  AttentionMetadataBuildOptions build_options;
  build_options.materialize_linear_state_validity = false;

  AttentionMetadata metadata = AttentionMetadataBuilder::build(
      params, /*enable_mla=*/false, {}, std::nullopt, build_options);

  EXPECT_FALSE(metadata.has_initial_states.defined());
  EXPECT_EQ(params.linear_state_validity_mask,
            LinearStateValidityMask({0, 1, 0}));
  EXPECT_EQ(params.embedding.linear_state_ids, std::vector<int32_t>({4, 2, 9}));
}

TEST(AttentionMetadataBuilderTest,
     SkippingInitialStateMaterializationKeepsValidation) {
  ModelInputParams params = make_params();
  params.linear_state_validity_mask.pop_back();
  AttentionMetadataBuildOptions build_options;
  build_options.materialize_linear_state_validity = false;

  EXPECT_DEATH(
      AttentionMetadataBuilder::build(
          params, /*enable_mla=*/false, {}, std::nullopt, build_options),
      "linear state mask row count mismatch");
}

TEST(AttentionMetadataBuilderTest, DecodeDoesNotMaterializeInitialStateMask) {
  ModelInputParams params = make_params();
  params.meta.batch_forward_type = BatchForwardType::DECODE;

  AttentionMetadata metadata =
      AttentionMetadataBuilder::build(params, /*enable_mla=*/false);

  EXPECT_FALSE(metadata.has_initial_states.defined());
}

TEST(AttentionMetadataBuilderTest, IgnoresTransportIdsWithoutLinearStateOps) {
  ModelInputParams params = make_params();
  params.embedding.linear_state_ids = {-1, -1, -1};
  params.embedding.linear_state_indices =
      torch::tensor({-1, -1, -1}, torch::kInt);
  params.linear_state_validity_mask.clear();

  AttentionMetadata metadata =
      AttentionMetadataBuilder::build(params, /*enable_mla=*/false);

  EXPECT_FALSE(metadata.has_initial_states.defined());
}

TEST(AttentionMetadataBuilderTest, MaterializesColdMaskForDummyShard) {
  ModelInputParams params;
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
  ASSERT_TRUE(metadata.has_initial_states.defined());
  EXPECT_TRUE(
      torch::equal(metadata.has_initial_states, torch::tensor({false})));
}

#if defined(USE_MUSA)
TEST(AttentionMetadataBuilderTest, BuildsMusaMetadataWithCommonBuilder) {
  ModelInputParams params;
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
