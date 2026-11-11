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
#include "core/framework/model/model_params_view.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/expanded_decode_metadata_builder.h"

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

TEST(AttentionMetadataBuilderTest, PreservesSharedFullFinalAddresses) {
  ModelParamsView params = make_params();
  const torch::Tensor final_pages =
      torch::tensor({{120, 60}, {122, 124}, {126, 60}}, torch::kInt32);
  const torch::Tensor final_slots =
      torch::tensor({15360, 15875, -1}, torch::kInt32);
  params.attention.device.block_tables =
      torch::tensor({{1, 0}, {2, 3}, {4, 0}}, torch::kInt32);
  params.attention.device.new_cache_slots =
      torch::tensor({128, 259, 0}, torch::kInt32);
  params.attention.device.shared_full_block_table = final_pages;
  params.attention.device.shared_full_slot_mapping = final_slots;

  for (const auto forward_type : {BatchForwardType::PREFILL,
                                  BatchForwardType::CHUNKED_PREFILL,
                                  BatchForwardType::DECODE}) {
    params.meta.batch_forward_type = forward_type;
    const AttentionMetadata metadata =
        AttentionMetadataBuilder::build(params, /*enable_mla=*/false);

    EXPECT_EQ(metadata.shared_full_block_table.data_ptr(),
              final_pages.data_ptr());
    EXPECT_EQ(metadata.shared_full_slot_mapping.data_ptr(),
              final_slots.data_ptr());
    EXPECT_TRUE(torch::equal(metadata.slot_mapping,
                             params.attention.device.new_cache_slots));
    EXPECT_EQ(metadata.shared_full_slot_mapping[2].item<int32_t>(), -1);
  }
}

TEST(AttentionMetadataBuilderTest, PreservesExpandedSharedFullFinalAddresses) {
  ModelParamsView params = make_params();
  params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  params.is_spec_verify = true;
  params.attention.device.shared_full_block_table = torch::tensor(
      {{120, 60}, {120, 60}, {122, 124}, {122, 124}}, torch::kInt32);
  params.attention.device.shared_full_slot_mapping =
      torch::tensor({15362, 15363, 15622, 15623}, torch::kInt32);
  ExpandedDecodeMetadataBuilder::populate_expanded_layout(
      params,
      torch::tensor({3, 4, 7, 8}, torch::kInt32),
      torch::tensor({{1, 0}, {1, 0}, {2, 3}, {2, 3}}, torch::kInt32),
      /*expanded_host_kv_seq_lens=*/{3, 4, 7, 8},
      /*block_size=*/128);

  const AttentionMetadata metadata =
      AttentionMetadataBuilder::build(params, /*enable_mla=*/false);

  ASSERT_TRUE(metadata.expanded_decode.enabled);
  EXPECT_EQ(metadata.shared_full_block_table.sizes(),
            metadata.expanded_decode.block_table.sizes());
  EXPECT_EQ(metadata.shared_full_block_table.data_ptr(),
            params.attention.device.shared_full_block_table.data_ptr());
  EXPECT_EQ(metadata.shared_full_slot_mapping.data_ptr(),
            params.attention.device.shared_full_slot_mapping.data_ptr());
  EXPECT_TRUE(torch::equal(metadata.expanded_decode.block_table,
                           params.graph.expanded_block_tables));
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
  EXPECT_EQ(metadata.q_seq_lens_vec, std::vector<int32_t>({1}));
  EXPECT_EQ(metadata.kv_seq_lens_vec, std::vector<int32_t>({1}));
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
