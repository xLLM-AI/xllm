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

TEST(AttentionMetadataBuilderTest, DecodeDoesNotMaterializeInitialStateMask) {
  ModelInputParams params = make_params();
  params.meta.batch_forward_type = BatchForwardType::DECODE;

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

}  // namespace
}  // namespace xllm::layer
