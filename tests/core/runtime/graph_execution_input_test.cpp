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

#include "core/runtime/graph_execution_input.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace xllm {
namespace {

TEST(GraphExecutionInputTest, BorrowsUnchangedModelInput) {
  LlmModelParams source;
  source.parallel.dp_global_token_nums = {4, 4};
  ModelParamsView source_view(source);
  GraphExecutionInput<ModelParamsView> input(source_view);

  input.pad_dp_token_counts(4);

  EXPECT_EQ(&input.params(), &source_view);
  EXPECT_EQ(input.params().parallel.dp_global_token_nums.data(),
            source.parallel.dp_global_token_nums.data());
}

TEST(GraphExecutionInputTest, KeepsSingleRankTokenCountUnchanged) {
  RecModelParams source;
  source.parallel.dp_global_token_nums = {2};
  GraphExecutionInput<RecModelParams> input(source);

  input.pad_dp_token_counts(8);

  EXPECT_EQ(&input.params(), &source);
  EXPECT_EQ(source.parallel.dp_global_token_nums, std::vector<int32_t>({2}));
}

TEST(GraphExecutionInputTest, PadsRecMetadataWithoutChangingSource) {
  RecModelParams source;
  source.parallel.dp_global_token_nums = {2, 4};
  source.attention.host.kv_seq_lens = {5, 9};
  source.embedding.input_embedding = torch::ones({2, 3});
  GraphExecutionInput<RecModelParams> input(source);

  input.pad_dp_token_counts(8);

  EXPECT_NE(&input.params(), &source);
  EXPECT_EQ(input.params().parallel.dp_global_token_nums,
            std::vector<int32_t>({8, 8}));
  EXPECT_EQ(source.parallel.dp_global_token_nums, std::vector<int32_t>({2, 4}));
  EXPECT_EQ(input.params().attention.host.kv_seq_lens,
            source.attention.host.kv_seq_lens);
  EXPECT_EQ(input.params().embedding.input_embedding.data_ptr(),
            source.embedding.input_embedding.data_ptr());
}

TEST(GraphExecutionInputTest, ReusesDetachedMetadataForFurtherPreparation) {
  LlmModelParams source;
  source.parallel.dp_global_token_nums = {1, 2};
  source.attention.host.kv_seq_lens = {3, 5};
  ModelParamsView source_view(source);
  GraphExecutionInput<ModelParamsView> input(source_view);

  input.pad_dp_token_counts(4);
  ModelParamsView& prepared = input.mutable_params();
  const int32_t* token_counts = prepared.parallel.dp_global_token_nums.data();
  prepared.attention.host.kv_seq_lens.front() = 7;
  input.pad_dp_token_counts(8);

  EXPECT_EQ(&input.params(), &prepared);
  EXPECT_EQ(input.params().parallel.dp_global_token_nums.data(), token_counts);
  EXPECT_EQ(input.params().parallel.dp_global_token_nums,
            std::vector<int32_t>({8, 8}));
  EXPECT_EQ(source.parallel.dp_global_token_nums, std::vector<int32_t>({1, 2}));
  EXPECT_EQ(source.attention.host.kv_seq_lens.front(), 3);
}

TEST(GraphExecutionInputTest, DetachedViewRetainsOwnerAfterAdapterDestruction) {
  ModelParamsView captured = [] {
    LlmModelParams source;
    source.attention.host.kv_seq_lens = {5, 9};
    source.embedding.input_embedding = torch::ones({2, 3});
    ModelParamsView source_view(source);
    GraphExecutionInput<ModelParamsView> input(source_view);
    input.mutable_params().attention.host.kv_seq_lens.front() = 7;
    return input.params();
  }();

  EXPECT_EQ(captured.attention.host.kv_seq_lens, std::vector<int32_t>({7, 9}));
  EXPECT_EQ(captured.embedding.input_embedding.sum().item<float>(), 6.0F);
}

TEST(GraphExecutionInputTest, SharesTensorStorageWhileIsolatingTensorHandles) {
  VlmModelParams source;
  source.embedding.input_embedding = torch::ones({1, 3});
  ModelParamsView source_view(source);
  GraphExecutionInput<ModelParamsView> input(source_view);
  ModelParamsView& prepared = input.mutable_params();

  prepared.embedding.input_embedding.fill_(2);
  EXPECT_EQ(source.embedding.input_embedding.sum().item<float>(), 6.0F);

  prepared.embedding.input_embedding = torch::zeros({1, 3});
  EXPECT_EQ(source.embedding.input_embedding.sum().item<float>(), 6.0F);
  EXPECT_EQ(prepared.embedding.input_embedding.sum().item<float>(), 0.0F);
}

}  // namespace
}  // namespace xllm
