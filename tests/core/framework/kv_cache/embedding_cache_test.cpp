/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "embedding_cache.h"

#include <gtest/gtest.h>

#include "platform/device.h"

namespace xllm {

namespace {

bool tensor_equal(const torch::Tensor& lhs, const torch::Tensor& rhs) {
  return lhs.defined() && rhs.defined() && torch::equal(lhs, rhs);
}

}  // namespace

TEST(EmbeddingCacheTest, WritePrefillTargetContextAndClear) {
  torch::Device device(Device::type_torch(), 0);
  EmbeddingCache cache(/*total_nums=*/4);

  std::vector<int32_t> ids = {3, 2};
  std::vector<std::string> request_ids = {"req_0", "req_1"};
  torch::Tensor target_tokens = torch::tensor({31, 41}, torch::kInt);
  torch::Tensor target_embeddings = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});

  cache.write_prefill_target_context(
      ids, request_ids, target_tokens, target_embeddings);

  std::vector<EmbeddingCache::DecodeState> states =
      cache.read_decode_states(ids, request_ids);
  ASSERT_EQ(states.size(), ids.size());
  EXPECT_TRUE(states[0].valid);
  EXPECT_EQ(states[0].request_id, "req_0");
  EXPECT_EQ(states[0].token_id, 31);
  EXPECT_EQ(states[0].position_offset, 0);
  EXPECT_FALSE(states[0].all_draft_accepted);
  EXPECT_EQ(states[0].prev_token_id, -1);
  EXPECT_TRUE(tensor_equal(states[0].embedding, target_embeddings[0]));
  EXPECT_TRUE(states[1].valid);
  EXPECT_EQ(states[1].token_id, 41);
  EXPECT_EQ(states[1].position_offset, 0);
  EXPECT_TRUE(tensor_equal(states[1].embedding, target_embeddings[1]));

  cache.clear(ids);
  states = cache.read_decode_states(ids, request_ids);
  EXPECT_FALSE(states[0].valid);
  EXPECT_EQ(states[0].token_id, 0);
  EXPECT_EQ(states[0].position_offset, 0);
  EXPECT_FALSE(states[0].embedding.defined());
  EXPECT_FALSE(states[1].valid);
  EXPECT_EQ(states[1].token_id, 0);
  EXPECT_EQ(states[1].position_offset, 0);
  EXPECT_FALSE(states[1].embedding.defined());
}

TEST(EmbeddingCacheTest, WritePrefillTargetContextSelectsEmbeddings) {
  EmbeddingCache cache(/*total_nums=*/4);

  std::vector<int32_t> ids = {1, 2};
  std::vector<std::string> request_ids = {"req_0", "req_1"};
  torch::Tensor target_tokens = torch::tensor({51, 61}, torch::kInt);
  torch::Tensor full_embeddings =
      torch::tensor({{1.0f, 1.1f}, {2.0f, 2.1f}, {3.0f, 3.1f}});
  torch::Tensor selected_idxes = torch::tensor({2, 0}, torch::kInt);

  cache.write_prefill_target_context(
      ids, request_ids, target_tokens, full_embeddings, selected_idxes);

  std::vector<EmbeddingCache::DecodeState> states =
      cache.read_decode_states(ids, request_ids);
  ASSERT_EQ(states.size(), ids.size());
  EXPECT_EQ(states[0].token_id, 51);
  EXPECT_TRUE(tensor_equal(states[0].embedding, full_embeddings[2]));
  EXPECT_EQ(states[1].token_id, 61);
  EXPECT_TRUE(tensor_equal(states[1].embedding, full_embeddings[0]));
}

TEST(EmbeddingCacheTest, WriteValidateTargetContext) {
  torch::Device device(Device::type_torch(), 0);
  EmbeddingCache cache(/*total_nums=*/2);
  std::vector<int32_t> ids = {0, 1};
  std::vector<std::string> request_ids = {"req_0", "req_1"};
  torch::Tensor accepted_tokens =
      torch::tensor({{11, 12, 13}, {21, -1, -1}}, torch::kInt);
  torch::Tensor accepted_embeddings =
      torch::tensor({{{1.0f, 1.1f}, {1.2f, 1.3f}, {1.4f, 1.5f}},
                     {{2.0f, 2.1f}, {2.2f, 2.3f}, {2.4f, 2.5f}}});

  cache.write_target_context(ids,
                             request_ids,
                             accepted_tokens,
                             accepted_embeddings,
                             /*num_speculative_tokens=*/2);

  std::vector<EmbeddingCache::DecodeState> states =
      cache.read_decode_states(ids, request_ids);
  EXPECT_EQ(states[0].token_id, 13);
  EXPECT_EQ(states[0].position_offset, 2);
  EXPECT_TRUE(states[0].all_draft_accepted);
  EXPECT_EQ(states[0].prev_token_id, 12);
  EXPECT_TRUE(
      tensor_equal(states[0].prev_embedding, accepted_embeddings[0][1]));
  EXPECT_TRUE(tensor_equal(states[0].embedding, accepted_embeddings[0][2]));

  EXPECT_EQ(states[1].token_id, 21);
  EXPECT_EQ(states[1].position_offset, 0);
  EXPECT_FALSE(states[1].all_draft_accepted);
  EXPECT_EQ(states[1].prev_token_id, -1);
  EXPECT_TRUE(tensor_equal(states[1].embedding, accepted_embeddings[1][0]));
}

TEST(EmbeddingCacheTest, RequestMismatchMaterializesMissingState) {
  EmbeddingCache cache(/*total_nums=*/2);
  std::vector<int32_t> ids = {0};
  std::vector<std::string> request_ids = {"old_req"};
  torch::Tensor target_tokens = torch::tensor({31}, torch::kInt);
  torch::Tensor target_embeddings = torch::tensor({{1.0f, 2.0f}});

  cache.write_prefill_target_context(
      ids, request_ids, target_tokens, target_embeddings);

  std::vector<EmbeddingCache::DecodeState> states =
      cache.read_decode_states(ids, {"new_req"});
  ASSERT_EQ(states.size(), ids.size());
  EXPECT_FALSE(states[0].valid);
  EXPECT_EQ(states[0].token_id, 0);
  EXPECT_FALSE(states[0].embedding.defined());
}

TEST(EmbeddingCacheTest, WriteMtpBootstrapContextStoresExactDecodeState) {
  EmbeddingCache cache(/*total_nums=*/2);
  torch::Tensor embedding = torch::tensor({1.0f, 2.0f, 3.0f});

  cache.write_mtp_bootstrap_context(
      /*embedding_id=*/1, "req_bootstrap", /*token_id=*/17, embedding);

  std::vector<EmbeddingCache::DecodeState> states =
      cache.read_decode_states({1}, {"req_bootstrap"});
  ASSERT_EQ(states.size(), 1u);
  EXPECT_TRUE(states[0].valid);
  EXPECT_EQ(states[0].request_id, "req_bootstrap");
  EXPECT_EQ(states[0].token_id, 17);
  EXPECT_EQ(states[0].position_offset, 0);
  EXPECT_FALSE(states[0].all_draft_accepted);
  EXPECT_TRUE(tensor_equal(states[0].embedding, embedding));

  cache.clear({1});
  states = cache.read_decode_states({1}, {"req_bootstrap"});
  EXPECT_FALSE(states[0].valid);
  EXPECT_FALSE(states[0].embedding.defined());
}

}  // namespace xllm
