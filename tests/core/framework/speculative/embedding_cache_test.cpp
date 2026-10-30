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

#include "core/framework/speculative/embedding_cache.h"

#include <gtest/gtest.h>

#include "core/framework/model/model_output.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/draft_extend_input.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {

namespace {

bool tensor_equal(const torch::Tensor& lhs, const torch::Tensor& rhs) {
  return lhs.defined() && rhs.defined() && torch::equal(lhs, rhs);
}

}  // namespace

TEST(EmbeddingCacheTest, WritePrefillTargetContextAndClear) {
  torch::Device device(Platform::type_torch(), 0);
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

TEST(EmbeddingCacheTest, WritePrefillTargetContextClonesEmbeddingRows) {
  EmbeddingCache cache(/*total_nums=*/2);

  std::vector<int32_t> ids = {0, 1};
  std::vector<std::string> request_ids = {"req_0", "req_1"};
  torch::Tensor target_tokens = torch::tensor({51, 61}, torch::kInt);
  torch::Tensor embeddings = torch::tensor({{3.0f, 4.0f}, {5.0f, 6.0f}});

  cache.write_prefill_target_context(
      ids, request_ids, target_tokens, embeddings);
  // Mutating the source after the write must not affect the stored copy.
  embeddings.fill_(0.0f);

  std::vector<EmbeddingCache::DecodeState> states =
      cache.read_decode_states(ids, request_ids);
  ASSERT_EQ(states.size(), ids.size());
  EXPECT_EQ(states[0].token_id, 51);
  EXPECT_TRUE(tensor_equal(states[0].embedding, torch::tensor({3.0f, 4.0f})));
  EXPECT_EQ(states[1].token_id, 61);
  EXPECT_TRUE(tensor_equal(states[1].embedding, torch::tensor({5.0f, 6.0f})));
}

TEST(EmbeddingCacheTest, WriteValidateTargetContext) {
  torch::Device device(Platform::type_torch(), 0);
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

TEST(EmbeddingCacheTest, CompactContextKeepsOnlyOwnedCurrentPreviousRows) {
  EmbeddingCache cache(/*total_nums=*/2);
  const std::vector<int32_t> ids = {0, 1};
  const std::vector<std::string> requests = {"compact_0", "compact_1"};
  const torch::Tensor tokens =
      torch::tensor({{11, 12, 13, 14}, {21, -1, -1, -1}}, torch::kLong);
  torch::Tensor hidden =
      torch::tensor({{2.0f, 3.0f}, {4.0f, 5.0f}, {6.0f, 7.0f}, {8.0f, 9.0f}});
  cache.write_target_context(ids,
                             requests,
                             tokens,
                             hidden,
                             torch::tensor({3, 0}, torch::kInt),
                             /*num_speculative_tokens=*/3);
  hidden.fill_(-9);
  const auto states = cache.read_decode_states(ids, requests);
  EXPECT_TRUE(tensor_equal(states[0].embedding, torch::tensor({4.0f, 5.0f})));
  EXPECT_TRUE(
      tensor_equal(states[0].prev_embedding, torch::tensor({2.0f, 3.0f})));
  EXPECT_TRUE(tensor_equal(states[1].embedding, torch::tensor({8.0f, 9.0f})));
  EXPECT_FALSE(states[1].prev_embedding.defined());
  EXPECT_EQ(states[0].position_offset, 3);
  EXPECT_EQ(states[1].position_offset, 0);
  EXPECT_TRUE(states[0].all_draft_accepted);
  EXPECT_FALSE(states[1].all_draft_accepted);
}

TEST(EmbeddingCacheTest, UnifiedB1CanRetainImmutableCompactSnapshotViews) {
  EmbeddingCache cache(/*total_nums=*/1);
  const std::vector<int32_t> ids = {0};
  const std::vector<std::string> requests = {"unified"};
  const torch::Tensor tokens = torch::tensor({{11, 12, 13, 14}}, torch::kLong);
  torch::Tensor hidden = torch::tensor({{2.0f, 3.0f}, {4.0f, 5.0f}});

  cache.write_target_context(ids,
                             requests,
                             tokens,
                             hidden,
                             torch::tensor({3}, torch::kInt),
                             /*num_speculative_tokens=*/3,
                             /*allow_immutable_compact_view=*/true);
  hidden.fill_(-9);

  const auto states = cache.read_decode_states(ids, requests);
  ASSERT_EQ(states.size(), 1);
  EXPECT_TRUE(tensor_equal(states[0].embedding, torch::tensor({-9.0f, -9.0f})));
  EXPECT_TRUE(
      tensor_equal(states[0].prev_embedding, torch::tensor({-9.0f, -9.0f})));
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

TEST(EmbeddingCacheTest, ReadAcceptedPrefixLengthsRejectsStaleRequest) {
  EmbeddingCache cache(/*total_nums=*/2);
  std::vector<int32_t> ids = {0, 1};
  std::vector<std::string> request_ids = {"req_0", "req_1"};
  torch::Tensor accepted_tokens =
      torch::tensor({{11, 12, 13}, {21, -1, -1}}, torch::kInt);
  torch::Tensor accepted_embeddings =
      torch::tensor({{{1.0f, 1.1f}, {1.2f, 1.3f}, {1.4f, 1.5f}},
                     {{2.0f, 2.1f}, {2.2f, 2.3f}, {2.4f, 2.5f}}});

  // Only embedding_id 0 receives target output; id 1 stays uninitialized.
  cache.write_target_context({0},
                             {"req_0"},
                             accepted_tokens.slice(/*dim=*/0, 0, 1),
                             accepted_embeddings.slice(/*dim=*/0, 0, 1),
                             /*num_speculative_tokens=*/2);

  // Matching request_id returns correction_position_offset + 1; an
  // uninitialized entry returns 1.
  std::vector<int32_t> lengths =
      cache.read_accepted_prefix_lengths(ids, request_ids);
  ASSERT_EQ(lengths.size(), ids.size());
  EXPECT_EQ(lengths[0], 3);
  EXPECT_EQ(lengths[1], 1);

  // A preempted-and-reused embedding_id now owned by a fresh request must not
  // leak the previous request's correction offset.
  std::vector<int32_t> reused =
      cache.read_accepted_prefix_lengths({0}, {"new_req"});
  ASSERT_EQ(reused.size(), 1u);
  EXPECT_EQ(reused[0], 1);
}

TEST(EmbeddingCacheTest, BootstrapPreservesExistingPrefillContext) {
  EmbeddingCache cache(2);
  torch::Tensor embedding = torch::tensor({{1.0f, 2.0f}});
  cache.write_prefill_target_context(
      {1}, {"request"}, torch::tensor({17}, torch::kInt), embedding);

  cache.write_mtp_bootstrap_context(
      1, "request", 99, torch::tensor({9.0f, 9.0f}));

  const auto states = cache.read_decode_states({1}, {"request"});
  ASSERT_EQ(states.size(), 1);
  EXPECT_EQ(states[0].token_id, 17);
  EXPECT_TRUE(tensor_equal(states[0].embedding, embedding[0]));
}

class BootstrapAfterValidateTest : public ::testing::TestWithParam<int32_t> {};

TEST_P(BootstrapAfterValidateTest, PreservesAcceptedStateAfterOverlapPause) {
  EmbeddingCache cache(2);
  const int32_t accepted_count = GetParam();
  torch::Tensor accepted_tokens = torch::tensor({{11, 12, 13}}, torch::kInt64);
  accepted_tokens.slice(1, accepted_count).fill_(-1);
  torch::Tensor accepted_embeddings =
      torch::tensor({{{1.0f, 1.1f}, {2.0f, 2.1f}, {3.0f, 3.1f}}});
  cache.write_target_context(
      {1}, {"request"}, accepted_tokens, accepted_embeddings, 2);

  cache.write_mtp_bootstrap_context(
      1, "request", 10 + accepted_count, torch::tensor({9.0f, 9.0f}));

  const auto states = cache.read_decode_states({1}, {"request"});
  ASSERT_EQ(states.size(), 1);
  EXPECT_TRUE(states[0].valid);
  EXPECT_EQ(states[0].request_id, "request");
  EXPECT_EQ(states[0].token_id, 10 + accepted_count);
  EXPECT_EQ(states[0].position_offset, accepted_count - 1);
  EXPECT_EQ(states[0].correction_token_id, 10 + accepted_count);
  EXPECT_EQ(states[0].correction_position_offset, accepted_count - 1);
  EXPECT_EQ(states[0].all_draft_accepted, accepted_count == 3);
  EXPECT_TRUE(tensor_equal(states[0].embedding,
                           accepted_embeddings[0][accepted_count - 1]));
  EXPECT_EQ(cache.read_accepted_prefix_lengths({1}, {"request"}),
            std::vector<int32_t>({accepted_count}));
  if (accepted_count > 1) {
    EXPECT_EQ(states[0].prev_token_id, 9 + accepted_count);
    EXPECT_TRUE(tensor_equal(states[0].prev_embedding,
                             accepted_embeddings[0][accepted_count - 2]));
  }
}

INSTANTIATE_TEST_SUITE_P(AcceptedWidths,
                         BootstrapAfterValidateTest,
                         ::testing::Values(1, 2, 3));

TEST(EmbeddingCacheTest, BootstrapInitializesRecycledSlotForNewRequest) {
  EmbeddingCache cache(2);
  cache.write_mtp_bootstrap_context(1, "old", 17, torch::tensor({1.0f, 2.0f}));
  torch::Tensor embedding = torch::tensor({3.0f, 4.0f});
  cache.write_mtp_bootstrap_context(1, "new", 27, embedding);

  const auto states = cache.read_decode_states({1}, {"new"});
  ASSERT_EQ(states.size(), 1);
  EXPECT_TRUE(states[0].valid);
  EXPECT_EQ(states[0].token_id, 27);
  EXPECT_TRUE(tensor_equal(states[0].embedding, embedding));
}

TEST(EmbeddingCacheTest, BootstrapInitializesClearedSlotForSameRequest) {
  EmbeddingCache cache(2);
  cache.write_mtp_bootstrap_context(
      1, "request", 17, torch::tensor({1.0f, 2.0f}));
  cache.clear({1});
  torch::Tensor embedding = torch::tensor({3.0f, 4.0f});
  cache.write_mtp_bootstrap_context(1, "request", 27, embedding);

  const auto states = cache.read_decode_states({1}, {"request"});
  ASSERT_EQ(states.size(), 1);
  EXPECT_TRUE(states[0].valid);
  EXPECT_EQ(states[0].token_id, 27);
  EXPECT_TRUE(tensor_equal(states[0].embedding, embedding));
}

TEST(EmbeddingCacheTest,
     BootstrapWithoutRequestIdentityRetainsOverwriteBehavior) {
  EmbeddingCache cache(2);
  cache.write_mtp_bootstrap_context(1, "", 17, torch::tensor({1.0f, 2.0f}));
  cache.write_mtp_bootstrap_context(1, "", 27, torch::tensor({3.0f, 4.0f}));

  const auto states = cache.read_decode_states({1}, {});
  ASSERT_EQ(states.size(), 1);
  EXPECT_EQ(states[0].token_id, 27);
}

TEST(EmbeddingCacheTest, WriteMtpBootstrapContextStoresExactDecodeState) {
  EmbeddingCache cache(/*total_nums=*/2);
  torch::Tensor embedding = torch::tensor({1.0f, 2.0f, 3.0f});

  cache.write_mtp_bootstrap_context(
      /*embedding_id=*/1,
      /*request_id=*/"req_bootstrap",
      /*token_id=*/17,
      embedding);

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

TEST(EmbeddingCacheTest, SpeculativeHiddenSelectionPreservesDrafterSource) {
  const torch::Tensor primary = torch::tensor({{10.0f, 11.0f}, {20.0f, 21.0f}});
  const torch::Tensor aux = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
  const torch::Tensor indices = torch::tensor({1}, torch::kInt32);
  const torch::Tensor gathered_primary = torch::tensor({{20.0f, 21.0f}});
  const torch::Tensor selected_aux = torch::tensor({{3.0f, 4.0f}});
  const ModelOutput model_output(primary, torch::Tensor(), aux);

  for (const bool is_target_prefill : {false, true}) {
    for (const bool cp_enabled : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "prefill=" << is_target_prefill << ", cp=" << cp_enabled);
      SampleOutput output;
      output_spec_hidden_states(output,
                                model_output,
                                indices,
                                gathered_primary,
                                is_target_prefill,
                                cp_enabled);
      EXPECT_TRUE(tensor_equal(
          is_target_prefill ? output.selected_embeddings : output.embeddings,
          selected_aux));
      if (is_target_prefill) {
        EXPECT_TRUE(tensor_equal(output.embeddings, aux));
      }
    }
  }

  SampleOutput output;
  output_spec_hidden_states(output,
                            ModelOutput(primary.narrow(0, 0, 1)),
                            indices,
                            gathered_primary,
                            /*is_target_prefill=*/false,
                            /*cp_enabled=*/true);
  EXPECT_TRUE(tensor_equal(output.embeddings, gathered_primary));
}

}  // namespace xllm
