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

#include "core/runtime/mtp_async_state.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace xllm::mtp_async {
namespace {

TEST(MtpAsyncStateTest, ClassifiesSupportedCombinedDraftExecutionPaths) {
  EXPECT_EQ(classify_combined_draft_execution_path("qwen3_5_mtp"),
            CombinedDraftExecutionPath::QWEN3_5_PAGED_ATTENTION);
  EXPECT_EQ(classify_combined_draft_execution_path("qwen3_5_moe_mtp"),
            CombinedDraftExecutionPath::QWEN3_5_PAGED_ATTENTION);
  EXPECT_EQ(classify_combined_draft_execution_path("glm_moe_dsa_mtp"),
            CombinedDraftExecutionPath::GLM5_DSA);
  EXPECT_EQ(classify_combined_draft_execution_path("mimo_mtp"),
            CombinedDraftExecutionPath::UNSUPPORTED);
}

TEST(MtpAsyncStateTest, BuildsMixedAcceptanceStateWithoutHostRoundTrip) {
  const torch::Tensor accepted_tokens = torch::tensor(
      {{10, 11, 12, 13}, {20, 21, -1, -1}, {30, -1, -1, -1}}, torch::kLong);
  const torch::Tensor accepted_embeddings =
      torch::arange(24, torch::kFloat).reshape({3, 4, 2});
  const torch::Tensor placeholder = torch::tensor({-100.0, -101.0});
  const torch::Tensor base_positions = torch::tensor({100, 200, 300});
  const torch::Tensor base_kv_seq_lens = torch::tensor({101, 201, 301});

  const AcceptedState state = build_accepted_state(accepted_tokens,
                                                   accepted_embeddings,
                                                   placeholder,
                                                   base_positions,
                                                   base_kv_seq_lens);

  EXPECT_TRUE(torch::equal(state.accepted_lengths,
                           torch::tensor({4, 2, 1}, torch::kLong)));
  EXPECT_TRUE(torch::equal(state.all_draft_accepted,
                           torch::tensor({true, false, false})));
  EXPECT_TRUE(torch::equal(state.last_tokens, torch::tensor({13, 21, 30})));
  EXPECT_TRUE(torch::equal(state.previous_tokens, torch::tensor({12, 20, 30})));
  EXPECT_TRUE(torch::equal(state.last_embeddings,
                           torch::stack({accepted_embeddings[0][3],
                                         accepted_embeddings[1][1],
                                         accepted_embeddings[2][0]})));
  EXPECT_TRUE(torch::equal(state.previous_embeddings,
                           torch::stack({accepted_embeddings[0][2],
                                         accepted_embeddings[1][0],
                                         placeholder})));
  EXPECT_TRUE(torch::equal(state.base_positions,
                           torch::tensor({104, 202, 301}, torch::kLong)));
  EXPECT_TRUE(torch::equal(state.base_kv_seq_lens,
                           torch::tensor({105, 203, 302}, torch::kLong)));
}

TEST(MtpAsyncStateTest, BuildsRowMetadataForChunkedAndDecodeLayouts) {
  AcceptedState state;
  state.base_positions = torch::tensor({104, 202, 301}, torch::kLong);
  state.base_kv_seq_lens = torch::tensor({105, 203, 302}, torch::kLong);
  const torch::Tensor offsets = torch::tensor({-1, 0}, torch::kLong);

  EXPECT_TRUE(torch::equal(
      make_row_positions(state, offsets),
      torch::tensor({{103, 104}, {201, 202}, {300, 301}}, torch::kLong)));
  EXPECT_TRUE(torch::equal(
      make_kv_seq_lens(state, offsets, /*use_chunked_prefill=*/true),
      state.base_kv_seq_lens));
  EXPECT_TRUE(torch::equal(
      make_kv_seq_lens(state, offsets, /*use_chunked_prefill=*/false),
      torch::tensor({104, 105, 202, 203, 301, 302}, torch::kLong)));
}

TEST(MtpAsyncStateTest, RedirectsUnusedRepairRowsToScratchPositions) {
  AcceptedState state;
  state.base_positions = torch::tensor({104, 202, 301}, torch::kLong);
  state.all_draft_accepted = torch::tensor({true, false, false});

  EXPECT_TRUE(torch::equal(make_repair_cache_positions(state),
                           torch::tensor({103, 203, 302}, torch::kLong)));
}

TEST(MtpAsyncStateTest, MapsPositionsAcrossCacheBlockBoundaries) {
  const torch::Tensor block_tables =
      torch::tensor({{10, 11, 12}, {20, 21, 22}}, torch::kInt);
  const torch::Tensor positions = torch::tensor({{3, 4}, {7, 8}}, torch::kLong);

  EXPECT_TRUE(torch::equal(
      map_positions_to_cache_slots(block_tables, positions, /*block_size=*/4),
      torch::tensor({43, 44, 87, 88}, torch::kInt)));
}

}  // namespace
}  // namespace xllm::mtp_async
