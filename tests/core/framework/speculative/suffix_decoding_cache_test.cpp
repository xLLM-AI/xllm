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

#include "core/framework/speculative/suffix_decoding_cache.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace xllm {

TEST(SuffixDecodingCacheTest, StartStopAndActiveSet) {
  SuffixDecodingCache cache(/*max_tree_depth=*/16, /*max_cached_requests=*/4);

  std::vector<int32_t> prompt = {10, 11, 12};
  cache.start_request("req_a", std::span<const int32_t>(prompt));

  EXPECT_TRUE(cache.has_active_request("req_a"));
  EXPECT_TRUE(cache.has_cached_request("req_a"));

  auto active = cache.active_requests();
  EXPECT_EQ(active.size(), 1);
  EXPECT_EQ(active[0], "req_a");

  cache.stop_request("req_a");
  EXPECT_FALSE(cache.has_active_request("req_a"));
  EXPECT_TRUE(cache.has_cached_request("req_a"));
}

TEST(SuffixDecodingCacheTest, AddResponseAndSpeculate) {
  SuffixDecodingCache cache(/*max_tree_depth=*/32, /*max_cached_requests=*/8);

  std::vector<int32_t> prompt = {1, 2, 3};
  cache.start_request("req_a", std::span<const int32_t>(prompt));

  std::vector<int32_t> out = {4, 5, 6, 7};
  cache.add_active_response("req_a", std::span<const int32_t>(out));

  std::vector<int32_t> ctx = {2, 3, 4};
  SuffixDecodingDraft draft = cache.speculate("req_a",
                                              std::span<const int32_t>(ctx),
                                              /*max_spec_tokens=*/4,
                                              /*max_spec_factor=*/2.0f,
                                              /*max_spec_offset=*/0.0f,
                                              /*min_token_prob=*/0.01f,
                                              /*use_tree_spec=*/false);

  EXPECT_FALSE(draft.token_ids.empty());
  EXPECT_EQ(draft.token_ids[0], 5);
}

TEST(SuffixDecodingCacheTest, EvictionByMaxCachedRequests) {
  SuffixDecodingCache cache(/*max_tree_depth=*/16, /*max_cached_requests=*/1);

  std::vector<int32_t> p0 = {1, 2};
  std::vector<int32_t> p1 = {3, 4};
  cache.start_request("req_a", std::span<const int32_t>(p0));
  cache.stop_request("req_a");

  cache.start_request("req_b", std::span<const int32_t>(p1));

  auto cached = cache.cached_requests();
  EXPECT_LE(cached.size(), 1);
  EXPECT_TRUE(cache.has_cached_request("req_b"));
}

TEST(SuffixDecodingCacheTest, AddPromptOnlyDoesNotPolluteGlobalCache) {
  SuffixDecodingCache cache(/*max_tree_depth=*/32, /*max_cached_requests=*/8);

  std::vector<int32_t> p0 = {10, 20, 30};
  cache.start_request("req_a", std::span<const int32_t>(p0));

  std::vector<int32_t> prompt_tail = {40, 50, 60};
  cache.add_active_prompt("req_a", std::span<const int32_t>(prompt_tail));
  cache.stop_request("req_a");

  std::vector<int32_t> p1 = {1, 2, 10, 20, 30, 40};
  cache.start_request("req_b", std::span<const int32_t>(p1));

  std::vector<int32_t> ctx = {20, 30, 40};
  auto draft = cache.speculate("req_b",
                               std::span<const int32_t>(ctx),
                               /*max_spec_tokens=*/3,
                               /*max_spec_factor=*/2.0f,
                               /*max_spec_offset=*/0.0f,
                               /*min_token_prob=*/0.01f,
                               /*use_tree_spec=*/false);

  EXPECT_TRUE(draft.token_ids.empty());
}

TEST(SuffixDecodingCacheTest, GlobalCacheSpeculateAcrossRequests) {
  SuffixDecodingCache cache(/*max_tree_depth=*/32, /*max_cached_requests=*/8);

  std::vector<int32_t> p0 = {10, 20, 30};
  cache.start_request("req_a", std::span<const int32_t>(p0));
  std::vector<int32_t> out0 = {40, 50, 60};
  cache.add_active_response("req_a", std::span<const int32_t>(out0));
  cache.stop_request("req_a");

  std::vector<int32_t> p1 = {1, 2, 10, 20, 30, 40};
  cache.start_request("req_b", std::span<const int32_t>(p1));

  std::vector<int32_t> ctx = {20, 30, 40};
  auto draft = cache.speculate("req_b",
                               std::span<const int32_t>(ctx),
                               /*max_spec_tokens=*/3,
                               /*max_spec_factor=*/2.0f,
                               /*max_spec_offset=*/0.0f,
                               /*min_token_prob=*/0.01f,
                               /*use_tree_spec=*/false);

  EXPECT_FALSE(draft.token_ids.empty());
  EXPECT_EQ(draft.token_ids[0], 50);
}

TEST(SuffixDecodingCacheTest, MaxCachedRequestsZeroDisablesGlobalCache) {
  SuffixDecodingCache cache(/*max_tree_depth=*/16, /*max_cached_requests=*/0);

  std::vector<int32_t> p0 = {1, 2, 3};
  cache.start_request("req_a", std::span<const int32_t>(p0));

  EXPECT_TRUE(cache.has_active_request("req_a"));
  EXPECT_FALSE(cache.has_cached_request("req_a"));

  auto cached = cache.cached_requests();
  EXPECT_TRUE(cached.empty());
}

TEST(SuffixDecodingCacheTest, LocalDraftOwnsResultsAfterCacheDestruction) {
  SuffixDecodingDraft draft;
  {
    SuffixDecodingCache cache(/*max_tree_depth=*/16, /*max_cached_requests=*/0);
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    const std::vector<int32_t> context = {9, 2, 3};
    cache.start_request("local", prompt);
    draft = cache.speculate("local",
                            context,
                            /*max_spec_tokens=*/4,
                            /*max_spec_factor=*/2.0f,
                            /*max_spec_offset=*/0.0f,
                            /*min_token_prob=*/0.1f,
                            /*use_tree_spec=*/false);
    auto second_draft = cache.speculate("local", context);
    EXPECT_EQ(second_draft.token_ids, draft.token_ids);
    cache.stop_request("local");
  }

  EXPECT_EQ(draft.token_ids, (std::vector<int32_t>{4, 5}));
  EXPECT_EQ(draft.parents, (std::vector<int32_t>{-1, 0}));
  EXPECT_EQ(draft.probs, (std::vector<float>{1.0f, 1.0f}));
  EXPECT_FLOAT_EQ(draft.score, 2.0f);
  EXPECT_EQ(draft.match_len, 2);
}

TEST(SuffixDecodingCacheTest,
     GlobalTreeDraftPreservesBranchesAndProbabilities) {
  SuffixDecodingDraft draft;
  {
    SuffixDecodingCache cache(/*max_tree_depth=*/16, /*max_cached_requests=*/8);
    const std::vector<int32_t> prompt = {1};
    const std::vector<int32_t> first_response = {10, 20, 30, 40};
    const std::vector<int32_t> second_response = {10, 20, 31, 41};
    cache.start_request("first", prompt);
    cache.add_active_response("first", first_response);
    cache.stop_request("first");
    cache.start_request("second", prompt);
    cache.add_active_response("second", second_response);
    cache.stop_request("second");

    const std::vector<int32_t> context = {99, 10, 20};
    cache.start_request("current", context);
    draft = cache.speculate("current",
                            context,
                            /*max_spec_tokens=*/4,
                            /*max_spec_factor=*/4.0f,
                            /*max_spec_offset=*/0.0f,
                            /*min_token_prob=*/0.4f,
                            /*use_tree_spec=*/true);
    auto filtered_draft = cache.speculate("current",
                                          context,
                                          /*max_spec_tokens=*/4,
                                          /*max_spec_factor=*/4.0f,
                                          /*max_spec_offset=*/0.0f,
                                          /*min_token_prob=*/0.6f,
                                          /*use_tree_spec=*/true);
    EXPECT_TRUE(filtered_draft.token_ids.empty());
    EXPECT_TRUE(filtered_draft.parents.empty());
    EXPECT_TRUE(filtered_draft.probs.empty());
    EXPECT_FLOAT_EQ(filtered_draft.score, 0.0f);
    cache.evict_cached_response("first");
    cache.evict_cached_response("second");
    cache.stop_request("current");
  }

  ASSERT_EQ(draft.token_ids.size(), 4);
  ASSERT_EQ(draft.parents.size(), draft.token_ids.size());
  ASSERT_EQ(draft.probs.size(), draft.token_ids.size());
  EXPECT_FLOAT_EQ(draft.score, 2.0f);
  EXPECT_EQ(draft.match_len, 2);
  for (int32_t token : {30, 31, 40, 41}) {
    auto position =
        std::find(draft.token_ids.begin(), draft.token_ids.end(), token);
    ASSERT_NE(position, draft.token_ids.end());
    size_t index = static_cast<size_t>(position - draft.token_ids.begin());
    EXPECT_FLOAT_EQ(draft.probs[index], 0.5f);
    if (token < 40) {
      EXPECT_EQ(draft.parents[index], -1);
      continue;
    }
    int32_t parent = draft.parents[index];
    ASSERT_GE(parent, 0);
    ASSERT_LT(static_cast<size_t>(parent), index);
    EXPECT_EQ(draft.token_ids[parent], token - 10);
  }
}

}  // namespace xllm
