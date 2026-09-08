/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "common/metrics.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <thread>
#include <vector>

namespace xllm {
namespace {

TEST(PrefixCacheHitMetricsTest, UpdatesCumulativeTokenHitRate) {
  const double prompt_before = COUNTER_VALUE(prefix_cache_prompt_tokens_total);
  const double hit_before = COUNTER_VALUE(prefix_cache_hit_tokens_total);

  record_prefix_cache_hit_metrics(/*prompt_tokens=*/100, /*hit_tokens=*/40);
  record_prefix_cache_hit_metrics(/*prompt_tokens=*/50, /*hit_tokens=*/30);

  const double expected_prompt = prompt_before + 150.0;
  const double expected_hit = hit_before + 70.0;
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(prefix_cache_prompt_tokens_total),
                   expected_prompt);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(prefix_cache_hit_tokens_total), expected_hit);
  EXPECT_DOUBLE_EQ(GAUGE_VALUE(prefix_cache_token_hit_rate_perc),
                   expected_hit * 100.0 / expected_prompt);
}

TEST(PrefixCacheHitMetricsTest, ClampsHitsAndIgnoresZeroPrompt) {
  const double prompt_before = COUNTER_VALUE(prefix_cache_prompt_tokens_total);
  const double hit_before = COUNTER_VALUE(prefix_cache_hit_tokens_total);

  record_prefix_cache_hit_metrics(/*prompt_tokens=*/10, /*hit_tokens=*/20);
  const double rate_after_clamp = GAUGE_VALUE(prefix_cache_token_hit_rate_perc);
  record_prefix_cache_hit_metrics(/*prompt_tokens=*/0, /*hit_tokens=*/20);

  EXPECT_DOUBLE_EQ(COUNTER_VALUE(prefix_cache_prompt_tokens_total),
                   prompt_before + 10.0);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(prefix_cache_hit_tokens_total),
                   hit_before + 10.0);
  EXPECT_DOUBLE_EQ(GAUGE_VALUE(prefix_cache_token_hit_rate_perc),
                   rate_after_clamp);
}

TEST(PrefixCacheHitMetricsTest, KeepsCountersAndGaugeConsistentConcurrently) {
  constexpr size_t kThreadCount = 8;
  constexpr size_t kUpdatesPerThread = 100;
  const double prompt_before = COUNTER_VALUE(prefix_cache_prompt_tokens_total);
  const double hit_before = COUNTER_VALUE(prefix_cache_hit_tokens_total);
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (size_t thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    threads.emplace_back([]() {
      for (size_t update = 0; update < kUpdatesPerThread; ++update) {
        record_prefix_cache_hit_metrics(/*prompt_tokens=*/4, /*hit_tokens=*/1);
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  const double expected_prompt =
      prompt_before + static_cast<double>(kThreadCount * kUpdatesPerThread * 4);
  const double expected_hit =
      hit_before + static_cast<double>(kThreadCount * kUpdatesPerThread);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(prefix_cache_prompt_tokens_total),
                   expected_prompt);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(prefix_cache_hit_tokens_total), expected_hit);
  EXPECT_DOUBLE_EQ(GAUGE_VALUE(prefix_cache_token_hit_rate_perc),
                   expected_hit * 100.0 / expected_prompt);
}

}  // namespace
}  // namespace xllm
