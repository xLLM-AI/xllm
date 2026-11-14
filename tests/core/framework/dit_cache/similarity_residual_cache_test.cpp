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

#include "core/framework/dit_cache/similarity_residual_cache.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace xllm {
namespace {

torch::Tensor residual(float value) {
  return torch::full({2, 2}, value, torch::kFloat32);
}

TEST(SimilarityResidualCacheTest,
     MissHistoryAndLinearPredictionMatchHandCalculation) {
  CacheDiTOptions options;
  options.residual_diff_threshold = 0.3F;
  options.linear_residual_prediction = true;
  SimilarityResidualCacheState cache(options);
  EXPECT_FALSE(cache.has_stack_residual());
  EXPECT_FALSE(cache.decide(0, residual(1.0F), 2).hit);
  cache.record_dense(0, residual(1.0F), residual(11.0F));
  EXPECT_FALSE(cache.decide(1, residual(2.0F), 2).hit);
  cache.record_dense(1, residual(2.0F), residual(16.0F));
  const SimilarityResidualCacheDecision first =
      cache.decide(2, residual(2.5F), 2);
  EXPECT_TRUE(first.hit);
  EXPECT_NEAR(first.relative_l1, 0.25F, 1e-6F);
  EXPECT_FLOAT_EQ(first.prediction_scale, 0.5F);
  EXPECT_TRUE(torch::equal(cache.apply(2, residual(2.5F)), residual(18.5F)));
  EXPECT_FALSE(cache.decide(3, residual(2.6F), 2).hit);
  cache.record_dense(3, residual(2.6F), residual(22.6F));
  const SimilarityResidualCacheDecision second =
      cache.decide(4, residual(2.7F), 2);
  EXPECT_TRUE(second.hit);
  EXPECT_NEAR(second.prediction_scale, 1.0F / 6.0F, 1e-6F);
  EXPECT_TRUE(torch::allclose(
      cache.apply(4, residual(2.7F)), residual(23.7F), 1e-6, 1e-6));
  EXPECT_EQ(cache.similarity_checks(), 3);
  EXPECT_EQ(cache.dense_forwards(), 3);
  EXPECT_EQ(cache.cache_hits(), 2);
}

TEST(SimilarityResidualCacheTest,
     QualifiedPresetUsesZeroBasedWarmupAndResetsPerRequest) {
  CacheDiTOptions options;
  options.warmup_steps = 30;
  options.max_cached_steps = 4;
  options.linear_residual_prediction = true;
  for (int32_t request = 0; request < 2; ++request) {
    SimilarityResidualCacheState cache(options);
    std::vector<int64_t> hit_steps;
    int64_t executed_blocks = 0;
    for (int64_t step = 0; step < 49; ++step) {
      const torch::Tensor front = residual(1.0F + request);
      ++executed_blocks;  // The front block really executes on every step.
      if (cache.decide(step, front, 2).hit) {
        hit_steps.push_back(step);
        EXPECT_TRUE(torch::equal(cache.apply(step, front), front + 10.0F));
      } else {
        for (int64_t block = 1; block < 50; ++block) {
          ++executed_blocks;
        }
        cache.record_dense(step, front, front + 10.0F);
      }
    }
    EXPECT_EQ(hit_steps, (std::vector<int64_t>{30, 32, 34, 36}));
    EXPECT_EQ(cache.dense_forwards(), 45);
    EXPECT_EQ(cache.cache_hits(), 4);
    EXPECT_EQ(cache.similarity_checks(), 4);
    EXPECT_EQ(executed_blocks, 2254);
  }
}

TEST(SimilarityResidualCacheTest, SelectedTargetsExcludeConditionsAndPadding) {
  SimilarityResidualCacheState cache(CacheDiTOptions{});
  const torch::Tensor indices = torch::tensor({int64_t{1}}, torch::kInt64);
  const torch::Tensor initial = torch::ones({4, 2});
  EXPECT_FALSE(cache.decide(0, initial, 3, nullptr, indices).hit);
  cache.record_dense(0, initial, initial + 3.0F);
  torch::Tensor next = torch::full({4, 2}, 1000.0F);
  next[1].fill_(1.0F);
  EXPECT_TRUE(cache.decide(1, next, 3, nullptr, indices).hit);
  EXPECT_EQ(cache.similarity_checks(), 1);
}

TEST(SimilarityResidualCacheTest, ZeroCapAndPredictionDisabledAreExplicit) {
  for (const int64_t cap : {0, 1}) {
    CacheDiTOptions options;
    options.max_cached_steps = cap;
    SimilarityResidualCacheState cache(options);
    for (int64_t step = 0; step < 5; ++step) {
      const torch::Tensor front = residual(1.0F);
      if (cache.decide(step, front, 2).hit) {
        EXPECT_TRUE(torch::equal(cache.apply(step, front), residual(4.0F)));
      } else {
        cache.record_dense(step, front, front + 3.0F);
      }
    }
    EXPECT_EQ(cache.cache_hits(), cap);
  }
  CacheDiTOptions options;
  options.residual_diff_threshold = 0.3F;
  options.linear_residual_prediction = false;
  SimilarityResidualCacheState cache(options);
  cache.decide(0, residual(1.0F), 2);
  cache.record_dense(0, residual(1.0F), residual(11.0F));
  cache.decide(1, residual(2.0F), 2);
  cache.record_dense(1, residual(2.0F), residual(16.0F));
  EXPECT_TRUE(cache.decide(2, residual(2.5F), 2).hit);
  EXPECT_TRUE(torch::equal(cache.apply(2, residual(2.5F)), residual(16.5F)));
}

TEST(SimilarityResidualCacheTest, RejectsInvalidOptionsAndIncompatibleHistory) {
  CacheDiTOptions options;
  options.warmup_steps = -1;
  EXPECT_THROW(SimilarityResidualCacheState{options}, std::invalid_argument);
  options = CacheDiTOptions{};
  options.residual_diff_threshold = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(SimilarityResidualCacheState{options}, std::invalid_argument);
  SimilarityResidualCacheState cache(CacheDiTOptions{});
  EXPECT_THROW(cache.apply(0, residual(1.0F)), std::logic_error);
  cache.decide(0, residual(1.0F), 2);
  cache.record_dense(0, residual(1.0F), residual(2.0F));
  EXPECT_THROW(cache.record_dense(0, residual(1.0F), residual(2.0F)),
               std::invalid_argument);
  EXPECT_THROW(cache.decide(1, torch::ones({3, 2}), 2), std::invalid_argument);
  const torch::Tensor empty = torch::empty({0}, torch::kInt64);
  EXPECT_THROW(cache.decide(1, residual(1.0F), 2, nullptr, empty),
               std::invalid_argument);
}

}  // namespace
}  // namespace xllm
