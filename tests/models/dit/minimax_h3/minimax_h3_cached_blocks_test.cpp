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

#include "models/dit/utils/minimax_h3_cached_blocks.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace xllm {
namespace {

// The real world16 test covers inter-rank communication. This boundary records
// the payload/count without starting a device runtime in the controller tests.
class RecordingSumGroup final : public ProcessGroup {
 public:
  RecordingSumGroup() : ProcessGroup(0, 16, torch::Device(torch::kCPU)) {}

  void allreduce(torch::Tensor& statistics) override {
    EXPECT_EQ(statistics.scalar_type(), torch::kFloat32);
    EXPECT_EQ(statistics.sizes(), (torch::IntArrayRef{5}));
    EXPECT_TRUE(statistics.device().is_cpu());
    ++calls;
    statistics.mul_(16.0F);
  }

  int64_t calls = 0;
};

TEST(H3CachedBlocksTest, QualifiedClockExecutesRealCallbacksAndResetsRequests) {
  CacheDiTOptions options;
  options.warmup_steps = 30;
  options.max_cached_steps = 4;
  options.linear_residual_prediction = true;
  const H3QualifiedCacheProbeRows rows{
      .used_rows = 2,
      .target_indices = torch::tensor({int64_t{0}, int64_t{1}}, torch::kInt64)};
  for (int32_t request = 0; request < 2; ++request) {
    SimilarityResidualCacheState cache(options);
    RecordingSumGroup world;
    std::vector<int64_t> hits;
    hits.reserve(4);
    int64_t actual_calls = 0;
    for (int64_t step = 0; step < 49; ++step) {
      std::vector<int64_t> layers;
      layers.reserve(50);
      const torch::Tensor input = torch::full({2, 3}, request * 100.0F);
      const H3QualifiedCachedBlockOutput output =
          run_h3_qualified_cached_blocks(
              input,
              step,
              /*block_count=*/50,
              cache,
              rows,
              &world,
              [&](int64_t layer, const torch::Tensor& hidden) {
                layers.emplace_back(layer);
                ++actual_calls;
                return hidden + static_cast<float>(layer + 1);
              });
      EXPECT_TRUE(torch::equal(
          output.hidden, torch::full({2, 3}, request * 100.0F + 1275.0F)));
      if (output.decision.hit) {
        hits.emplace_back(step);
        EXPECT_EQ(layers, (std::vector<int64_t>{0}));
      } else {
        ASSERT_EQ(layers.size(), 50);
        for (int64_t layer = 0; layer < 50; ++layer) {
          EXPECT_EQ(layers[static_cast<size_t>(layer)], layer);
        }
      }
      EXPECT_TRUE(torch::equal(input, torch::full({2, 3}, request * 100.0F)));
    }
    EXPECT_EQ(hits, (std::vector<int64_t>{30, 32, 34, 36}));
    EXPECT_EQ(actual_calls, 2254);
    EXPECT_EQ(cache.dense_forwards(), 45);
    EXPECT_EQ(cache.cache_hits(), 4);
    EXPECT_EQ(cache.similarity_checks(), 4);
    EXPECT_EQ(world.calls, 4);
  }
}

TEST(H3CachedBlocksTest, BackBlocksRunAfterBothDenseAndReconstructedMiddle) {
  CacheDiTOptions options;
  options.back_blocks = 1;
  options.max_cached_steps = 1;
  SimilarityResidualCacheState cache(options);
  const H3QualifiedCacheProbeRows rows{.used_rows = 1};
  std::vector<int64_t> layers;
  layers.reserve(4);
  const auto execute = [&](int64_t layer, const torch::Tensor& hidden) {
    layers.emplace_back(layer);
    return layer == 3 ? hidden * 10.0F : hidden + static_cast<float>(layer + 1);
  };
  const auto first = run_h3_qualified_cached_blocks(
      torch::zeros({1, 1}), 0, 4, cache, rows, nullptr, execute);
  EXPECT_FALSE(first.decision.hit);
  EXPECT_EQ(layers, (std::vector<int64_t>{0, 1, 2, 3}));
  EXPECT_FLOAT_EQ(first.hidden.item<float>(), 60.0F);
  layers.clear();
  const auto second = run_h3_qualified_cached_blocks(
      torch::full({1, 1}, 2.0F), 1, 4, cache, rows, nullptr, execute);
  EXPECT_TRUE(second.decision.hit);
  EXPECT_EQ(layers, (std::vector<int64_t>{0, 3}));
  EXPECT_FLOAT_EQ(second.hidden.item<float>(), 80.0F);
  EXPECT_EQ(cache.dense_forwards(), 1);
}

TEST(H3CachedBlocksTest, ExplicitZeroCapExecutesEveryBlockWithoutStatistics) {
  CacheDiTOptions options;
  options.max_cached_steps = 0;
  SimilarityResidualCacheState cache(options);
  RecordingSumGroup world;
  const H3QualifiedCacheProbeRows rows{.used_rows = 1};
  int64_t calls = 0;
  for (int64_t step = 0; step < 3; ++step) {
    const auto output = run_h3_qualified_cached_blocks(
        torch::zeros({1, 1}),
        step,
        4,
        cache,
        rows,
        &world,
        [&](int64_t /*layer*/, const torch::Tensor& hidden) {
          ++calls;
          return hidden + 1.0F;
        });
    EXPECT_FALSE(output.decision.hit);
    EXPECT_FLOAT_EQ(output.hidden.item<float>(), 4.0F);
  }
  EXPECT_EQ(calls, 12);
  EXPECT_EQ(world.calls, 0);
  EXPECT_EQ(cache.dense_forwards(), 3);
  EXPECT_EQ(cache.cache_hits(), 0);
}

TEST(H3CachedBlocksTest, BlockFailurePropagatesWithoutRecordingPartialMiddle) {
  SimilarityResidualCacheState cache(CacheDiTOptions{});
  const H3QualifiedCacheProbeRows rows{.used_rows = 1};
  int64_t calls = 0;
  EXPECT_THROW(run_h3_qualified_cached_blocks(
                   torch::zeros({1, 1}),
                   0,
                   4,
                   cache,
                   rows,
                   nullptr,
                   [&](int64_t layer, const torch::Tensor& hidden) {
                     ++calls;
                     if (layer == 1) {
                       throw std::runtime_error("block execution failed");
                     }
                     return hidden + 1.0F;
                   }),
               std::runtime_error);
  EXPECT_EQ(calls, 2);
  EXPECT_FALSE(cache.has_stack_residual());
  EXPECT_EQ(cache.dense_forwards(), 0);
}

}  // namespace
}  // namespace xllm
