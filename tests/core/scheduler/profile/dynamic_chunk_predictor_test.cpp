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

#include "core/scheduler/profile/dynamic_chunk_predictor.h"

#include <gtest/gtest.h>

#include <limits>

namespace xllm {
namespace {
std::vector<std::pair<int32_t, double>> measurements(double a,
                                                     double b,
                                                     double c) {
  std::vector<std::pair<int32_t, double>> samples;
  samples.reserve(16);
  for (int32_t length = 256; length <= 4096; length += 256) {
    samples.emplace_back(length, a * length * length + b * length + c);
  }
  return samples;
}

TEST(DynamicChunkPredictorTest, EqualizesIncrementalLatency) {
  DynamicChunkPredictor predictor(/*base_chunk=*/4096,
                                  /*min_chunk=*/64,
                                  /*smooth_factor=*/1.0,
                                  /*alignment=*/64);
  constexpr double kA = 0.00001;
  constexpr double kB = 0.01;
  ASSERT_TRUE(predictor.fit(measurements(kA, kB, 12.0)));
  EXPECT_NEAR(
      predictor.target_latency_ms(), kA * 4096 * 4096 + kB * 4096, 1e-8);
  size_t previous = 4096;
  for (size_t history : {0, 4096, 16384, 65536, 1048576}) {
    const size_t chunk = predictor.predict(
        /*history=*/history, /*remaining=*/2097152, /*budget=*/8192);
    EXPECT_LE(chunk, previous);
    EXPECT_EQ(chunk % 64, 0u);
    const double latency = kA * chunk * (chunk + 2.0 * history) + kB * chunk;
    if (chunk > 64) {
      EXPECT_LE(latency, predictor.target_latency_ms() + 1e-8);
      const double next = chunk + 64;
      EXPECT_GT(kA * next * (next + 2.0 * history) + kB * next,
                predictor.target_latency_ms());
    }
    previous = chunk;
  }
}

TEST(DynamicChunkPredictorTest, CapsBudgetsAndAllowsFinalPartialBlock) {
  DynamicChunkPredictor predictor(/*base_chunk=*/4096,
                                  /*min_chunk=*/256,
                                  /*smooth_factor=*/1.0,
                                  /*alignment=*/256);
  ASSERT_TRUE(
      predictor.fit(measurements(/*a=*/0.00001, /*b=*/0.01, /*c=*/12.0)));
  EXPECT_EQ(
      predictor.predict(/*history=*/0, /*remaining=*/10000, /*budget=*/700),
      512u);
  EXPECT_EQ(
      predictor.predict(/*history=*/0, /*remaining=*/10000, /*budget=*/255),
      0u);
  EXPECT_EQ(predictor.predict(/*history=*/0, /*remaining=*/31, /*budget=*/1000),
            31u);
  EXPECT_EQ(
      predictor.predict(/*history=*/4096, /*remaining=*/129, /*budget=*/1000),
      129u);
  EXPECT_EQ(
      predictor.predict(/*history=*/4096, /*remaining=*/0, /*budget=*/1000),
      0u);
  EXPECT_EQ(
      predictor.predict(/*history=*/4096, /*remaining=*/1000, /*budget=*/0),
      0u);
  EXPECT_EQ(predictor.predict(
                /*history=*/1048576, /*remaining=*/10000, /*budget=*/4096),
            256u);
}

TEST(DynamicChunkPredictorTest, SmoothsAndHandlesLinearCost) {
  DynamicChunkPredictor full(/*base_chunk=*/4096,
                             /*min_chunk=*/256,
                             /*smooth_factor=*/1.0,
                             /*alignment=*/64);
  DynamicChunkPredictor smooth(/*base_chunk=*/4096,
                               /*min_chunk=*/256,
                               /*smooth_factor=*/0.5,
                               /*alignment=*/64);
  ASSERT_TRUE(full.fit(measurements(/*a=*/0.00001, /*b=*/0.01, /*c=*/12.0)));
  ASSERT_TRUE(smooth.fit(measurements(/*a=*/0.00001, /*b=*/0.01, /*c=*/12.0)));
  EXPECT_GT(
      smooth.predict(/*history=*/65536, /*remaining=*/10000, /*budget=*/4096),
      full.predict(/*history=*/65536, /*remaining=*/10000, /*budget=*/4096));
  ASSERT_TRUE(full.fit(measurements(/*a=*/0.0, /*b=*/0.01, /*c=*/12.0)));
  EXPECT_GE(
      full.predict(/*history=*/1048576, /*remaining=*/10000, /*budget=*/4096),
      4032u);
}

TEST(DynamicChunkPredictorTest, RejectsBadDataAndResetsReadiness) {
  DynamicChunkPredictor predictor(/*base_chunk=*/4096,
                                  /*min_chunk=*/256,
                                  /*smooth_factor=*/1.0,
                                  /*alignment=*/64);
  EXPECT_FALSE(predictor.fit({}));
  EXPECT_FALSE(
      predictor.fit(std::vector<std::pair<int32_t, double>>(16, {64, 1.0})));
  EXPECT_FALSE(predictor.fit(measurements(/*a=*/0, /*b=*/0, /*c=*/12)));
  ASSERT_TRUE(predictor.fit(measurements(/*a=*/0.00001, /*b=*/0.01, /*c=*/12)));
  auto invalid = measurements(/*a=*/0.00001, /*b=*/0.01, /*c=*/12);
  invalid[4].second = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(predictor.fit(invalid));
  EXPECT_FALSE(predictor.ready());
  EXPECT_EQ(
      predictor.predict(/*history=*/100000, /*remaining=*/77, /*budget=*/1024),
      77u);
  EXPECT_EQ(predictor.predict(
                /*history=*/100000, /*remaining=*/10000, /*budget=*/1024),
            1024u);
}

TEST(DynamicChunkPredictorTest, ValidatesParameters) {
  EXPECT_DEATH(DynamicChunkPredictor(/*base_chunk=*/4096,
                                     /*min_chunk=*/0,
                                     /*smooth_factor=*/1.0,
                                     /*alignment=*/64),
               "min_chunk");
  EXPECT_DEATH(DynamicChunkPredictor(/*base_chunk=*/4096,
                                     /*min_chunk=*/8192,
                                     /*smooth_factor=*/1.0,
                                     /*alignment=*/64),
               "min_chunk");
  EXPECT_DEATH(DynamicChunkPredictor(/*base_chunk=*/4096,
                                     /*min_chunk=*/256,
                                     /*smooth_factor=*/0.0,
                                     /*alignment=*/64),
               "smooth_factor");
  EXPECT_DEATH(DynamicChunkPredictor(/*base_chunk=*/32,
                                     /*min_chunk=*/16,
                                     /*smooth_factor=*/1.0,
                                     /*alignment=*/64),
               "base_chunk");
}

TEST(DynamicChunkPredictorTest, PlansCausalPipelineChunksWithinOneReservation) {
  DynamicChunkPredictor predictor(/*base_chunk=*/4096,
                                  /*min_chunk=*/256,
                                  /*smooth_factor=*/1.0,
                                  /*alignment=*/128,
                                  /*pipeline_depth=*/3);
  ASSERT_TRUE(
      predictor.fit(measurements(/*a=*/0.00001, /*b=*/0.01, /*c=*/12.0)));
  const auto chunks = predictor.plan(/*history=*/0,
                                     /*remaining=*/20000,
                                     /*budget=*/8192);
  ASSERT_EQ(chunks.size(), 3);
  EXPECT_GT(chunks[0], chunks[1]);
  EXPECT_GT(chunks[1], chunks[2]);
  const size_t total = chunks[0] + chunks[1] + chunks[2];
  EXPECT_LE(total, 8192);
  EXPECT_EQ(predictor.predict_budget(
                /*history=*/0, /*remaining=*/20000, /*budget=*/8192),
            total);
  // Replanning the committed reservation must not introduce a short chunk
  // or exceed the allocation when the budget is applied again by a policy.
  EXPECT_EQ(
      predictor.plan(/*history=*/0, /*remaining=*/20000, /*budget=*/total),
      chunks);
  EXPECT_TRUE(predictor.plan(/*history=*/0, /*remaining=*/20000, /*budget=*/127)
                  .empty());
  EXPECT_EQ(
      predictor.plan(/*history=*/20000, /*remaining=*/31, /*budget=*/8192),
      (std::vector<uint32_t>{31}));
}

TEST(DynamicChunkPredictorTest, UnfittedPipelineKeepsAlignedStaticChunks) {
  DynamicChunkPredictor predictor(/*base_chunk=*/4096,
                                  /*min_chunk=*/256,
                                  /*smooth_factor=*/1.0,
                                  /*alignment=*/128,
                                  /*pipeline_depth=*/2);
  EXPECT_EQ(predictor.plan(/*history=*/0, /*remaining=*/6000, /*budget=*/8192),
            (std::vector<uint32_t>{4096, 1904}));
  EXPECT_EQ(predictor.plan(/*history=*/0, /*remaining=*/10000, /*budget=*/5000),
            (std::vector<uint32_t>{4096, 896}));
}
}  // namespace
}  // namespace xllm
