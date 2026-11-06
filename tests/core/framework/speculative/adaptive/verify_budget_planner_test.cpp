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

#include "core/framework/speculative/adaptive/verify_budget_planner.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <vector>

namespace xllm {
namespace {

VerifyBudgetOptions profile_options() {
  VerifyBudgetOptions options;
  options.fixed_draft_time_ms = 1.0;
  options.commit_time_ms = 0.5;
  return options;
}

const std::array<SpecGraphCost, 4> kCosts = {SpecGraphCost{2, 1.0},
                                             {4, 2.0},
                                             {8, 4.0},
                                             {16, 8.0}};

const std::array<SpecGraphCost, 9> kCalibrationGraphs = {SpecGraphCost{1, 0.0},
                                                         {2, 0.0},
                                                         {4, 0.0},
                                                         {8, 0.0},
                                                         {16, 0.0},
                                                         {32, 0.0},
                                                         {48, 0.0},
                                                         {64, 0.0},
                                                         {128, 0.0}};

TEST(VerifyBudgetPlannerTest, SupportsConfiguredDraftWidthAtCommonBatchSizes) {
  for (const uint32_t draft_width : {1U, 3U, 5U, 6U, 7U, 8U}) {
    for (const uint32_t batch_size : {1U, 8U, 16U}) {
      SCOPED_TRACE(draft_width);
      SCOPED_TRACE(batch_size);
      auto options = profile_options();
      options.draft_width = draft_width;
      const std::array<SpecGraphCost, 2> costs = {
          SpecGraphCost{batch_size, 1.0},
          {batch_size * (draft_width + 1), 8.0}};
      const auto full =
          VerifyBudgetPlanner::plan(batch_size, {}, costs, options);
      ASSERT_TRUE(full);
      EXPECT_EQ(full->draft_budget, batch_size * draft_width);
      EXPECT_EQ(full->graph_rows, batch_size * (draft_width + 1));
      const std::vector<double> confidence(batch_size * draft_width, 0.0);
      const auto pruned =
          VerifyBudgetPlanner::plan(batch_size, confidence, costs, options);
      ASSERT_TRUE(pruned);
      EXPECT_EQ(pruned->draft_budget, 0U);
      EXPECT_EQ(pruned->graph_rows, batch_size);
      const auto calibration = VerifyBudgetPlanner::calibration_plan(
          batch_size, batch_size, costs.back().graph_rows, costs, draft_width);
      ASSERT_TRUE(calibration);
      EXPECT_EQ(calibration->draft_budget, batch_size * draft_width);
    }
  }
}

TEST(VerifyBudgetPlannerTest, RejectsInvalidWidthsAndMismatchedFeedback) {
  auto options = profile_options();
  for (const uint32_t draft_width :
       {0U, std::numeric_limits<uint32_t>::max()}) {
    options.draft_width = draft_width;
    EXPECT_FALSE(VerifyBudgetPlanner::plan(1, {}, kCosts, options));
    EXPECT_FALSE(
        VerifyBudgetPlanner::calibration_plan(1, 1, 8, kCosts, draft_width));
  }
  options.draft_width = 5;
  EXPECT_FALSE(VerifyBudgetPlanner::plan(
      1, std::vector<double>(7, 0.5), kCosts, options));
  options.draft_width = 7;
  EXPECT_FALSE(VerifyBudgetPlanner::plan(
      1, std::vector<double>(5, 0.5), kCosts, options));
}

TEST(VerifyBudgetPlannerTest, CalibrationUsesKnownTierWithoutCostMeasurements) {
  const auto plan = VerifyBudgetPlanner::calibration_plan(
      /*logical_requests=*/8,
      /*physical_requests=*/8,
      /*requested_graph_rows=*/16,
      kCalibrationGraphs);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->graph_rows, 16U);
  EXPECT_EQ(plan->draft_budget, 8U);
}

TEST(VerifyBudgetPlannerTest, CalibrationAllowsSmallTransientRequestBuckets) {
  const auto plan = VerifyBudgetPlanner::calibration_plan(
      /*logical_requests=*/1,
      /*physical_requests=*/1,
      /*requested_graph_rows=*/64,
      kCalibrationGraphs);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->graph_rows, 8U);
  EXPECT_EQ(plan->draft_budget, 7U);
}

TEST(VerifyBudgetPlannerTest, CalibrationAllowsLargerTransientRequestBuckets) {
  const auto plan = VerifyBudgetPlanner::calibration_plan(
      /*logical_requests=*/16,
      /*physical_requests=*/16,
      /*requested_graph_rows=*/8,
      kCalibrationGraphs);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->graph_rows, 128U);
  EXPECT_EQ(plan->draft_budget, 112U);
}

TEST(VerifyBudgetPlannerTest, CalibrationUsesLogicalBudgetAndPhysicalCapacity) {
  const auto plan = VerifyBudgetPlanner::calibration_plan(
      /*logical_requests=*/3,
      /*physical_requests=*/4,
      /*requested_graph_rows=*/4,
      kCalibrationGraphs);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->graph_rows, 4U);
  EXPECT_EQ(plan->draft_budget, 1U);
  const auto transient = VerifyBudgetPlanner::calibration_plan(
      /*logical_requests=*/3,
      /*physical_requests=*/3,
      /*requested_graph_rows=*/64,
      kCalibrationGraphs);
  ASSERT_TRUE(transient);
  EXPECT_EQ(transient->graph_rows, 32U);
  EXPECT_EQ(transient->draft_budget, 21U);
}

TEST(VerifyBudgetPlannerTest, CalibrationSupportsAnchorOnlyVerification) {
  const auto plan = VerifyBudgetPlanner::calibration_plan(
      /*logical_requests=*/8,
      /*physical_requests=*/8,
      /*requested_graph_rows=*/8,
      kCalibrationGraphs);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->graph_rows, 8U);
  EXPECT_EQ(plan->draft_budget, 0U);
}

TEST(VerifyBudgetPlannerTest, CalibrationUsesWholeRoundBindingForPartialBatch) {
  const std::array<SpecGraphCost, 4> actual = {
      SpecGraphCost{8, 0.0}, {16, 0.0}, {32, 0.0}, {48, 0.0}};
  const auto transient = VerifyBudgetPlanner::calibration_plan(
      /*logical_requests=*/5,
      /*physical_requests=*/8,
      /*requested_graph_rows=*/64,
      kCalibrationGraphs,
      /*draft_width=*/7,
      actual);
  ASSERT_TRUE(transient);
  EXPECT_EQ(transient->graph_rows, 48U);
  EXPECT_EQ(transient->draft_budget, 35U);
  const auto available = VerifyBudgetPlanner::calibration_plan(
      /*logical_requests=*/5,
      /*physical_requests=*/8,
      /*requested_graph_rows=*/16,
      kCalibrationGraphs,
      /*draft_width=*/7,
      actual);
  ASSERT_TRUE(available);
  EXPECT_EQ(available->graph_rows, 16U);
  EXPECT_EQ(available->draft_budget, 11U);
}

TEST(VerifyBudgetPlannerTest, CalibrationRejectsStableMissingWholeRoundTier) {
  const std::array<SpecGraphCost, 4> actual = {
      SpecGraphCost{8, 0.0}, {16, 0.0}, {32, 0.0}, {64, 0.0}};
  EXPECT_FALSE(
      VerifyBudgetPlanner::calibration_plan(/*logical_requests=*/8,
                                            /*physical_requests=*/8,
                                            /*requested_graph_rows=*/48,
                                            kCalibrationGraphs,
                                            /*draft_width=*/7,
                                            actual));
  EXPECT_FALSE(
      VerifyBudgetPlanner::calibration_plan(/*logical_requests=*/5,
                                            /*physical_requests=*/8,
                                            /*requested_graph_rows=*/64,
                                            kCalibrationGraphs,
                                            /*draft_width=*/7,
                                            std::span(actual).first(3)));
  const std::array<SpecGraphCost, 2> unknown = {SpecGraphCost{8, 0.0},
                                                {40, 0.0}};
  EXPECT_FALSE(
      VerifyBudgetPlanner::calibration_plan(/*logical_requests=*/5,
                                            /*physical_requests=*/8,
                                            /*requested_graph_rows=*/64,
                                            kCalibrationGraphs,
                                            /*draft_width=*/7,
                                            unknown));
}

TEST(VerifyBudgetPlannerTest, CalibrationUsesDraftFiveWholeRoundCoverage) {
  const std::array<SpecGraphCost, 4> actual = {
      SpecGraphCost{8, 0.0}, {16, 0.0}, {32, 0.0}, {48, 0.0}};
  const auto plan = VerifyBudgetPlanner::calibration_plan(
      /*logical_requests=*/8,
      /*physical_requests=*/8,
      /*requested_graph_rows=*/48,
      kCalibrationGraphs,
      /*draft_width=*/5,
      actual);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->graph_rows, 48U);
  EXPECT_EQ(plan->draft_budget, 40U);
  const auto transient = VerifyBudgetPlanner::calibration_plan(
      /*logical_requests=*/5,
      /*physical_requests=*/8,
      /*requested_graph_rows=*/48,
      kCalibrationGraphs,
      /*draft_width=*/5,
      std::span(actual).first(3),
      /*calibration_requests=*/8);
  ASSERT_TRUE(transient);
  EXPECT_EQ(transient->graph_rows, 32U);
  EXPECT_EQ(transient->draft_budget, 25U);
}

TEST(VerifyBudgetPlannerTest,
     CalibrationRecognizesDifferentWholeRoundRequestSet) {
  const std::array<SpecGraphCost, 4> actual = {
      SpecGraphCost{8, 0.0}, {16, 0.0}, {32, 0.0}, {64, 0.0}};
  // A sample for B=10 can use G=48. An interim B=Bcap=8 cannot replay
  // that whole graph, but can advance at its prewarmed full-width G=64.
  const auto transient = VerifyBudgetPlanner::calibration_plan(
      /*logical_requests=*/8,
      /*physical_requests=*/8,
      /*requested_graph_rows=*/48,
      kCalibrationGraphs,
      /*draft_width=*/7,
      actual,
      /*calibration_requests=*/10);
  ASSERT_TRUE(transient);
  EXPECT_EQ(transient->graph_rows, 64U);
  EXPECT_EQ(transient->draft_budget, 56U);
  EXPECT_FALSE(
      VerifyBudgetPlanner::calibration_plan(/*logical_requests=*/8,
                                            /*physical_requests=*/8,
                                            /*requested_graph_rows=*/48,
                                            kCalibrationGraphs,
                                            /*draft_width=*/7,
                                            actual,
                                            /*calibration_requests=*/8));
}

TEST(VerifyBudgetPlannerTest,
     CalibrationRejectsUnknownTierOrInsufficientCapacity) {
  EXPECT_FALSE(VerifyBudgetPlanner::calibration_plan(/*logical_requests=*/1,
                                                     /*physical_requests=*/1,
                                                     /*requested_graph_rows=*/3,
                                                     kCalibrationGraphs));
  EXPECT_FALSE(VerifyBudgetPlanner::calibration_plan(/*logical_requests=*/1,
                                                     /*physical_requests=*/1,
                                                     /*requested_graph_rows=*/0,
                                                     kCalibrationGraphs));
  const std::array<SpecGraphCost, 2> insufficient = {SpecGraphCost{8, 0.0},
                                                     {16, 0.0}};
  EXPECT_FALSE(
      VerifyBudgetPlanner::calibration_plan(/*logical_requests=*/8,
                                            /*physical_requests=*/8,
                                            /*requested_graph_rows=*/16,
                                            insufficient));
  EXPECT_FALSE(VerifyBudgetPlanner::calibration_plan(/*logical_requests=*/1,
                                                     /*physical_requests=*/1,
                                                     /*requested_graph_rows=*/8,
                                                     {}));
}

TEST(VerifyBudgetPlannerTest,
     CalibrationRejectsInvalidGeometryAndTierOrdering) {
  EXPECT_FALSE(VerifyBudgetPlanner::calibration_plan(/*logical_requests=*/0,
                                                     /*physical_requests=*/1,
                                                     /*requested_graph_rows=*/8,
                                                     kCalibrationGraphs));
  EXPECT_FALSE(VerifyBudgetPlanner::calibration_plan(/*logical_requests=*/1,
                                                     /*physical_requests=*/0,
                                                     /*requested_graph_rows=*/8,
                                                     kCalibrationGraphs));
  EXPECT_FALSE(VerifyBudgetPlanner::calibration_plan(/*logical_requests=*/2,
                                                     /*physical_requests=*/1,
                                                     /*requested_graph_rows=*/8,
                                                     kCalibrationGraphs));
  EXPECT_FALSE(VerifyBudgetPlanner::calibration_plan(
      /*logical_requests=*/1,
      /*physical_requests=*/std::numeric_limits<uint32_t>::max(),
      /*requested_graph_rows=*/8,
      kCalibrationGraphs));
  const std::array<SpecGraphCost, 2> duplicated = {SpecGraphCost{8, 0.0},
                                                   {8, 0.0}};
  EXPECT_FALSE(VerifyBudgetPlanner::calibration_plan(/*logical_requests=*/1,
                                                     /*physical_requests=*/1,
                                                     /*requested_graph_rows=*/8,
                                                     duplicated));
}

TEST(VerifyBudgetPlannerTest, LowConfidenceSelectsSmallerPhysicalGraph) {
  const std::vector<double> confidence(14, 0.01);
  const auto plan =
      VerifyBudgetPlanner::plan(2, confidence, kCosts, profile_options());
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 0U);
  EXPECT_EQ(plan->graph_rows, 2U);
}

TEST(VerifyBudgetPlannerTest, HighConfidenceKeepsAllDrafts) {
  const std::vector<double> confidence(14, 1.0);
  const auto plan =
      VerifyBudgetPlanner::plan(2, confidence, kCosts, profile_options());
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 14U);
  EXPECT_EQ(plan->graph_rows, 16U);
}

TEST(VerifyBudgetPlannerTest, AllocatesOneBatchBudgetFromConditionalPaths) {
  const std::vector<double> confidence{0.99,
                                       0.99,
                                       0.99,
                                       0.99,
                                       0.99,
                                       0.99,
                                       0.99,
                                       0.01,
                                       0.01,
                                       0.01,
                                       0.01,
                                       0.01,
                                       0.01,
                                       0.01};
  const std::array<SpecGraphCost, 2> costs = {SpecGraphCost{4, 0.1},
                                              {16, 100.0}};
  const auto plan =
      VerifyBudgetPlanner::plan(2, confidence, costs, profile_options());
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 2U);
  EXPECT_EQ(plan->graph_rows, 4U);
}

TEST(VerifyBudgetPlannerTest, FillsUnusedCapacityWithoutChangingGraphCost) {
  const std::vector<double> confidence(7, 0.01);
  const std::array<SpecGraphCost, 2> costs = {SpecGraphCost{1, 1.0}, {8, 1.0}};
  const auto plan = VerifyBudgetPlanner::plan(
      1, confidence, costs, profile_options(), VerifyBudgetPlan{5, 8});
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 7U);
  EXPECT_EQ(plan->graph_rows, 8U);
}

TEST(VerifyBudgetPlannerTest, RoundsFullWidthUpToWarmedCapacity) {
  const std::vector<double> confidence(21, 1.0);
  const std::array<SpecGraphCost, 2> costs = {SpecGraphCost{8, 3.0}, {32, 4.0}};
  const auto plan =
      VerifyBudgetPlanner::plan(3, confidence, costs, profile_options());
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 21U);
  EXPECT_EQ(plan->graph_rows, 32U);
}

TEST(VerifyBudgetPlannerTest, TiedScoresChooseSmallestGraphDeterministically) {
  const std::vector<double> confidence(14, 0.0);
  const std::array<SpecGraphCost, 3> costs = {
      SpecGraphCost{2, 1.0}, {8, 1.0}, {16, 1.0}};
  for (uint32_t repeat = 0; repeat < 5; ++repeat) {
    const auto plan =
        VerifyBudgetPlanner::plan(2, confidence, costs, profile_options());
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->draft_budget, 0U);
    EXPECT_EQ(plan->graph_rows, 2U);
  }
}

TEST(VerifyBudgetPlannerTest, FillsSpareRowsEvenWhenPathProbabilityIsZero) {
  const std::vector<double> confidence(14, 0.0);
  const std::array<SpecGraphCost, 1> costs = {SpecGraphCost{16, 1.0}};
  const auto plan = VerifyBudgetPlanner::plan(
      2, confidence, costs, profile_options(), VerifyBudgetPlan{5, 16});
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 14U);
  EXPECT_EQ(plan->graph_rows, 16U);
}

TEST(VerifyBudgetPlannerTest, IncludesFixedDraftAndCommitTimeInScore) {
  const std::vector<double> confidence(7, 0.3);
  const std::array<SpecGraphCost, 2> costs = {SpecGraphCost{2, 1.0}, {8, 2.0}};
  VerifyBudgetOptions options;
  options.fixed_draft_time_ms = 0.001;
  auto plan = VerifyBudgetPlanner::plan(1, confidence, costs, options);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 1U);
  options.commit_time_ms = 100.0;
  plan = VerifyBudgetPlanner::plan(1, confidence, costs, options);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 7U);
}

TEST(VerifyBudgetPlannerTest,
     HysteresisKeepsPreviousPlanUntilGainIsSufficient) {
  const std::vector<double> confidence(7, 0.5);
  const std::array<SpecGraphCost, 2> costs = {SpecGraphCost{2, 1.0}, {8, 2.0}};
  VerifyBudgetOptions options;
  options.fixed_draft_time_ms = 1.0;
  options.min_gain = 0.2;
  const VerifyBudgetPlan previous{7, 8};
  auto plan =
      VerifyBudgetPlanner::plan(1, confidence, costs, options, previous);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 7U);
  options.min_gain = 0.0;
  plan = VerifyBudgetPlanner::plan(1, confidence, costs, options, previous);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 1U);
}

TEST(VerifyBudgetPlannerTest, HysteresisCanRetainAnchorOnlyPreviousPlan) {
  const std::vector<double> confidence(7, 0.5);
  const std::array<SpecGraphCost, 3> costs = {
      SpecGraphCost{1, 1.0}, {2, 1.4}, {8, 10.0}};
  VerifyBudgetOptions options;
  options.cost_scope = SpecCostScope::FULL_STEP;
  options.min_gain = 0.1;
  const VerifyBudgetPlan previous{0, 1};
  auto plan =
      VerifyBudgetPlanner::plan(1, confidence, costs, options, previous);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->draft_budget, 0U);
  EXPECT_EQ(plan->graph_rows, 1U);
  options.min_gain = 0.0;
  plan = VerifyBudgetPlanner::plan(1, confidence, costs, options, previous);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->draft_budget, 1U);
  EXPECT_EQ(plan->graph_rows, 2U);
}

TEST(VerifyBudgetPlannerTest, PreservesBatchFirstRoundingAtScoreTie) {
  // Every prefix has the same tiny path probability. Adding each term to the
  // batch count rounds it away; summing terms before adding the count would
  // instead change the winning graph at an otherwise exact score tie.
  std::vector<double> confidence(7, 1.0);
  confidence.front() = std::numeric_limits<double>::epsilon() / 4.0;
  const std::array<SpecGraphCost, 2> costs = {SpecGraphCost{1, 1.0}, {8, 1.0}};
  VerifyBudgetOptions options;
  options.cost_scope = SpecCostScope::FULL_STEP;
  const auto plan = VerifyBudgetPlanner::plan(1, confidence, costs, options);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->draft_budget, 0U);
  EXPECT_EQ(plan->graph_rows, 1U);
}

TEST(VerifyBudgetPlannerTest, ForcesPeriodicFullWidthExploration) {
  const std::vector<double> confidence(14, 0.01);
  VerifyBudgetOptions options = profile_options();
  options.exploration_interval = 3;
  auto plan = VerifyBudgetPlanner::plan(
      2, confidence, kCosts, options, std::nullopt, 2);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 0U);
  plan = VerifyBudgetPlanner::plan(
      2, confidence, kCosts, options, VerifyBudgetPlan{0, 2}, 3);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 14U);
  EXPECT_EQ(plan->graph_rows, 16U);
}

TEST(VerifyBudgetPlannerTest, MissingFeedbackOrProfileKeepsFullWidth) {
  const auto without_feedback =
      VerifyBudgetPlanner::plan(2, {}, kCosts, profile_options());
  ASSERT_TRUE(without_feedback.has_value());
  EXPECT_EQ(without_feedback->draft_budget, 14U);
  const std::vector<double> confidence(14, 0.01);
  const std::array<SpecGraphCost, 3> unprofiled = {
      SpecGraphCost{2, 0.0}, {8, 0.0}, {16, 0.0}};
  const auto without_cost =
      VerifyBudgetPlanner::plan(2, confidence, unprofiled, profile_options());
  ASSERT_TRUE(without_cost.has_value());
  EXPECT_EQ(without_cost->draft_budget, 14U);
  const std::array<SpecGraphCost, 2> missing_full_cost = {SpecGraphCost{2, 1.0},
                                                          {16, 0.0}};
  const auto without_full_cost = VerifyBudgetPlanner::plan(
      2, confidence, missing_full_cost, profile_options());
  ASSERT_TRUE(without_full_cost.has_value());
  EXPECT_EQ(without_full_cost->draft_budget, 14U);
  const auto without_draft_cost =
      VerifyBudgetPlanner::plan(2, confidence, kCosts, {});
  ASSERT_TRUE(without_draft_cost.has_value());
  EXPECT_EQ(without_draft_cost->draft_budget, 14U);
}

TEST(VerifyBudgetPlannerTest, SkipsUnprofiledSmallerGraphs) {
  const std::vector<double> confidence(14, 0.01);
  const std::array<SpecGraphCost, 2> costs = {SpecGraphCost{2, 0.0}, {16, 8.0}};
  const auto plan =
      VerifyBudgetPlanner::plan(2, confidence, costs, profile_options());
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 14U);
}

TEST(VerifyBudgetPlannerTest, RejectsInsufficientWarmedCapacity) {
  const std::array<SpecGraphCost, 1> costs = {SpecGraphCost{8, 1.0}};
  EXPECT_FALSE(VerifyBudgetPlanner::plan(2, {}, costs, {}).has_value());
  EXPECT_FALSE(VerifyBudgetPlanner::plan(0, {}, kCosts, {}).has_value());
  EXPECT_FALSE(VerifyBudgetPlanner::plan(
                   std::numeric_limits<uint32_t>::max(), {}, kCosts, {})
                   .has_value());
}

TEST(VerifyBudgetPlannerTest, RejectsInvalidConfidence) {
  const std::vector<double> short_feedback(13, 0.5);
  EXPECT_FALSE(
      VerifyBudgetPlanner::plan(2, short_feedback, kCosts, profile_options())
          .has_value());
  for (double invalid : {-0.01,
                         1.01,
                         std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    std::vector<double> confidence(14, 0.5);
    confidence[7] = invalid;
    EXPECT_FALSE(
        VerifyBudgetPlanner::plan(2, confidence, kCosts, profile_options())
            .has_value());
  }
}

TEST(VerifyBudgetPlannerTest, RejectsInvalidBucketsAndOptions) {
  const std::array<SpecGraphCost, 2> duplicate = {SpecGraphCost{16, 1.0},
                                                  {16, 2.0}};
  const std::array<SpecGraphCost, 2> unsorted = {SpecGraphCost{16, 2.0},
                                                 {2, 1.0}};
  const std::array<SpecGraphCost, 1> negative = {SpecGraphCost{16, -1.0}};
  const std::array<SpecGraphCost, 1> nonfinite = {
      SpecGraphCost{16, std::numeric_limits<double>::infinity()}};
  EXPECT_FALSE(VerifyBudgetPlanner::plan(2, {}, {}, {}).has_value());
  EXPECT_FALSE(VerifyBudgetPlanner::plan(2, {}, duplicate, {}).has_value());
  EXPECT_FALSE(VerifyBudgetPlanner::plan(2, {}, unsorted, {}).has_value());
  EXPECT_FALSE(VerifyBudgetPlanner::plan(2, {}, negative, {}).has_value());
  EXPECT_FALSE(VerifyBudgetPlanner::plan(2, {}, nonfinite, {}).has_value());
  VerifyBudgetOptions options = profile_options();
  options.min_gain = -1.0;
  EXPECT_FALSE(VerifyBudgetPlanner::plan(2, {}, kCosts, options).has_value());
  options = profile_options();
  options.exploration_interval = 0;
  EXPECT_FALSE(VerifyBudgetPlanner::plan(2, {}, kCosts, options).has_value());
  options.fixed_draft_time_ms = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(VerifyBudgetPlanner::plan(2, {}, kCosts, options).has_value());
  options = profile_options();
  options.fixed_draft_time_ms = std::numeric_limits<double>::max();
  options.commit_time_ms = std::numeric_limits<double>::max();
  EXPECT_FALSE(VerifyBudgetPlanner::plan(2, {}, kCosts, options).has_value());
}

TEST(VerifyBudgetPlannerTest, RejectsPreviousPlanFromDifferentCapacity) {
  EXPECT_FALSE(VerifyBudgetPlanner::plan(
                   2, {}, kCosts, profile_options(), VerifyBudgetPlan{15, 16})
                   .has_value());
  EXPECT_FALSE(VerifyBudgetPlanner::plan(
                   2, {}, kCosts, profile_options(), VerifyBudgetPlan{7, 8})
                   .has_value());
  EXPECT_FALSE(VerifyBudgetPlanner::plan(
                   2, {}, kCosts, profile_options(), VerifyBudgetPlan{1, 3})
                   .has_value());
}

TEST(VerifyBudgetPlannerTest, FullStepCostsDoNotRequireSeparateDraftTiming) {
  VerifyBudgetOptions options;
  options.cost_scope = SpecCostScope::FULL_STEP;
  const std::vector<double> confidence(14, 0.01);
  const auto plan = VerifyBudgetPlanner::plan(2, confidence, kCosts, options);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->draft_budget, 0U);
  EXPECT_EQ(plan->graph_rows, 2U);
  options.fixed_draft_time_ms = 1.0;
  EXPECT_FALSE(VerifyBudgetPlanner::plan(2, confidence, kCosts, options));
  options.fixed_draft_time_ms = 0.0;
  options.commit_time_ms = 1.0;
  EXPECT_FALSE(VerifyBudgetPlanner::plan(2, confidence, kCosts, options));
}

TEST(VerifyBudgetPlannerTest,
     FullStepKeepsExplorationAndMissingFeedbackPolicy) {
  VerifyBudgetOptions options;
  options.cost_scope = SpecCostScope::FULL_STEP;
  options.exploration_interval = 2;
  auto plan = VerifyBudgetPlanner::plan(2, {}, kCosts, options);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->draft_budget, 14U);
  plan = VerifyBudgetPlanner::plan(
      2, std::vector<double>(14, 0.01), kCosts, options, std::nullopt, 2);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->draft_budget, 14U);
}

TEST(VerifyBudgetPlannerTest, DpBudgetsShareOneGraphWithUnequalAndEmptyShards) {
  const auto options = profile_options();
  const std::array<int32_t, 2> counts{1, 2};
  const std::vector<float> confidence(21, 0.01F);
  const auto plan = VerifyBudgetPlanner::parallel_plan(
      counts, confidence, {}, kCosts, options);
  ASSERT_TRUE(plan);
  ASSERT_EQ(plan->size(), 2U);
  EXPECT_EQ((*plan)[0].graph_rows, 2U);
  EXPECT_EQ((*plan)[1].graph_rows, 2U);
  EXPECT_EQ((*plan)[0].draft_budget, 1U);
  EXPECT_EQ((*plan)[1].draft_budget, 0U);

  const std::array<int32_t, 2> idle{0, 2};
  const auto empty = VerifyBudgetPlanner::parallel_plan(
      idle, std::vector<float>(14, 0.01F), {}, kCosts, options);
  ASSERT_TRUE(empty);
  EXPECT_EQ((*empty)[0].draft_budget, 0U);
  EXPECT_EQ((*empty)[0].graph_rows, (*empty)[1].graph_rows);
}

TEST(VerifyBudgetPlannerTest,
     DpMissingFeedbackAndExplorationKeepRequiredFullWidth) {
  const std::array<int32_t, 2> counts{1, 2};
  const auto options = profile_options();
  const auto cold =
      VerifyBudgetPlanner::parallel_plan(counts, {}, {}, kCosts, options);
  ASSERT_TRUE(cold);
  EXPECT_EQ((*cold)[0].graph_rows, 16U);
  EXPECT_EQ((*cold)[0].draft_budget, 7U);
  EXPECT_EQ((*cold)[1].draft_budget, 14U);
  std::vector<float> confidence(7, -1.0F);
  confidence.insert(confidence.end(), 14, 0.01F);
  const auto partial = VerifyBudgetPlanner::parallel_plan(
      counts, confidence, {}, kCosts, options);
  ASSERT_TRUE(partial);
  EXPECT_EQ((*partial)[0].graph_rows, 8U);
  EXPECT_EQ((*partial)[0].draft_budget, 7U);
  EXPECT_EQ((*partial)[1].draft_budget, 6U);
  const std::array<uint64_t, 3> rounds{0, 32, 0};
  const auto explored = VerifyBudgetPlanner::parallel_plan(
      counts, std::vector<float>(21, 0.01F), rounds, kCosts, options);
  ASSERT_TRUE(explored);
  EXPECT_EQ((*explored)[0].graph_rows, 16U);
  EXPECT_EQ((*explored)[1].draft_budget, 14U);
}

TEST(VerifyBudgetPlannerTest, DpKeepsKernelLimitsOutsideSharedPlanning) {
  const std::array<int32_t, 2> counts{1, 1};
  const std::array<SpecGraphCost, 2> costs{{{1, 1.0}, {128, 128.0}}};
  for (const uint32_t width : {1U, 3U, 5U, 7U, 64U}) {
    auto options = profile_options();
    options.draft_width = width;
    const auto plan =
        VerifyBudgetPlanner::parallel_plan(counts, {}, {}, costs, options);
    ASSERT_TRUE(plan);
    EXPECT_EQ((*plan)[0].draft_budget, width);
    EXPECT_EQ((*plan)[1].draft_budget, width);
    options.draft_width = std::numeric_limits<uint32_t>::max() - 1;
    const std::array<int32_t, 2> overflowing{2, 1};
    EXPECT_FALSE(VerifyBudgetPlanner::parallel_plan(
        overflowing, {}, {}, costs, options));
  }
}

TEST(VerifyBudgetPlannerTest, DpSupportsBothWidthsAndRejectsMalformedMetadata) {
  const std::array<int32_t, 2> counts{1, 1};
  for (const uint32_t width : {5U, 7U}) {
    auto options = profile_options();
    options.draft_width = width;
    const auto plan = VerifyBudgetPlanner::parallel_plan(
        counts, std::vector<float>(2 * width, 0.01F), {}, kCosts, options);
    ASSERT_TRUE(plan);
    EXPECT_EQ((*plan)[0].graph_rows, (*plan)[1].graph_rows);
    EXPECT_LT((*plan)[0].draft_budget, width);
    EXPECT_FALSE(VerifyBudgetPlanner::parallel_plan(
        counts, std::vector<float>(width, 0.01F), {}, kCosts, options));
    EXPECT_FALSE(VerifyBudgetPlanner::parallel_plan(
        counts, std::vector<float>(2 * width, -0.5F), {}, kCosts, options));
    const std::array<int32_t, 2> negative{-1, 1};
    EXPECT_FALSE(
        VerifyBudgetPlanner::parallel_plan(negative, {}, {}, kCosts, options));
    const std::array<int32_t, 2> idle{0, 0};
    EXPECT_FALSE(
        VerifyBudgetPlanner::parallel_plan(idle, {}, {}, kCosts, options));
  }
}

}  // namespace
}  // namespace xllm
