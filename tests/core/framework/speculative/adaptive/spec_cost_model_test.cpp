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

#include "core/framework/speculative/adaptive/spec_cost_model.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <nlohmann/json.hpp>
#include <vector>

#include "core/framework/speculative/adaptive/verify_budget_planner.h"

namespace xllm {
namespace {

nlohmann::json sps_table() {
  return {{"sample_batch_tokens", {2, 4, 8}},
          {"sample_steps_per_sec", {100.0, 50.0, 25.0}},
          {"max_batch_tokens", 16}};
}

nlohmann::json additive_table() {
  return {{"bias_seconds", 0.010},
          {"bs_probes", {1, 3}},
          {"alpha_seconds", {0.001, 0.003}},
          {"m_probes", {0, 8}},
          {"theta_seconds", {0.0, 0.008}}};
}

TEST(SpecCostModelTest, SpsUsesFloorProbeAndMilliseconds) {
  const auto model = SpecCostModel::from_json(sps_table().dump());
  ASSERT_TRUE(model);
  EXPECT_EQ(model->kind(), SpecCostModel::Kind::SPS);
  EXPECT_DOUBLE_EQ(*model->step_time_ms(1, 1), 10.0);
  EXPECT_DOUBLE_EQ(*model->step_time_ms(1, 2), 10.0);
  EXPECT_DOUBLE_EQ(*model->step_time_ms(1, 3), 10.0);
  EXPECT_DOUBLE_EQ(*model->step_time_ms(1, 4), 20.0);
  EXPECT_DOUBLE_EQ(*model->step_time_ms(1, 8), 40.0);
  EXPECT_DOUBLE_EQ(*model->step_time_ms(1, 16), 40.0);
  EXPECT_FALSE(model->step_time_ms(1, 17));
  EXPECT_FALSE(model->step_time_ms(0, 1));
  EXPECT_FALSE(model->step_time_ms(8, 4));
}

TEST(SpecCostModelTest, SpsSupportsSingleProbeAndContextBound) {
  auto table = sps_table();
  table["sample_batch_tokens"] = {1};
  table["sample_steps_per_sec"] = {25.0};
  table["max_kv_length"] = 4096;
  const auto model = SpecCostModel::from_json(table.dump());
  ASSERT_TRUE(model);
  EXPECT_EQ(model->max_kv_length(), 4096U);
  EXPECT_DOUBLE_EQ(*model->step_time_ms(2, 8), 40.0);
}

TEST(SpecCostModelTest, AdditiveInterpolatesBothAxesAndClampsEdges) {
  const auto model = SpecCostModel::from_json(additive_table().dump());
  ASSERT_TRUE(model);
  EXPECT_EQ(model->kind(), SpecCostModel::Kind::ADDITIVE);
  EXPECT_NEAR(*model->step_time_ms(2, 4), 16.0, 1e-12);
  EXPECT_NEAR(*model->step_time_ms(1, 1), 12.0, 1e-12);
  EXPECT_NEAR(*model->step_time_ms(3, 8), 21.0, 1e-12);
  EXPECT_NEAR(*model->step_time_ms(16, 128), 21.0, 1e-12);
  EXPECT_EQ(model->max_kv_length(), std::numeric_limits<uint32_t>::max());
}

TEST(SpecCostModelTest, AdditiveAcceptsNegativeOffsetsWithPositiveTotals) {
  auto table = additive_table();
  table["alpha_seconds"] = {-0.002, -0.001};
  const auto model = SpecCostModel::from_json(table.dump());
  ASSERT_TRUE(model);
  EXPECT_NEAR(*model->step_time_ms(1, 8), 16.0, 1e-12);
  table["alpha_seconds"] = {-0.020, -0.001};
  EXPECT_FALSE(SpecCostModel::from_json(table.dump()));
}

TEST(SpecCostModelTest, AdditiveSupportsSinglePointAxes) {
  auto table = additive_table();
  table["bs_probes"] = {1};
  table["alpha_seconds"] = {0.002};
  table["m_probes"] = {0};
  table["theta_seconds"] = {0.003};
  const auto model = SpecCostModel::from_json(table.dump());
  ASSERT_TRUE(model);
  EXPECT_NEAR(*model->step_time_ms(8, 64), 15.0, 1e-12);
}

TEST(SpecCostModelTest, RejectsMalformedAmbiguousOrIncompleteJson) {
  for (const auto* text :
       {"{", "[]", "null", "{}", "{\"bias_seconds\":1e999}"}) {
    EXPECT_FALSE(SpecCostModel::from_json(text)) << text;
  }
  auto table = sps_table();
  table["bias_seconds"] = 1.0;
  EXPECT_FALSE(SpecCostModel::from_json(table.dump()));
  table = additive_table();
  table.erase("theta_seconds");
  EXPECT_FALSE(SpecCostModel::from_json(table.dump()));
}

TEST(SpecCostModelTest, RejectsInvalidAxesAndValues) {
  for (const auto& invalid : nlohmann::json{nlohmann::json::array(),
                                            {2, 2, 8},
                                            {8, 4, 2},
                                            {0, 4, 8},
                                            {-1, 4, 8},
                                            {2.0, 4, 8},
                                            {true, 4, 8},
                                            {4294967296ULL, 4, 8}}) {
    auto table = sps_table();
    table["sample_batch_tokens"] = invalid;
    EXPECT_FALSE(SpecCostModel::from_json(table.dump())) << table;
  }
  for (const auto& invalid : nlohmann::json{nlohmann::json::array(),
                                            {100, 50},
                                            {0, 50, 25},
                                            {-1, 50, 25},
                                            {"100", 50, 25},
                                            {true, 50, 25},
                                            {1e-310, 50, 25}}) {
    auto table = sps_table();
    table["sample_steps_per_sec"] = invalid;
    EXPECT_FALSE(SpecCostModel::from_json(table.dump())) << table;
  }
}

TEST(SpecCostModelTest, RejectsInvalidBoundsAndOverflow) {
  for (const auto& invalid :
       nlohmann::json{0, -1, 7, 16.0, true, 4294967296ULL}) {
    auto table = sps_table();
    table["max_batch_tokens"] = invalid;
    EXPECT_FALSE(SpecCostModel::from_json(table.dump())) << table;
  }
  for (const auto& invalid : nlohmann::json{0, -1, 4096.0, true}) {
    auto table = additive_table();
    table["max_kv_length"] = invalid;
    EXPECT_FALSE(SpecCostModel::from_json(table.dump())) << table;
  }
  auto table = additive_table();
  table["bias_seconds"] = std::numeric_limits<double>::max();
  EXPECT_FALSE(SpecCostModel::from_json(table.dump()));
}

TEST(SpecCostModelTest, BothModelsDrivePhysicalGraphPlanner) {
  for (const auto& table : {sps_table(), additive_table()}) {
    const auto model = SpecCostModel::from_json(table.dump());
    ASSERT_TRUE(model);
    std::vector<SpecGraphCost> costs;
    costs.reserve(3);
    for (const uint32_t graph : {2U, 4U, 8U}) {
      costs.emplace_back(SpecGraphCost{graph, *model->step_time_ms(1, graph)});
    }
    VerifyBudgetOptions options;
    options.cost_scope = SpecCostScope::FULL_STEP;
    const auto plan = VerifyBudgetPlanner::plan(
        1, std::vector<double>(7, 0.01), costs, options);
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->graph_rows, 2U);
    EXPECT_EQ(plan->draft_budget, 1U);
  }
}

nlohmann::json graph_table() {
  return {{"graph_costs",
           {{{"draft_requests", 1},
             {"target_graph_rows", 2},
             {"step_time_ms", 2.0}},
            {{"draft_requests", 1},
             {"target_graph_rows", 4},
             {"step_time_ms", 12.0}},
            {{"draft_requests", 1},
             {"target_graph_rows", 8},
             {"step_time_ms", 4.0}},
            {{"draft_requests", 2},
             {"target_graph_rows", 2},
             {"step_time_ms", 12.0}},
            {{"draft_requests", 2},
             {"target_graph_rows", 4},
             {"step_time_ms", 2.0}},
            {{"draft_requests", 2},
             {"target_graph_rows", 16},
             {"step_time_ms", 20.0}}}},
          {"max_kv_length", 4096}};
}

TEST(SpecCostModelTest, GraphPreservesBatchInteractionAndNonmonotonicCosts) {
  const auto model = SpecCostModel::from_json(graph_table().dump());
  ASSERT_TRUE(model);
  EXPECT_EQ(model->kind(), SpecCostModel::Kind::GRAPH);
  EXPECT_EQ(SpecCostModel::kind_name(model->kind()), "graph");
  EXPECT_EQ(model->max_kv_length(), 4096U);
  EXPECT_DOUBLE_EQ(*model->step_time_ms(1, 4), 12.0);
  EXPECT_DOUBLE_EQ(*model->step_time_ms(2, 4), 2.0);
  EXPECT_DOUBLE_EQ(*model->step_time_ms(1, 8), 4.0);
  EXPECT_FALSE(model->step_time_ms(1, 3));
  EXPECT_FALSE(model->step_time_ms(3, 4));
  EXPECT_FALSE(model->step_time_ms(2, 8));
  EXPECT_FALSE(model->step_time_ms(0, 4));
  EXPECT_FALSE(model->step_time_ms(2, 1));
}

TEST(SpecCostModelTest, GraphKeyRetainsBothFullWidthUnsignedCoordinates) {
  constexpr uint32_t kMaximum = std::numeric_limits<uint32_t>::max();
  const nlohmann::json table{{"graph_costs",
                              {{{"draft_requests", 1},
                                {"target_graph_rows", kMaximum},
                                {"step_time_ms", 2.0}},
                               {{"draft_requests", kMaximum},
                                {"target_graph_rows", kMaximum},
                                {"step_time_ms", 3.0}}}},
                             {"max_kv_length", 4096}};
  const auto model = SpecCostModel::from_json(table.dump());
  ASSERT_TRUE(model);
  EXPECT_DOUBLE_EQ(*model->step_time_ms(1, kMaximum), 2.0);
  EXPECT_DOUBLE_EQ(*model->step_time_ms(kMaximum, kMaximum), 3.0);
  EXPECT_FALSE(model->step_time_ms(kMaximum - 1, kMaximum));
}

TEST(SpecCostModelTest, GraphRejectsMalformedAmbiguousAndDuplicateCells) {
  for (const auto& invalid : std::vector<nlohmann::json>{
           nlohmann::json::array(),
           nullptr,
           "invalid",
           nlohmann::json::object(),
           {nlohmann::json::object()},
           {{{"draft_requests", 1}, {"target_graph_rows", 4}}}}) {
    auto table = graph_table();
    table["graph_costs"] = invalid;
    EXPECT_FALSE(SpecCostModel::from_json(table.dump())) << table;
  }
  for (const auto& patch : std::vector<nlohmann::json>{
           {{"draft_requests", 0}},
           {{"draft_requests", -1}},
           {{"draft_requests", 1.0}},
           {{"draft_requests", true}},
           {{"draft_requests", 3}},
           {{"draft_requests", 4294967296ULL}},
           {{"target_graph_rows", 0}},
           {{"target_graph_rows", -1}},
           {{"target_graph_rows", 2.0}},
           {{"target_graph_rows", true}},
           {{"target_graph_rows", 4294967296ULL}},
           {{"step_time_ms", 0}},
           {{"step_time_ms", -1}},
           {{"step_time_ms", "2"}},
           {{"step_time_ms", true}},
           {{"step_time_ms", 1e-310}},
           {{"step_time_ms", std::numeric_limits<double>::infinity()}}}) {
    auto table = graph_table();
    table["graph_costs"][0].update(patch);
    EXPECT_FALSE(SpecCostModel::from_json(table.dump())) << table;
  }
  auto table = graph_table();
  table.erase("max_kv_length");
  EXPECT_FALSE(SpecCostModel::from_json(table.dump()));
  table = graph_table();
  table["graph_costs"].push_back(table["graph_costs"][0]);
  EXPECT_FALSE(SpecCostModel::from_json(table.dump()));
  for (const auto& legacy : {sps_table(), additive_table()}) {
    table = graph_table();
    table.update(legacy);
    EXPECT_FALSE(SpecCostModel::from_json(table.dump()));
  }
}

TEST(SpecCostModelTest, GraphOnlyOffersMeasuredCapturedVerificationBuckets) {
  const auto model = SpecCostModel::from_json(graph_table().dump());
  ASSERT_TRUE(model);
  const std::array<uint32_t, 5> captured{1, 2, 4, 8, 16};
  auto costs = model->verification_costs(2, 7, captured);
  ASSERT_TRUE(costs);
  ASSERT_EQ(costs->size(), 3U);
  EXPECT_EQ((*costs)[0].graph_rows, 2U);
  EXPECT_EQ((*costs)[1].graph_rows, 4U);
  EXPECT_EQ((*costs)[2].graph_rows, 16U);
  EXPECT_DOUBLE_EQ((*costs)[1].target_time_ms, 2.0);
  costs = model->verification_costs(1, 7, std::array<uint32_t, 3>{2, 8, 16});
  ASSERT_TRUE(costs);
  ASSERT_EQ(costs->size(), 2U);
  EXPECT_EQ(costs->back().graph_rows, 8U);
  // The captured full-width bucket must itself be measured. A larger measured
  // bucket cannot stand in for it when the physical replay chooses the first.
  EXPECT_FALSE(
      model->verification_costs(1, 5, std::array<uint32_t, 4>{2, 4, 6, 8}));
  EXPECT_FALSE(
      model->verification_costs(2, 7, std::array<uint32_t, 3>{2, 4, 8}));
  EXPECT_FALSE(model->verification_costs(3, 7, captured));
  EXPECT_FALSE(model->verification_costs(0, 7, captured));
  EXPECT_FALSE(model->verification_costs(1, 0, captured));
  EXPECT_FALSE(model->verification_costs(1, 7, {}));
  EXPECT_FALSE(
      model->verification_costs(1, 7, std::array<uint32_t, 3>{8, 4, 2}));
  EXPECT_FALSE(
      model->verification_costs(1, 7, std::array<uint32_t, 3>{2, 8, 8}));
  EXPECT_FALSE(
      model->verification_costs(1, 7, std::array<uint32_t, 3>{0, 2, 8}));
  EXPECT_FALSE(model->verification_costs(
      1, std::numeric_limits<uint32_t>::max(), captured));
}

TEST(SpecCostModelTest, SparseGraphCoverageKeepsFullWidthForBothDraftWidths) {
  auto table = graph_table();
  table["graph_costs"].emplace_back(nlohmann::json{
      {"draft_requests", 2}, {"target_graph_rows", 12}, {"step_time_ms", 7.0}});
  const auto model = SpecCostModel::from_json(table.dump());
  ASSERT_TRUE(model);
  const std::array<uint32_t, 6> captured{1, 2, 4, 8, 12, 16};
  for (const uint32_t width : {5U, 7U}) {
    const auto costs = model->verification_costs(2, width, captured);
    ASSERT_TRUE(costs);
    EXPECT_EQ(costs->back().graph_rows, 2 * (width + 1));
    EXPECT_TRUE(
        std::none_of(costs->begin(), costs->end(), [](const auto& cost) {
          return cost.graph_rows == 8;
        }));
  }
  table["graph_costs"].erase(table["graph_costs"].end() - 1);
  const auto missing = SpecCostModel::from_json(table.dump());
  ASSERT_TRUE(missing);
  EXPECT_FALSE(missing->verification_costs(2, 5, captured));
  EXPECT_TRUE(missing->verification_costs(2, 7, captured));
}

TEST(SpecCostModelTest, LegacyVerificationProfilesKeepExistingCosts) {
  const std::array<uint32_t, 5> captured{1, 2, 4, 8, 16};
  for (const auto& table : {sps_table(), additive_table()}) {
    const auto model = SpecCostModel::from_json(table.dump());
    ASSERT_TRUE(model);
    const auto costs = model->verification_costs(1, 7, captured);
    ASSERT_TRUE(costs);
    ASSERT_EQ(costs->size(), 4U);
    for (size_t index = 0; index < costs->size(); ++index) {
      EXPECT_EQ((*costs)[index].graph_rows, captured[index]);
      EXPECT_DOUBLE_EQ((*costs)[index].target_time_ms,
                       *model->step_time_ms(1, captured[index]));
    }
  }
}

TEST(SpecCostModelTest,
     SparseGraphProfileKeepsFullWidthForUnmeasuredLogicalBatch) {
  const auto model = SpecCostModel::from_json(graph_table().dump());
  ASSERT_TRUE(model);
  const std::array<uint32_t, 5> captured{1, 2, 4, 8, 16};
  const auto measured = model->verification_costs(2, 7, captured);
  ASSERT_TRUE(measured);
  // The pipeline may admit one logical request into physical draft bucket 2.
  // Populate only observed costs in its prewarmed candidate list, as serving
  // admission does. Its own full-width G=8 lacks a measurement, so it must not
  // prune using the cheap G=4 cost from this physical Bcap.
  std::vector<SpecGraphCost> candidates;
  candidates.reserve(captured.size());
  for (const uint32_t rows : captured) {
    const auto found = std::find_if(
        measured->begin(), measured->end(), [&](const SpecGraphCost& cost) {
          return cost.graph_rows == rows;
        });
    candidates.emplace_back(SpecGraphCost{
        rows, found == measured->end() ? 0.0 : found->target_time_ms});
  }
  VerifyBudgetOptions options;
  options.cost_scope = SpecCostScope::FULL_STEP;
  const auto plan = VerifyBudgetPlanner::plan(
      1, std::vector<double>(7, 0.01), candidates, options);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->graph_rows, 8U);
  EXPECT_EQ(plan->draft_budget, 7U);
}

}  // namespace
}  // namespace xllm
