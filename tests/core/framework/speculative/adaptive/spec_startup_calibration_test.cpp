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

#include "core/framework/speculative/adaptive/spec_startup_calibration.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>

#include "core/framework/speculative/adaptive/spec_cost_profile_store.h"
#include "core/framework/speculative/adaptive/verify_budget_planner.h"

namespace xllm {
namespace {

nlohmann::json configuration() {
  return {{"model", "sps"},
          {"request_buckets", {1, 4}},
          {"contexts",
           {{{"prefill_tokens", 1868},
             {"min_kv_length", 1800},
             {"max_kv_length", 4096}}}}};
}

SpecSteadyWindow window(uint32_t requests, uint32_t rows, double cost) {
  SpecSteadyWindow result;
  result.first_generation = 1;
  result.last_generation = 33;
  result.logical_requests = requests;
  result.draft_requests = requests;
  result.target_graph_rows = rows;
  result.min_kv_length = 2000;
  result.max_kv_length = 2100;
  result.steps = 32;
  result.draft_tokens =
      static_cast<uint64_t>(std::min(requests * 7, rows - requests)) * 32;
  result.elapsed_ms = cost * 32;
  return result;
}

nlohmann::json sps_table(uint32_t maximum = 4096) {
  return {{"sample_batch_tokens", {1, 2, 4, 8, 16, 32}},
          {"sample_steps_per_sec", {100, 90, 80, 70, 60, 50}},
          {"max_batch_tokens", 32},
          {"max_kv_length", maximum}};
}

nlohmann::json bundle() {
  return {{"schema_version", 1},
          {"measurement", "steady_full_step"},
          {"compatibility",
           {{"model", "unit-test"}, {"quantization", "static-w8a8"}}},
          {"profiles",
           {{{"min_kv_length", 1},
             {"max_kv_length", 4096},
             {"request_buckets", {1, 4}},
             {"table", sps_table()}}}}};
}

TEST(SpecStartupCalibrationTest, DefaultsUseAdditiveCostsAndBoundedContext) {
  const auto config = default_spec_startup_calibration(202752);
  ASSERT_TRUE(config);
  EXPECT_EQ(config->model, SpecCostModel::Kind::ADDITIVE);
  EXPECT_TRUE(config->request_buckets.empty());
  EXPECT_TRUE(config->target_graph_rows.empty());
  ASSERT_EQ(config->contexts.size(), 1U);
  EXPECT_EQ(config->contexts.front().prefill_tokens, 2048U);
  EXPECT_EQ(config->contexts.front().min_kv_length, 1U);
  EXPECT_EQ(config->contexts.front().max_kv_length, 4096U);
  EXPECT_EQ(config->warmup_steps, 8U);
  EXPECT_EQ(config->window_steps, 32U);
  EXPECT_EQ(config->windows_per_point, 3U);
  EXPECT_EQ(config->token_seed, 1U);
}

TEST(SpecStartupCalibrationTest, DraftFiveUsesItsOwnCapacityAndGeometry) {
  const auto config = parse_spec_startup_calibration(configuration().dump(), 5);
  ASSERT_TRUE(config);
  EXPECT_EQ(config->draft_width, 5U);
  const std::string serialized = serialize_spec_startup_calibration(*config);
  EXPECT_TRUE(parse_spec_startup_calibration(serialized, 5));
  EXPECT_FALSE(parse_spec_startup_calibration(serialized, 7));
  EXPECT_FALSE(parse_spec_startup_calibration(serialized, 0));
  const std::array<uint32_t, 2> requests{1, 4};
  const std::array<uint32_t, 8> graphs{1, 2, 4, 6, 8, 16, 24, 32};
  const auto points = plan_spec_calibration(*config, requests, graphs);
  ASSERT_TRUE(points);
  EXPECT_EQ(points->back().target_graph_rows, 24U);
  EXPECT_EQ(points->back().requests, 4U);
  EXPECT_TRUE(default_spec_startup_calibration(864, 5));
  EXPECT_FALSE(default_spec_startup_calibration(864, 7));
  EXPECT_FALSE(default_spec_startup_calibration(4096, 6));
  EXPECT_EQ(spec_calibration_warmup_length(8192, 32768, 128, 2048, 5), 4093U);
}

TEST(SpecStartupCalibrationTest, DraftFiveFitsMatchingGeneratedTokenCounts) {
  auto sample = window(1, 8, 10.0);
  sample.draft_tokens = sample.steps * 5;
  const std::array<SpecSteadyWindow, 1> samples{sample};
  const auto fitted =
      fit_spec_cost_table(samples, SpecCostModel::Kind::SPS, 1800, 4096, 5);
  ASSERT_TRUE(fitted);
  const auto data = nlohmann::json::parse(*fitted);
  EXPECT_EQ(data["draft_width"], 5);
  EXPECT_EQ(data["metadata"]["cells"][0]["generated_draft_tokens"], 160);
  EXPECT_FALSE(
      fit_spec_cost_table(samples, SpecCostModel::Kind::SPS, 1800, 4096, 7));
}

TEST(SpecStartupCalibrationTest, UnversionedCostsCannotCrossDraftWidths) {
  auto table = sps_table();
  const std::string five = nlohmann::json{{"num_speculative_tokens", 5}}.dump();
  const std::string seven =
      nlohmann::json{{"num_speculative_tokens", 7}}.dump();
  EXPECT_FALSE(SpecCostTableBundle::from_json(table.dump(), five));
  EXPECT_TRUE(SpecCostTableBundle::from_json(table.dump(), seven));
  table["draft_width"] = 5;
  EXPECT_TRUE(SpecCostTableBundle::from_json(table.dump(), five));
  EXPECT_FALSE(SpecCostTableBundle::from_json(table.dump(), seven));
  table["draft_width"] = 6;
  EXPECT_FALSE(SpecCostTableBundle::from_json(table.dump()));
}

TEST(SpecStartupCalibrationTest, DefaultsRespectShortModelContexts) {
  auto config = default_spec_startup_calibration(1024);
  ASSERT_TRUE(config);
  EXPECT_EQ(config->contexts.front().max_kv_length, 1024U);
  EXPECT_LT(config->contexts.front().prefill_tokens, 2048U);
  config->request_buckets = {1};
  EXPECT_TRUE(parse_spec_startup_calibration(
      serialize_spec_startup_calibration(*config)));
  EXPECT_FALSE(default_spec_startup_calibration(864));
}

TEST(SpecStartupCalibrationTest,
     EffectiveJsonIncludesOmittedDefaultsAndResolvedBuckets) {
  auto config = parse_spec_startup_calibration(configuration().dump());
  ASSERT_TRUE(config);
  const auto data =
      nlohmann::json::parse(serialize_spec_startup_calibration(*config));
  EXPECT_EQ(data.at("warmup_steps"), 8);
  EXPECT_EQ(data.at("window_steps"), 32);
  EXPECT_EQ(data.at("windows_per_point"), 3);
  EXPECT_EQ(data.at("token_seed"), 1);
  EXPECT_EQ(data.at("request_buckets"), nlohmann::json({1, 4}));
  EXPECT_EQ(data.at("contexts")[0].at("prefill_tokens"), 1868);
  EXPECT_TRUE(parse_spec_startup_calibration(data.dump()));
  config->target_graph_rows = {1, 4, 8, 16, 32};
  EXPECT_EQ(nlohmann::json::parse(serialize_spec_startup_calibration(*config))
                .at("target_graph_rows"),
            nlohmann::json({1, 4, 8, 16, 32}));
}

TEST(SpecStartupCalibrationTest, ParsesConfigurationAndZeroSettling) {
  auto data = configuration();
  data["warmup_steps"] = 0;
  data["token_seed"] = 0;
  const auto config = parse_spec_startup_calibration(data.dump());
  ASSERT_TRUE(config);
  EXPECT_EQ(config->model, SpecCostModel::Kind::SPS);
  EXPECT_EQ(config->warmup_steps, 0U);
  EXPECT_EQ(config->window_steps, 32U);
  EXPECT_EQ(config->windows_per_point, 3U);
  EXPECT_EQ(config->contexts.front().prefill_tokens, 1868U);
}

TEST(SpecStartupCalibrationTest, RejectsInvalidConfigurationAndInsufficientKv) {
  for (const auto& patch :
       std::vector<nlohmann::json>{{{"model", "events"}},
                                   {{"request_buckets", {4, 1}}},
                                   {{"request_buckets", {1, 1}}},
                                   {{"request_buckets", {1, 129}}},
                                   {{"window_steps", 0}},
                                   {{"window_steps", true}},
                                   {{"warmup_steps", -1}},
                                   {{"windows_per_point", 4294967296ULL}},
                                   {{"contexts", nlohmann::json::array()}}}) {
    auto data = configuration();
    data.update(patch);
    EXPECT_FALSE(parse_spec_startup_calibration(data.dump())) << data;
  }
  auto data = configuration();
  data["contexts"][0]["max_kv_length"] = 1900;
  EXPECT_FALSE(parse_spec_startup_calibration(data.dump()));
  data = configuration();
  data["contexts"].emplace_back(data["contexts"][0]);
  EXPECT_FALSE(parse_spec_startup_calibration(data.dump()));
}

TEST(SpecStartupCalibrationTest, PlansActualCapturedFullWidthRatherThanEightB) {
  auto config = *parse_spec_startup_calibration(configuration().dump());
  config.request_buckets = {1};
  const std::array<uint32_t, 2> requests{1, 4};
  const std::array<uint32_t, 6> graphs{1, 2, 4, 10, 16, 32};
  const auto points = plan_spec_calibration(config, requests, graphs);
  ASSERT_TRUE(points);
  ASSERT_EQ(points->size(), 4U);
  EXPECT_EQ(points->back().target_graph_rows, 10U);
  EXPECT_EQ(points->back().requests, 1U);
}

TEST(SpecStartupCalibrationTest, PlansKvCoverageBeforeFirstCapture) {
  EXPECT_EQ(spec_calibration_warmup_length(4096, 32768, 128, 2048), 2048U);
  EXPECT_EQ(spec_calibration_warmup_length(4097, 32768, 128, 2048), 2048U);
  EXPECT_EQ(spec_calibration_warmup_length(4098, 32768, 128, 2048), 4091U);
  EXPECT_EQ(spec_calibration_warmup_length(8192, 32768, 128, 2048), 4091U);
  EXPECT_EQ(spec_calibration_warmup_length(32768, 32768, 128, 2048), 16379U);
  EXPECT_EQ(spec_calibration_warmup_length(8192, 32768, 256, 16), 16U);
  EXPECT_EQ(spec_calibration_warmup_length(8, 32, 1, 32), 16U);
  EXPECT_FALSE(spec_calibration_warmup_length(0, 32768, 128, 2048));
  EXPECT_FALSE(spec_calibration_warmup_length(32769, 32768, 128, 2048));
  EXPECT_FALSE(spec_calibration_warmup_length(8200, 8200, 128, 2048));
  EXPECT_FALSE(spec_calibration_warmup_length(8, 16, 1, 16));
  EXPECT_FALSE(spec_calibration_warmup_length(4096, 32768, 0, 2048));
  EXPECT_FALSE(spec_calibration_warmup_length(4096, 32768, 128, 0));
}

TEST(SpecStartupCalibrationTest, PlansOnlyCapturedWholeRoundBindings) {
  auto config = *parse_spec_startup_calibration(configuration().dump());
  config.request_buckets = {1, 8};
  config.model = SpecCostModel::Kind::ADDITIVE;
  const std::array<uint32_t, 2> requests{1, 8};
  const std::array<uint32_t, 8> graphs{1, 2, 4, 8, 16, 32, 48, 64};
  const std::vector<SpecCalibrationGraphCoverage> coverage{
      {1, {1, 2, 4, 8}}, {8, {8, 16, 32, 64}}};
  const auto split = plan_spec_calibration(config, requests, graphs);
  const auto unified =
      plan_spec_calibration(config, requests, graphs, coverage);
  ASSERT_TRUE(split);
  ASSERT_TRUE(unified);
  EXPECT_EQ(split->size(), 9U);
  EXPECT_EQ(unified->size(), 8U);
  EXPECT_TRUE(
      std::none_of(unified->begin(), unified->end(), [](const auto& point) {
        return point.target_graph_rows == 48;
      }));
  EXPECT_EQ(unified->back().target_graph_rows, 64U);
}

TEST(SpecStartupCalibrationTest, RejectsExplicitUncapturedWholeRoundPoint) {
  auto config = *parse_spec_startup_calibration(configuration().dump());
  config.request_buckets = {8};
  config.target_graph_rows = {8, 16, 32, 48, 64};
  const std::array<uint32_t, 1> requests{8};
  const std::array<uint32_t, 5> graphs{8, 16, 32, 48, 64};
  const std::vector<SpecCalibrationGraphCoverage> coverage{
      {8, {8, 16, 32, 64}}};
  EXPECT_FALSE(plan_spec_calibration(config, requests, graphs, coverage));
  config.target_graph_rows = {8, 32, 64};
  EXPECT_TRUE(plan_spec_calibration(config, requests, graphs, coverage));
}

TEST(SpecStartupCalibrationTest, RejectsIncompleteWholeRoundCoverage) {
  auto config = *parse_spec_startup_calibration(configuration().dump());
  config.request_buckets = {1, 8};
  const std::array<uint32_t, 2> requests{1, 8};
  const std::array<uint32_t, 8> graphs{1, 2, 4, 8, 16, 32, 48, 64};
  const std::vector<std::vector<SpecCalibrationGraphCoverage>> invalid{
      {{1, {1, 2, 4, 8}}},
      {{1, {1, 2, 4, 8}}, {8, {8, 16, 32, 48}}},
      {{1, {1, 2, 4, 8}}, {8, {8, 16, 16, 64}}},
      {{1, {1, 2, 4, 8}}, {8, {8, 16, 40, 64}}},
      {{8, {8, 16, 32, 64}}, {1, {1, 2, 4, 8}}},
      {{1, {1, 2, 4, 8}}, {2, {2, 4, 8, 16}}, {8, {8, 16, 32, 64}}}};
  for (const auto& coverage : invalid) {
    EXPECT_FALSE(plan_spec_calibration(config, requests, graphs, coverage));
  }
}

TEST(SpecStartupCalibrationTest, ChecksConnectivityAfterWholeRoundFiltering) {
  auto config = *parse_spec_startup_calibration(configuration().dump());
  config.request_buckets = {1, 8};
  config.model = SpecCostModel::Kind::ADDITIVE;
  const std::array<uint32_t, 2> requests{1, 8};
  const std::array<uint32_t, 7> graphs{1, 2, 4, 8, 16, 32, 64};
  const std::vector<SpecCalibrationGraphCoverage> coverage{{1, {1, 2, 4, 8}},
                                                           {8, {16, 32, 64}}};
  EXPECT_FALSE(plan_spec_calibration(config, requests, graphs, coverage));
  config.model = SpecCostModel::Kind::SPS;
  EXPECT_TRUE(plan_spec_calibration(config, requests, graphs, coverage));
}

TEST(SpecStartupCalibrationTest, RejectsUncapturedAndMissingFullWidthPoints) {
  auto config = *parse_spec_startup_calibration(configuration().dump());
  const std::array<uint32_t, 2> requests{1, 4};
  const std::array<uint32_t, 6> graphs{1, 2, 4, 8, 16, 32};
  config.target_graph_rows = {1, 3, 8, 32};
  EXPECT_FALSE(plan_spec_calibration(config, requests, graphs));
  config.target_graph_rows = {1, 2, 4, 8};
  EXPECT_FALSE(plan_spec_calibration(config, requests, graphs));
  config.target_graph_rows.clear();
  config.request_buckets = {2};
  EXPECT_FALSE(plan_spec_calibration(config, requests, graphs));
}

TEST(SpecStartupCalibrationTest, SpsCellMediansMatchReferenceValues) {
  const std::vector<SpecSteadyWindow> windows{window(1, 4, 10),
                                              window(1, 4, 12),
                                              window(1, 4, 100),
                                              window(2, 4, 20),
                                              window(1, 8, 30)};
  const auto fitted =
      fit_spec_cost_table(windows, SpecCostModel::Kind::SPS, 1800, 4096);
  ASSERT_TRUE(fitted);
  const auto table = nlohmann::json::parse(*fitted);
  EXPECT_EQ(table["sample_batch_tokens"], nlohmann::json({4, 8}));
  EXPECT_DOUBLE_EQ(table["sample_steps_per_sec"][0].get<double>(), 62.5);
  EXPECT_DOUBLE_EQ(table["sample_steps_per_sec"][1].get<double>(),
                   1000.0 / 30.0);
  EXPECT_EQ(table["metadata"]["sampled_cells"], 3);
  EXPECT_EQ(table["metadata"]["cells"][0]["selected_draft_tokens"], 288);
  EXPECT_EQ(table["metadata"]["cells"][0]["generated_draft_tokens"], 672);
  EXPECT_EQ(table["metadata"]["cells"][0]["request_rounds"], 96);
  EXPECT_EQ(table["metadata"]["cells"][0]["sample_min_kv_length"], 2000);
  EXPECT_EQ(table["metadata"]["cells"][0]["sample_max_kv_length"], 2100);
}

TEST(SpecStartupCalibrationTest, SpsProjectionWeightsCellsNotWindowCounts) {
  const std::vector<SpecSteadyWindow> windows{
      window(1, 4, 10), window(1, 4, 12), window(1, 4, 100), window(2, 4, 200)};
  const auto fitted =
      fit_spec_cost_table(windows, SpecCostModel::Kind::SPS, 1800, 4096);
  ASSERT_TRUE(fitted);
  const auto table = nlohmann::json::parse(*fitted);
  // Match the offline fitter: median(12, 200), not median(10, 12, 100, 200).
  EXPECT_DOUBLE_EQ(table["sample_steps_per_sec"][0].get<double>(),
                   1000.0 / 106.0);
}

TEST(SpecStartupCalibrationTest, RejectsDisconnectedGridBeforeExecution) {
  auto config = *parse_spec_startup_calibration(configuration().dump());
  config.model = SpecCostModel::Kind::ADDITIVE;
  const std::array<uint32_t, 3> requests{1, 4, 16};
  const std::array<uint32_t, 8> graphs{1, 2, 4, 8, 16, 32, 64, 128};
  config.request_buckets = {1, 16};
  EXPECT_FALSE(plan_spec_calibration(config, requests, graphs));
  config.request_buckets = {1, 4, 16};
  EXPECT_TRUE(plan_spec_calibration(config, requests, graphs));
}

TEST(SpecStartupCalibrationTest, AdditiveRecoversConnectedAxesInSeconds) {
  const std::vector<SpecSteadyWindow> windows{window(1, 1, 10),
                                              window(1, 2, 11),
                                              window(1, 4, 13),
                                              window(1, 8, 18),
                                              window(2, 2, 14),
                                              window(2, 4, 16),
                                              window(2, 8, 21),
                                              window(2, 16, 25),
                                              window(8, 8, 24),
                                              window(8, 16, 28),
                                              window(8, 32, 36),
                                              window(8, 64, 41)};
  const auto fitted =
      fit_spec_cost_table(windows, SpecCostModel::Kind::ADDITIVE, 1800, 4096);
  ASSERT_TRUE(fitted);
  const auto table = nlohmann::json::parse(*fitted);
  EXPECT_NEAR(table["bias_seconds"].get<double>(), 0.010, 1e-12);
  EXPECT_NEAR(table["alpha_seconds"][1].get<double>(), 0.003, 1e-12);
  EXPECT_NEAR(table["alpha_seconds"][2].get<double>(), 0.006, 1e-12);
  EXPECT_LT(
      table["metadata"]["training_error"]["mean_absolute_ms"].get<double>(),
      1e-10);
}

TEST(SpecStartupCalibrationTest,
     AdditiveRejectsDisconnectedAndNonpositiveGrid) {
  for (const auto& windows : std::vector<std::vector<SpecSteadyWindow>>{
           {window(1, 1, 10), window(2, 2, 20)},
           {window(1, 1, 10), window(1, 4, 30), window(4, 4, 1)}}) {
    EXPECT_FALSE(fit_spec_cost_table(
        windows, SpecCostModel::Kind::ADDITIVE, 1800, 4096));
  }
}

TEST(SpecStartupCalibrationTest, FitRejectsIncompleteAndOutOfBucketWindows) {
  for (uint32_t fault = 0; fault < 5; ++fault) {
    auto sample = window(1, 8, 10);
    if (fault == 0) {
      sample.steps = 0;
    } else if (fault == 1) {
      sample.last_generation = 34;
    } else if (fault == 2) {
      sample.logical_requests = 2;
    } else if (fault == 3) {
      sample.max_kv_length = 4097;
    } else {
      sample.elapsed_ms = std::numeric_limits<double>::infinity();
    }
    EXPECT_FALSE(fit_spec_cost_table(
        std::span(&sample, 1), SpecCostModel::Kind::SPS, 1800, 4096));
  }
}

TEST(SpecStartupCalibrationTest,
     BundleSelectsContextWithoutCrossingCoverageGaps) {
  auto data = bundle();
  data["profiles"].emplace_back(nlohmann::json{{"min_kv_length", 5000},
                                               {"max_kv_length", 8192},
                                               {"request_buckets", {1}},
                                               {"table", sps_table(8192)}});
  const auto model =
      SpecCostTableBundle::from_json(data.dump(), data["compatibility"].dump());
  ASSERT_TRUE(model);
  ASSERT_NE(model->lookup(4, 4096), nullptr);
  EXPECT_EQ(model->lookup(4, 4096)->max_kv_length, 4096U);
  EXPECT_EQ(model->lookup(1, 4500), nullptr);
  EXPECT_EQ(model->lookup(4, 6000), nullptr);
  ASSERT_NE(model->lookup(1, 6000), nullptr);
  EXPECT_EQ(model->lookup(1, 9000), nullptr);
  EXPECT_EQ(model->lookup(0, 1000), nullptr);
}

TEST(SpecStartupCalibrationTest, BundleRejectsIncompatibleOrCorruptTables) {
  auto data = bundle();
  EXPECT_FALSE(
      SpecCostTableBundle::from_json(data.dump(), R"({"model":"different"})"));
  for (const auto& patch :
       std::vector<nlohmann::json>{{{"schema_version", 2}},
                                   {{"schema_version", true}},
                                   {{"measurement", "device_events"}},
                                   {{"compatibility", nullptr}},
                                   {{"profiles", nlohmann::json::array()}}}) {
    auto invalid = data;
    invalid.update(patch);
    EXPECT_FALSE(SpecCostTableBundle::from_json(invalid.dump())) << invalid;
  }
  data["profiles"].emplace_back(data["profiles"][0]);
  EXPECT_FALSE(SpecCostTableBundle::from_json(data.dump()));
  data = bundle();
  data["profiles"][0]["table"]["max_kv_length"] = 8192;
  EXPECT_FALSE(SpecCostTableBundle::from_json(data.dump()));
  data = bundle();
  data["profiles"][0]["request_buckets"] = {1, 129};
  EXPECT_FALSE(SpecCostTableBundle::from_json(data.dump()));
}

TEST(SpecStartupCalibrationTest, SessionCollectsIntervalsRatherThanEndpoints) {
  auto config = *parse_spec_startup_calibration(configuration().dump());
  config.warmup_steps = 0;
  config.window_steps = 2;
  config.windows_per_point = 2;
  const SpecCalibrationPoint point{1, 8, config.contexts.front()};
  SpecCalibrationSession session(config, point);
  const std::vector<std::string> ids{"calibration-request"};
  SpecSteadyStep step;
  step.logical_requests = 1;
  step.draft_requests = 1;
  step.target_graph_rows = 8;
  step.max_kv_length = 2000;
  step.draft_tokens = 7;
  step.accepted_tokens = 2;
  step.request_ids = ids;
  for (uint64_t generation = 1; generation <= 5; ++generation) {
    step.generation = generation;
    step.completed_ns = generation * 1000000;
    session.observe(step);
    EXPECT_EQ(session.finished(), generation == 5);
  }
  ASSERT_EQ(session.windows().size(), 2U);
  EXPECT_EQ(session.windows()[0].steps, 2U);
  EXPECT_EQ(session.windows()[0].accepted_tokens, 4U);
  EXPECT_DOUBLE_EQ(session.windows()[0].elapsed_ms, 2.0);
  EXPECT_EQ(session.windows()[1].first_generation, 3U);
  EXPECT_EQ(session.windows()[1].last_generation, 5U);
}

TEST(SpecStartupCalibrationTest, SessionDiscardsTransientMembershipWindow) {
  auto config = *parse_spec_startup_calibration(configuration().dump());
  config.warmup_steps = 0;
  config.window_steps = 2;
  config.windows_per_point = 1;
  SpecCalibrationSession session(config, {1, 8, config.contexts.front()});
  const std::vector<std::string> ids{"calibration-request"};
  SpecSteadyStep step;
  step.logical_requests = 1;
  step.draft_requests = 1;
  step.target_graph_rows = 8;
  step.max_kv_length = 2000;
  step.draft_tokens = 7;
  step.request_ids = ids;
  for (uint64_t generation = 1; generation <= 6; ++generation) {
    step.generation = generation;
    step.completed_ns = generation * 1000000;
    step.logical_requests = generation == 3 ? 2 : 1;
    session.observe(step);
  }
  ASSERT_TRUE(session.finished());
  ASSERT_EQ(session.windows().size(), 1U);
  EXPECT_EQ(session.windows()[0].first_generation, 4U);
  EXPECT_EQ(session.windows()[0].last_generation, 6U);
}

TEST(SpecStartupCalibrationTest, LegacySingleTableRemainsLoadable) {
  const auto model = SpecCostTableBundle::from_json(sps_table().dump());
  ASSERT_TRUE(model);
  ASSERT_NE(model->lookup(16, 2000), nullptr);
  EXPECT_EQ(model->lookup(1, 4097), nullptr);
  EXPECT_NEAR(
      *model->lookup(16, 2000)->model.step_time_ms(16, 32), 20.0, 1e-12);
}

TEST(SpecStartupCalibrationTest,
     GraphConfigurationRoundTripsAndAllowsSparseCoverage) {
  auto data = configuration();
  data["model"] = "graph";
  data["request_buckets"] = {1, 8};
  const auto config = parse_spec_startup_calibration(data.dump());
  ASSERT_TRUE(config);
  EXPECT_EQ(config->model, SpecCostModel::Kind::GRAPH);
  const auto round_trip = parse_spec_startup_calibration(
      serialize_spec_startup_calibration(*config));
  ASSERT_TRUE(round_trip);
  EXPECT_EQ(round_trip->model, SpecCostModel::Kind::GRAPH);
  const std::array<uint32_t, 2> requests{1, 8};
  const std::array<uint32_t, 7> graphs{1, 2, 4, 8, 16, 32, 64};
  const std::vector<SpecCalibrationGraphCoverage> coverage{{1, {1, 2, 4, 8}},
                                                           {8, {16, 32, 64}}};
  const auto points =
      plan_spec_calibration(*config, requests, graphs, coverage);
  ASSERT_TRUE(points);
  EXPECT_EQ(points->size(), 7U);
  EXPECT_EQ(points->back().target_graph_rows, 64U);
}

TEST(SpecStartupCalibrationTest, GraphMediansRetainEachPhysicalBinding) {
  const std::vector<SpecSteadyWindow> windows{window(1, 4, 10),
                                              window(1, 4, 12),
                                              window(1, 4, 100),
                                              window(2, 4, 20),
                                              window(1, 8, 30)};
  const auto fitted =
      fit_spec_cost_table(windows, SpecCostModel::Kind::GRAPH, 1800, 4096);
  ASSERT_TRUE(fitted);
  const auto model = SpecCostModel::from_json(*fitted);
  ASSERT_TRUE(model);
  EXPECT_DOUBLE_EQ(*model->step_time_ms(1, 4), 12.0);
  EXPECT_DOUBLE_EQ(*model->step_time_ms(2, 4), 20.0);
  EXPECT_DOUBLE_EQ(*model->step_time_ms(1, 8), 30.0);
  EXPECT_FALSE(model->step_time_ms(2, 8));
  EXPECT_FALSE(model->step_time_ms(1, 6));
  const auto table = nlohmann::json::parse(*fitted);
  EXPECT_EQ(table["metadata"]["model"], "graph");
  EXPECT_EQ(table["metadata"]["sampled_cells"], 3);
  EXPECT_DOUBLE_EQ(
      table["metadata"]["training_error"]["mean_absolute_ms"].get<double>(),
      0.0);
  EXPECT_EQ(table["draft_width"], 7);
  EXPECT_FALSE(SpecCostTableBundle::from_json(*fitted));
  auto data = bundle();
  data["profiles"][0]["min_kv_length"] = 1800;
  data["profiles"][0]["request_buckets"] = {1, 2};
  data["profiles"][0]["table"] = table;
  const auto restored =
      SpecCostTableBundle::from_json(data.dump(), data["compatibility"].dump());
  ASSERT_TRUE(restored);
  EXPECT_EQ(restored->lookup(1, 1799), nullptr);
  ASSERT_NE(restored->lookup(2, 4096), nullptr);
  EXPECT_DOUBLE_EQ(*restored->lookup(2, 4096)->model.step_time_ms(2, 4), 20.0);
  EXPECT_EQ(restored->lookup(2, 4097), nullptr);
}

TEST(SpecStartupCalibrationTest,
     GraphBundleRejectsMismatchedIdentityAndBounds) {
  const std::vector<SpecSteadyWindow> windows{window(1, 8, 4.0)};
  const auto fitted =
      fit_spec_cost_table(windows, SpecCostModel::Kind::GRAPH, 1800, 4096);
  ASSERT_TRUE(fitted);
  auto data = bundle();
  data["profiles"][0]["min_kv_length"] = 1800;
  data["profiles"][0]["request_buckets"] = {1};
  data["profiles"][0]["table"] = nlohmann::json::parse(*fitted);
  const std::string identity = data["compatibility"].dump();
  const auto restored = SpecCostTableBundle::from_json(data.dump(), identity);
  ASSERT_TRUE(restored);
  EXPECT_NE(restored->lookup(1, 1800), nullptr);
  EXPECT_NE(restored->lookup(1, 4096), nullptr);
  EXPECT_EQ(restored->lookup(1, 1799), nullptr);
  EXPECT_EQ(restored->lookup(1, 4097), nullptr);
  EXPECT_EQ(restored->lookup(2, 2000), nullptr);
  data["compatibility"]["model"] = "different-model";
  EXPECT_FALSE(SpecCostTableBundle::from_json(data.dump(), identity));
  data["compatibility"] = nlohmann::json::parse(identity);
  data["profiles"][0]["table"]["max_kv_length"] = 8192;
  EXPECT_FALSE(SpecCostTableBundle::from_json(data.dump(), identity));
}

TEST(SpecStartupCalibrationTest, GraphProfilePreservesNonadditiveOptimum) {
  const std::vector<SpecSteadyWindow> windows{window(1, 2, 2),
                                              window(1, 4, 12),
                                              window(1, 8, 4),
                                              window(2, 2, 12),
                                              window(2, 4, 2),
                                              window(2, 8, 12),
                                              window(2, 16, 20)};
  const auto fitted =
      fit_spec_cost_table(windows, SpecCostModel::Kind::GRAPH, 1800, 4096);
  ASSERT_TRUE(fitted);
  const auto model = SpecCostModel::from_json(*fitted);
  ASSERT_TRUE(model);
  const std::array<uint32_t, 5> captured{1, 2, 4, 8, 16};
  auto measured = model->verification_costs(2, 7, captured);
  ASSERT_TRUE(measured);
  SpecCostProfile profile;
  profile.draft_requests = 2;
  profile.min_kv_length = 1800;
  profile.max_kv_length = 4096;
  profile.cost_scope = SpecCostScope::FULL_STEP;
  profile.target_graph_costs = std::move(*measured);
  SpecCostProfileStore store;
  ASSERT_TRUE(store.store(std::move(profile)));
  const auto installed = store.lookup(2, 2500);
  ASSERT_TRUE(installed);
  EXPECT_FALSE(store.lookup(3, 2500));
  EXPECT_FALSE(store.lookup(2, 4097));
  VerifyBudgetOptions options;
  options.cost_scope = installed->cost_scope;
  const auto plan = VerifyBudgetPlanner::plan(
      2, std::vector<double>(14, 0.8), installed->target_graph_costs, options);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->graph_rows, 4U);
  EXPECT_EQ(plan->draft_budget, 2U);
  double optimum = 0.0;
  for (uint32_t left = 0; left <= 7; ++left) {
    for (uint32_t right = 0; right <= 7; ++right) {
      const uint32_t rows =
          *std::lower_bound(captured.begin(), captured.end(), 2 + left + right);
      double expected = 2.0;
      for (uint32_t index = 1; index <= left; ++index) {
        expected += std::pow(0.8, index);
      }
      for (uint32_t index = 1; index <= right; ++index) {
        expected += std::pow(0.8, index);
      }
      optimum = std::max(optimum, expected / *model->step_time_ms(2, rows));
    }
  }
  EXPECT_NEAR(optimum, 3.6 / *model->step_time_ms(2, plan->graph_rows), 1e-12);
  // G-only projection and additive fitting lose this measured interaction.
  // Exact cells preserve the optimum without assuming a separable surface.
  for (const SpecCostModel::Kind kind :
       {SpecCostModel::Kind::SPS, SpecCostModel::Kind::ADDITIVE}) {
    SCOPED_TRACE(SpecCostModel::kind_name(kind));
    const auto fitted_alternative =
        fit_spec_cost_table(windows, kind, 1800, 4096);
    ASSERT_TRUE(fitted_alternative);
    const auto alternative = SpecCostModel::from_json(*fitted_alternative);
    ASSERT_TRUE(alternative);
    const auto alternative_costs =
        alternative->verification_costs(2, 7, captured);
    ASSERT_TRUE(alternative_costs);
    const auto alternative_plan = VerifyBudgetPlanner::plan(
        2, std::vector<double>(14, 0.8), *alternative_costs, options);
    ASSERT_TRUE(alternative_plan);
    EXPECT_NE(alternative_plan->graph_rows, plan->graph_rows);
  }
}

}  // namespace
}  // namespace xllm
