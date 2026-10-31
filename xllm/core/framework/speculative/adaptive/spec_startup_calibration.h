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

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/framework/speculative/adaptive/spec_cost_model.h"
#include "core/framework/speculative/adaptive/spec_window_collector.h"

namespace xllm {

// Startup control-plane messages, never issued between serving decode rounds.
enum class SpecCalibrationAction : uint8_t {
  CAPABILITIES = 1,
  BEGIN_SAMPLE = 2,
  FINISH_SAMPLE = 3,
  INSTALL_COSTS = 4,
};

struct SpecCalibrationCommand {
  SpecCalibrationAction action = SpecCalibrationAction::CAPABILITIES;
  std::string payload;
};

struct SpecCalibrationContext {
  uint32_t prefill_tokens = 0;
  uint32_t min_kv_length = 1;
  uint32_t max_kv_length = 0;
};

struct SpecStartupCalibrationConfig {
  SpecCostModel::Kind model = SpecCostModel::Kind::SPS;
  std::vector<uint32_t> request_buckets;
  std::vector<uint32_t> target_graph_rows;
  std::vector<SpecCalibrationContext> contexts;
  uint32_t warmup_steps = 8;
  uint32_t window_steps = 32;
  uint32_t windows_per_point = 3;
  uint32_t token_seed = 1;
  uint32_t draft_width = 7;
};

struct SpecCalibrationPoint {
  uint32_t requests = 0;
  uint32_t target_graph_rows = 0;
  SpecCalibrationContext context;
};

// Whole-round graphs bind actual B as well as G. An empty coverage list keeps
// the split-graph policy; otherwise each listed request bucket is explicit.
struct SpecCalibrationGraphCoverage {
  uint32_t requests = 0;
  std::vector<uint32_t> target_graph_rows;
};

std::optional<SpecStartupCalibrationConfig> parse_spec_startup_calibration(
    std::string_view contents,
    uint32_t draft_width = 7,
    bool autoregressive_draft = false);

// Default contexts are bounded by the model; request buckets are resolved from
// the actual captured geometry after warmup, not guessed from HTTP concurrency.
std::optional<SpecStartupCalibrationConfig> default_spec_startup_calibration(
    uint32_t max_positions,
    uint32_t draft_width = 7,
    bool autoregressive_draft = false);

// Materialize numeric defaults; an omitted target axis selects legal captures.
std::string serialize_spec_startup_calibration(
    const SpecStartupCalibrationConfig& config);

// Keep the legacy bound when compact-page rounding already covers the sample;
// otherwise choose the smallest sufficient capture. Both in-flight allocations
// of draft_width + 1 tokens must fit the model limit.
std::optional<uint32_t> spec_calibration_warmup_length(
    uint32_t max_kv_length,
    uint32_t max_positions,
    uint32_t logical_block_size,
    uint32_t default_length,
    uint32_t draft_width = 7,
    bool autoregressive_draft = false);

// Only captured physical buckets are legal. Full width is the first captured
// bucket covering (draft_width + 1)*Bcap, which need not be that exact size.
std::optional<std::vector<SpecCalibrationPoint>> plan_spec_calibration(
    const SpecStartupCalibrationConfig& config,
    std::span<const uint32_t> captured_requests,
    std::span<const uint32_t> captured_target_rows,
    std::span<const SpecCalibrationGraphCoverage> graph_coverage = {});

// Host-only startup aggregation using exact graph-cell medians, SPS projection
// or an identifiable additive model. No device/runtime or subprocess is
// involved. The serialized table includes training diagnostics.
std::optional<std::string> fit_spec_cost_table(
    std::span<const SpecSteadyWindow> windows,
    SpecCostModel::Kind kind,
    uint32_t min_kv_length,
    uint32_t max_kv_length,
    uint32_t draft_width = 7);

// Reuses the same median/SPS/additive fitter on complete autoregressive draft
// cycles. Each candidate step count has its own measured full-cycle windows.
std::optional<std::string> fit_draft_step_cost_table(
    std::span<const SpecSteadyWindow> windows,
    SpecCostModel::Kind kind,
    uint32_t min_kv_length,
    uint32_t max_kv_length,
    uint32_t draft_width);

// The owner starts/replaces a sample only after draining the pipeline.
// observe() runs on the existing result-retirement thread, without file I/O.
class SpecCalibrationSession final {
 public:
  SpecCalibrationSession(const SpecStartupCalibrationConfig& config,
                         SpecCalibrationPoint point);
  void observe(const SpecSteadyStep& step);
  bool finished() const;
  std::span<const SpecSteadyWindow> windows() const { return windows_; }
  uint32_t target_graph_rows() const { return point_.target_graph_rows; }

 private:
  SpecCalibrationPoint point_;
  SpecSteadyWindowCollector collector_;
  std::vector<SpecSteadyWindow> windows_;
};

struct SpecCostTableEntry {
  uint32_t min_kv_length;
  uint32_t max_kv_length;
  std::vector<uint32_t> request_buckets;
  SpecCostModel model;
};

// Versioned immutable context profiles. Legacy single-table JSON remains one
// unrestricted request profile. Gaps/out-of-coverage return no table rather
// than silently applying a different context's measurements.
class SpecCostTableBundle final {
 public:
  static std::optional<SpecCostTableBundle> from_json(
      std::string_view contents,
      std::string_view expected_compatibility = {});

  const std::vector<SpecCostTableEntry>& entries() const { return entries_; }
  const SpecCostTableEntry* lookup(uint32_t requests,
                                   uint32_t max_kv_length) const;

 private:
  std::vector<SpecCostTableEntry> entries_;
};

}  // namespace xllm
