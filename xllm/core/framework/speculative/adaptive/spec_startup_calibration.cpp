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

#include <Eigen/Dense>
#include <algorithm>
#include <bit>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <unordered_set>
#include <utility>

namespace xllm {
namespace {

bool supported_calibration_width(uint32_t width, bool autoregressive) {
  return autoregressive ? width >= 1 && width <= 64 : width == 5 || width == 7;
}

bool read_index(const nlohmann::json& value,
                uint32_t& result,
                bool allow_zero = false) {
  if (!value.is_number_integer() ||
      (value.is_number_unsigned() &&
       value.get<uint64_t>() > std::numeric_limits<uint32_t>::max())) {
    return false;
  }
  const int64_t number = value.get<int64_t>();
  if (number < (allow_zero ? 0 : 1) ||
      number > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  result = static_cast<uint32_t>(number);
  return true;
}

bool read_axis(const nlohmann::json& value, std::vector<uint32_t>& result) {
  if (!value.is_array() || value.empty()) {
    return false;
  }
  result.reserve(value.size());
  for (const auto& item : value) {
    uint32_t number = 0;
    if (!read_index(item, number) ||
        (!result.empty() && number <= result.back())) {
      return false;
    }
    result.emplace_back(number);
  }
  return true;
}

bool sorted_axis(std::span<const uint32_t> values) {
  return !values.empty() && values.front() > 0 &&
         std::adjacent_find(values.begin(),
                            values.end(),
                            std::greater_equal<uint32_t>()) == values.end();
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  return values.size() % 2 != 0
             ? values[middle]
             : values[middle - 1] / 2.0 + values[middle] / 2.0;
}

struct Cell {
  uint32_t requests;
  uint32_t graph_rows;
  double step_ms;
  size_t windows;
};

struct CellSamples {
  std::vector<double> step_times_ms;
  uint64_t accepted_tokens = 0;
  uint64_t selected_draft_tokens = 0;
  uint64_t rounds = 0;
  uint32_t min_kv_length = std::numeric_limits<uint32_t>::max();
  uint32_t max_kv_length = 0;
};

SpecWindowConfig session_config(const SpecStartupCalibrationConfig& config,
                                const SpecCalibrationPoint& point) {
  SpecWindowConfig result;
  result.target_graph_rows = point.target_graph_rows;
  result.min_kv_length = point.context.min_kv_length;
  result.max_kv_length = point.context.max_kv_length;
  result.warmup_steps = config.warmup_steps;
  result.window_steps = config.window_steps;
  result.max_windows = config.windows_per_point;
  result.draft_width = config.draft_width;
  return result;
}

}  // namespace

std::optional<SpecStartupCalibrationConfig> parse_spec_startup_calibration(
    std::string_view contents,
    uint32_t draft_width,
    bool autoregressive_draft) {
  if (!supported_calibration_width(draft_width, autoregressive_draft)) {
    return std::nullopt;
  }
  const auto data = nlohmann::json::parse(contents, nullptr, false);
  if (!data.is_object() || !data.contains("model") ||
      !data["model"].is_string() || !data.contains("request_buckets") ||
      !data.contains("contexts") || !data["contexts"].is_array() ||
      data["contexts"].empty()) {
    return std::nullopt;
  }
  SpecStartupCalibrationConfig config;
  config.draft_width = draft_width;
  if (data.contains("draft_width") &&
      (!read_index(data["draft_width"], config.draft_width) ||
       config.draft_width != draft_width)) {
    return std::nullopt;
  }
  const std::string model = data["model"].get<std::string>();
  if (model != "sps" && model != "additive" && model != "graph") {
    return std::nullopt;
  }
  config.model = model == "sps"     ? SpecCostModel::Kind::SPS
                 : model == "graph" ? SpecCostModel::Kind::GRAPH
                                    : SpecCostModel::Kind::ADDITIVE;
  if (!read_axis(data["request_buckets"], config.request_buckets) ||
      config.request_buckets.back() > 128 ||
      (data.contains("target_graph_rows") &&
       !read_axis(data["target_graph_rows"], config.target_graph_rows))) {
    return std::nullopt;
  }
  for (const auto& [key, destination] :
       {std::pair{"warmup_steps", &config.warmup_steps},
        {"window_steps", &config.window_steps},
        {"windows_per_point", &config.windows_per_point},
        {"token_seed", &config.token_seed}}) {
    if (data.contains(key) &&
        !read_index(data[key],
                    *destination,
                    std::string_view(key) == "warmup_steps" ||
                        std::string_view(key) == "token_seed")) {
      return std::nullopt;
    }
  }
  if (config.warmup_steps > 1000000 || config.window_steps > 4096 ||
      config.windows_per_point > 1000000) {
    return std::nullopt;
  }
  config.contexts.reserve(data["contexts"].size());
  for (const auto& value : data["contexts"]) {
    SpecCalibrationContext context;
    if (!value.is_object() || !value.contains("prefill_tokens") ||
        !value.contains("min_kv_length") || !value.contains("max_kv_length") ||
        !read_index(value["prefill_tokens"], context.prefill_tokens) ||
        !read_index(value["min_kv_length"], context.min_kv_length) ||
        !read_index(value["max_kv_length"], context.max_kv_length) ||
        context.min_kv_length > context.prefill_tokens ||
        context.prefill_tokens >= context.max_kv_length ||
        (!config.contexts.empty() &&
         context.min_kv_length <= config.contexts.back().max_kv_length)) {
      return std::nullopt;
    }
    // Capacity must cover the worst accepted length, the bootstrap, and both
    // in-flight Slots. Never rely on rejection to stay within the KV bucket.
    const uint64_t rounds =
        static_cast<uint64_t>(config.warmup_steps) +
        static_cast<uint64_t>(config.window_steps) * config.windows_per_point +
        4;
    if (rounds * (draft_width + 1) + context.prefill_tokens >
        context.max_kv_length) {
      return std::nullopt;
    }
    config.contexts.emplace_back(context);
  }
  return config;
}

std::optional<SpecStartupCalibrationConfig> default_spec_startup_calibration(
    uint32_t max_positions,
    uint32_t draft_width,
    bool autoregressive_draft) {
  if (!supported_calibration_width(draft_width, autoregressive_draft)) {
    return std::nullopt;
  }
  SpecStartupCalibrationConfig config;
  config.draft_width = draft_width;
  config.model = SpecCostModel::Kind::ADDITIVE;
  const uint32_t max_kv_length = std::min(max_positions, uint32_t{4096});
  const uint64_t reserved_tokens =
      (static_cast<uint64_t>(config.warmup_steps) +
       static_cast<uint64_t>(config.window_steps) * config.windows_per_point +
       4) *
      (draft_width + 1);
  if (max_kv_length <= reserved_tokens) {
    return std::nullopt;
  }
  const uint32_t prefill_tokens = std::min(
      uint32_t{2048}, max_kv_length - static_cast<uint32_t>(reserved_tokens));
  config.contexts.emplace_back(
      SpecCalibrationContext{prefill_tokens, 1, max_kv_length});
  return config;
}

std::string serialize_spec_startup_calibration(
    const SpecStartupCalibrationConfig& config) {
  nlohmann::json contexts = nlohmann::json::array();
  for (const auto& context : config.contexts) {
    contexts.emplace_back(
        nlohmann::json{{"prefill_tokens", context.prefill_tokens},
                       {"min_kv_length", context.min_kv_length},
                       {"max_kv_length", context.max_kv_length}});
  }
  nlohmann::json data{{"model", SpecCostModel::kind_name(config.model)},
                      {"request_buckets", config.request_buckets},
                      {"contexts", std::move(contexts)},
                      {"warmup_steps", config.warmup_steps},
                      {"window_steps", config.window_steps},
                      {"windows_per_point", config.windows_per_point},
                      {"token_seed", config.token_seed}};
  data["draft_width"] = config.draft_width;
  if (!config.target_graph_rows.empty()) {
    data["target_graph_rows"] = config.target_graph_rows;
  }
  return data.dump();
}

std::optional<uint32_t> spec_calibration_warmup_length(
    uint32_t max_kv_length,
    uint32_t max_positions,
    uint32_t logical_block_size,
    uint32_t default_length,
    uint32_t draft_width,
    bool autoregressive_draft) {
  const uint64_t in_flight_tokens =
      2 * (static_cast<uint64_t>(draft_width) + 1);
  if (!supported_calibration_width(draft_width, autoregressive_draft) ||
      max_kv_length == 0 || max_kv_length > max_positions ||
      logical_block_size == 0 || default_length < 2 ||
      max_positions < in_flight_tokens + 2) {
    return std::nullopt;
  }
  const uint64_t page_size = logical_block_size;
  const uint64_t max_pages = (max_positions + page_size - 1) / page_size;
  const uint64_t minimum_pages = 16 + (draft_width + page_size - 1) / page_size;
  const auto covers = [&](uint32_t length) {
    // Match the prepared block-draft compact binding, including its bonus-token
    // allowance. Increasing a 2048-token warmup to 4089 would waste physical KV
    // allocations: its existing 32-page capture already covers 4096 tokens.
    const uint64_t required_pages =
        (static_cast<uint64_t>(length) + draft_width - 1 + page_size - 1) /
        page_size;
    const uint64_t pages = std::min(
        std::bit_ceil(std::max(required_pages, minimum_pages)), max_pages);
    return pages * page_size + 1 >= max_kv_length;
  };
  uint32_t low = std::min(
      default_length, max_positions - static_cast<uint32_t>(in_flight_tokens));
  uint32_t high = max_positions - static_cast<uint32_t>(in_flight_tokens);
  if (!covers(high)) {
    return std::nullopt;
  }
  while (low < high) {
    const uint32_t middle = low + (high - low) / 2;
    if (covers(middle)) {
      high = middle;
    } else {
      low = middle + 1;
    }
  }
  return low;
}

std::optional<std::vector<SpecCalibrationPoint>> plan_spec_calibration(
    const SpecStartupCalibrationConfig& config,
    std::span<const uint32_t> captured_requests,
    std::span<const uint32_t> captured_target_rows,
    std::span<const SpecCalibrationGraphCoverage> graph_coverage) {
  if ((config.draft_width != 5 && config.draft_width != 7) ||
      !sorted_axis(captured_requests) || !sorted_axis(captured_target_rows) ||
      !sorted_axis(config.request_buckets) || config.contexts.empty() ||
      (!config.target_graph_rows.empty() &&
       !sorted_axis(config.target_graph_rows))) {
    return std::nullopt;
  }
  uint32_t previous_request = 0;
  for (const auto& coverage : graph_coverage) {
    if (coverage.requests <= previous_request ||
        !std::binary_search(captured_requests.begin(),
                            captured_requests.end(),
                            coverage.requests) ||
        !sorted_axis(coverage.target_graph_rows)) {
      return std::nullopt;
    }
    previous_request = coverage.requests;
    for (const uint32_t rows : coverage.target_graph_rows) {
      if (rows < coverage.requests ||
          !std::binary_search(
              captured_target_rows.begin(), captured_target_rows.end(), rows)) {
        return std::nullopt;
      }
    }
  }
  std::vector<SpecCalibrationPoint> points;
  for (const auto& context : config.contexts) {
    for (const uint32_t requests : config.request_buckets) {
      const auto full = std::lower_bound(
          captured_target_rows.begin(),
          captured_target_rows.end(),
          static_cast<uint64_t>(requests) * (config.draft_width + 1));
      if (!std::binary_search(
              captured_requests.begin(), captured_requests.end(), requests) ||
          full == captured_target_rows.end()) {
        return std::nullopt;
      }
      std::span<const uint32_t> available = captured_target_rows;
      if (!graph_coverage.empty()) {
        const auto found = std::find_if(
            graph_coverage.begin(),
            graph_coverage.end(),
            [&](const auto& entry) { return entry.requests == requests; });
        if (found == graph_coverage.end()) {
          return std::nullopt;
        }
        available = found->target_graph_rows;
      }
      const auto& selected = config.target_graph_rows;
      const std::span<const uint32_t> candidates =
          selected.empty() ? available : std::span(selected);
      bool has_full = false;
      for (const uint32_t rows : candidates) {
        if (rows < requests || rows > *full) {
          continue;
        }
        if (!std::binary_search(available.begin(), available.end(), rows)) {
          return std::nullopt;
        }
        points.emplace_back(SpecCalibrationPoint{requests, rows, context});
        has_full = has_full || rows == *full;
      }
      if (!has_full) {
        return std::nullopt;
      }
    }
  }
  if (config.model == SpecCostModel::Kind::ADDITIVE) {
    std::unordered_set<uint32_t> connected_requests{
        config.request_buckets.front()};
    std::unordered_set<uint32_t> connected_graphs;
    bool changed = true;
    while (changed) {
      changed = false;
      for (const auto& point : points) {
        if (point.context.max_kv_length !=
            config.contexts.front().max_kv_length) {
          break;
        }
        if (connected_requests.contains(point.requests)) {
          changed = connected_graphs.insert(point.target_graph_rows).second ||
                    changed;
        }
        if (connected_graphs.contains(point.target_graph_rows)) {
          changed = connected_requests.insert(point.requests).second || changed;
        }
      }
    }
    if (connected_requests.size() != config.request_buckets.size()) {
      return std::nullopt;
    }
  }
  return points;
}

namespace {

std::optional<std::string> fit_full_step_cost_table(
    std::span<const SpecSteadyWindow> windows,
    SpecCostModel::Kind kind,
    uint32_t min_kv_length,
    uint32_t max_kv_length,
    uint32_t draft_width,
    bool autoregressive_draft) {
  if (!supported_calibration_width(draft_width, autoregressive_draft) ||
      windows.empty() || min_kv_length == 0 || min_kv_length > max_kv_length) {
    return std::nullopt;
  }
  std::map<std::pair<uint32_t, uint32_t>, CellSamples> samples;
  for (const auto& window : windows) {
    if (window.logical_requests == 0 || window.logical_requests > 128 ||
        window.logical_requests != window.draft_requests ||
        window.target_graph_rows < window.draft_requests || window.steps == 0 ||
        (autoregressive_draft &&
         window.target_graph_rows <
             static_cast<uint64_t>(window.logical_requests) *
                 (draft_width + 1)) ||
        window.steps > 4096 || window.first_generation == 0 ||
        window.last_generation <= window.first_generation ||
        window.last_generation - window.first_generation != window.steps ||
        window.min_kv_length < min_kv_length ||
        window.max_kv_length > max_kv_length ||
        window.min_kv_length > window.max_kv_length ||
        window.draft_tokens !=
            std::min(static_cast<uint64_t>(window.draft_requests) * draft_width,
                     static_cast<uint64_t>(window.target_graph_rows -
                                           window.draft_requests)) *
                window.steps ||
        window.accepted_tokens > window.draft_tokens ||
        !std::isfinite(window.elapsed_ms) || window.elapsed_ms <= 0.0) {
      return std::nullopt;
    }
    const double cost = window.elapsed_ms / window.steps;
    if (!std::isfinite(cost) || cost <= 0.0 || !std::isfinite(1000.0 / cost)) {
      return std::nullopt;
    }
    auto& sampled = samples[{window.draft_requests, window.target_graph_rows}];
    sampled.step_times_ms.emplace_back(cost);
    sampled.accepted_tokens += window.accepted_tokens;
    sampled.selected_draft_tokens += window.draft_tokens;
    sampled.rounds += window.steps;
    sampled.min_kv_length =
        std::min(sampled.min_kv_length, window.min_kv_length);
    sampled.max_kv_length =
        std::max(sampled.max_kv_length, window.max_kv_length);
  }
  std::vector<Cell> cells;
  cells.reserve(samples.size());
  std::map<uint32_t, std::vector<double>> rows;
  std::map<uint32_t, size_t> batches;
  for (auto& [key, sampled] : samples) {
    const double cost = median(sampled.step_times_ms);
    cells.emplace_back(
        Cell{key.first, key.second, cost, sampled.step_times_ms.size()});
    rows[key.second].emplace_back(cost);
    batches.emplace(key.first, 0);
  }
  nlohmann::json table;
  std::vector<double> predictions;
  predictions.reserve(cells.size());
  if (kind == SpecCostModel::Kind::GRAPH) {
    table["graph_costs"] = nlohmann::json::array();
    for (const auto& cell : cells) {
      table["graph_costs"].emplace_back(
          nlohmann::json{{"draft_requests", cell.requests},
                         {"target_graph_rows", cell.graph_rows},
                         {"step_time_ms", cell.step_ms}});
      predictions.emplace_back(cell.step_ms);
    }
  } else if (kind == SpecCostModel::Kind::SPS) {
    std::map<uint32_t, double> medians;
    table["sample_batch_tokens"] = nlohmann::json::array();
    table["sample_steps_per_sec"] = nlohmann::json::array();
    for (auto& [row, times] : rows) {
      const double cost = median(times);
      medians.emplace(row, cost);
      table["sample_batch_tokens"].emplace_back(row);
      table["sample_steps_per_sec"].emplace_back(1000.0 / cost);
    }
    table["max_batch_tokens"] = rows.rbegin()->first;
    for (const auto& cell : cells) {
      predictions.emplace_back(medians.at(cell.graph_rows));
    }
  } else if (kind == SpecCostModel::Kind::ADDITIVE) {
    std::map<uint32_t, size_t> row_columns;
    size_t column = 1;
    for (auto& [batch, index] : batches) {
      index = batch == batches.begin()->first ? 0 : column++;
    }
    for (const auto& [row, times] : rows) {
      row_columns.emplace(row, row == rows.begin()->first ? 0 : column++);
    }
    Eigen::MatrixXd design = Eigen::MatrixXd::Zero(cells.size(), column);
    Eigen::VectorXd target(cells.size());
    for (size_t index = 0; index < cells.size(); ++index) {
      const auto& cell = cells[index];
      design(index, 0) = 1.0;
      if (batches.at(cell.requests) != 0) {
        design(index, batches.at(cell.requests)) = 1.0;
      }
      if (row_columns.at(cell.graph_rows) != 0) {
        design(index, row_columns.at(cell.graph_rows)) = 1.0;
      }
      target(index) = cell.step_ms;
    }
    const auto solver = design.colPivHouseholderQr();
    if (static_cast<size_t>(solver.rank()) != column) {
      return std::nullopt;
    }
    const Eigen::VectorXd coefficients = solver.solve(target);
    if (!coefficients.allFinite()) {
      return std::nullopt;
    }
    table["bias_seconds"] = coefficients(0) / 1000.0;
    table["bs_probes"] = nlohmann::json::array();
    table["alpha_seconds"] = nlohmann::json::array();
    for (const auto& [batch, index] : batches) {
      table["bs_probes"].emplace_back(batch);
      table["alpha_seconds"].emplace_back(
          index == 0 ? 0.0 : coefficients(index) / 1000.0);
    }
    table["m_probes"] = nlohmann::json::array();
    table["theta_seconds"] = nlohmann::json::array();
    for (const auto& [row, index] : row_columns) {
      table["m_probes"].emplace_back(row);
      table["theta_seconds"].emplace_back(
          index == 0 ? 0.0 : coefficients(index) / 1000.0);
    }
    const Eigen::VectorXd fitted = design * coefficients;
    for (Eigen::Index index = 0; index < fitted.size(); ++index) {
      predictions.emplace_back(fitted(index));
    }
  } else {
    return std::nullopt;
  }
  double absolute = 0.0;
  double square = 0.0;
  double relative = 0.0;
  nlohmann::json cell_records = nlohmann::json::array();
  for (size_t index = 0; index < cells.size(); ++index) {
    const auto& cell = cells[index];
    const auto& sampled = samples.at({cell.requests, cell.graph_rows});
    const double error = std::abs(predictions[index] - cell.step_ms);
    absolute += error / static_cast<double>(cells.size());
    square = std::hypot(square, error);
    relative = std::max(relative, error / cell.step_ms * 100.0);
    cell_records.emplace_back(
        nlohmann::json{{"draft_requests", cell.requests},
                       {"target_graph_rows", cell.graph_rows},
                       {"median_step_ms", cell.step_ms},
                       {"windows", cell.windows},
                       {"sample_min_kv_length", sampled.min_kv_length},
                       {"sample_max_kv_length", sampled.max_kv_length},
                       {"accepted_tokens", sampled.accepted_tokens},
                       {"selected_draft_tokens", sampled.selected_draft_tokens},
                       {"generated_draft_tokens",
                        sampled.rounds * cell.requests * draft_width},
                       {"request_rounds", sampled.rounds * cell.requests}});
  }
  const double rms = square / std::sqrt(static_cast<double>(cells.size()));
  if (!std::isfinite(absolute) || !std::isfinite(rms) ||
      !std::isfinite(relative)) {
    return std::nullopt;
  }
  table["max_kv_length"] = max_kv_length;
  table["draft_width"] = draft_width;
  table["metadata"] = {{"measurement", "steady_full_step"},
                       {"model", SpecCostModel::kind_name(kind)},
                       {"min_kv_length", min_kv_length},
                       {"input_windows", windows.size()},
                       {"sampled_cells", cells.size()},
                       {"cells", std::move(cell_records)},
                       {"training_error",
                        {{"mean_absolute_ms", absolute},
                         {"root_mean_square_ms", rms},
                         {"max_relative_percent", relative}}}};
  const std::string serialized = table.dump();
  return SpecCostModel::from_json(serialized) ? std::optional(serialized)
                                              : std::nullopt;
}

}  // namespace

std::optional<std::string> fit_spec_cost_table(
    std::span<const SpecSteadyWindow> windows,
    SpecCostModel::Kind kind,
    uint32_t min_kv_length,
    uint32_t max_kv_length,
    uint32_t draft_width) {
  return fit_full_step_cost_table(windows,
                                  kind,
                                  min_kv_length,
                                  max_kv_length,
                                  draft_width,
                                  /*autoregressive_draft=*/false);
}

std::optional<std::string> fit_draft_step_cost_table(
    std::span<const SpecSteadyWindow> windows,
    SpecCostModel::Kind kind,
    uint32_t min_kv_length,
    uint32_t max_kv_length,
    uint32_t draft_width) {
  return fit_full_step_cost_table(windows,
                                  kind,
                                  min_kv_length,
                                  max_kv_length,
                                  draft_width,
                                  /*autoregressive_draft=*/true);
}

SpecCalibrationSession::SpecCalibrationSession(
    const SpecStartupCalibrationConfig& config,
    SpecCalibrationPoint point)
    : point_(point), collector_(session_config(config, point)) {
  windows_.reserve(config.windows_per_point);
}

void SpecCalibrationSession::observe(const SpecSteadyStep& step) {
  if (step.logical_requests != point_.requests) {
    collector_.reset();
    return;
  }
  const auto completed = collector_.observe(step);
  if (completed) {
    windows_.emplace_back(*completed);
  }
}

bool SpecCalibrationSession::finished() const { return collector_.finished(); }

std::optional<SpecCostTableBundle> SpecCostTableBundle::from_json(
    std::string_view contents,
    std::string_view expected_compatibility) {
  const auto data = nlohmann::json::parse(contents, nullptr, false);
  if (!data.is_object()) {
    return std::nullopt;
  }
  SpecCostTableBundle bundle;
  if (!data.contains("schema_version")) {
    // Older unversioned tables predate Draft5 support and describe Draft7.
    // A newly fitted bare table records its width explicitly; never silently
    // apply a legacy Draft7 cost curve to a Draft5 startup.
    uint32_t draft_width = 7;
    if ((data.contains("draft_width") &&
         !read_index(data["draft_width"], draft_width)) ||
        (draft_width != 5 && draft_width != 7)) {
      return std::nullopt;
    }
    if (!expected_compatibility.empty()) {
      const auto expected =
          nlohmann::json::parse(expected_compatibility, nullptr, false);
      if (!expected.is_object() ||
          (expected.contains("num_speculative_tokens") &&
           expected["num_speculative_tokens"] != draft_width)) {
        return std::nullopt;
      }
    }
    auto model = SpecCostModel::from_json(contents);
    // Exact measurements require explicit request/context coverage and model
    // compatibility; do not load them through the legacy unrestricted path.
    if (!model || model->kind() == SpecCostModel::Kind::GRAPH) {
      return std::nullopt;
    }
    bundle.entries_.emplace_back(
        SpecCostTableEntry{1, model->max_kv_length(), {}, std::move(*model)});
    return bundle;
  }
  uint32_t version = 0;
  if (!read_index(data["schema_version"], version) || version != 1 ||
      data.value("measurement", nlohmann::json()) != "steady_full_step" ||
      !data.contains("compatibility") || !data["compatibility"].is_object() ||
      data["compatibility"].empty() || !data.contains("profiles") ||
      !data["profiles"].is_array() || data["profiles"].empty()) {
    return std::nullopt;
  }
  if (!expected_compatibility.empty()) {
    const auto expected =
        nlohmann::json::parse(expected_compatibility, nullptr, false);
    if (!expected.is_object() || expected != data["compatibility"]) {
      return std::nullopt;
    }
  }
  for (const auto& value : data["profiles"]) {
    uint32_t low = 0;
    uint32_t high = 0;
    std::vector<uint32_t> requests;
    if (!value.is_object() || !value.contains("min_kv_length") ||
        !value.contains("max_kv_length") ||
        !value.contains("request_buckets") || !value.contains("table") ||
        !read_index(value["min_kv_length"], low) ||
        !read_index(value["max_kv_length"], high) || low > high ||
        !read_axis(value["request_buckets"], requests) ||
        requests.back() > 128 ||
        (!bundle.entries_.empty() &&
         low <= bundle.entries_.back().max_kv_length)) {
      return std::nullopt;
    }
    auto model = SpecCostModel::from_json(value["table"].dump());
    if (!model || model->max_kv_length() != high) {
      return std::nullopt;
    }
    bundle.entries_.emplace_back(
        SpecCostTableEntry{low, high, std::move(requests), std::move(*model)});
  }
  return bundle;
}

const SpecCostTableEntry* SpecCostTableBundle::lookup(
    uint32_t requests,
    uint32_t max_kv_length) const {
  if (requests == 0 || max_kv_length == 0) {
    return nullptr;
  }
  const auto entry =
      std::lower_bound(entries_.begin(),
                       entries_.end(),
                       max_kv_length,
                       [](const SpecCostTableEntry& item, uint32_t length) {
                         return item.max_kv_length < length;
                       });
  if (entry == entries_.end() || entry->min_kv_length > max_kv_length ||
      (!entry->request_buckets.empty() &&
       !std::binary_search(entry->request_buckets.begin(),
                           entry->request_buckets.end(),
                           requests))) {
    return nullptr;
  }
  return &*entry;
}

}  // namespace xllm
