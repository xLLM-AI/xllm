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

#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>

namespace xllm {
namespace {

uint64_t graph_cost_key(uint32_t requests, uint32_t rows) {
  return (static_cast<uint64_t>(requests) << 32) | rows;
}

bool read_index(const nlohmann::json& value, uint32_t& output) {
  if (!value.is_number_integer()) {
    return false;
  }
  if (value.is_number_unsigned()) {
    const uint64_t number = value.get<uint64_t>();
    if (number > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
    output = static_cast<uint32_t>(number);
    return true;
  }
  const int64_t number = value.get<int64_t>();
  if (number < 0 ||
      static_cast<uint64_t>(number) > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  output = static_cast<uint32_t>(number);
  return true;
}

bool read_axis(const nlohmann::json& data,
               const char* probes_key,
               const char* values_key,
               std::vector<uint32_t>& probes,
               std::vector<double>& values) {
  const auto edges = data.find(probes_key);
  const auto samples = data.find(values_key);
  if (edges == data.end() || samples == data.end() || !edges->is_array() ||
      !samples->is_array() || edges->empty() ||
      edges->size() != samples->size()) {
    return false;
  }
  probes.reserve(edges->size());
  values.reserve(samples->size());
  for (const auto& edge : *edges) {
    uint32_t probe = 0;
    if (!read_index(edge, probe) ||
        (!probes.empty() && probe <= probes.back())) {
      return false;
    }
    probes.emplace_back(probe);
  }
  for (const auto& sample : *samples) {
    if (!sample.is_number()) {
      return false;
    }
    const double value = sample.get<double>();
    if (!std::isfinite(value)) {
      return false;
    }
    values.emplace_back(value);
  }
  return true;
}

double interpolate(const std::vector<uint32_t>& probes,
                   const std::vector<double>& values,
                   uint32_t point) {
  if (point <= probes.front()) {
    return values.front();
  }
  if (point >= probes.back()) {
    return values.back();
  }
  const auto upper = std::upper_bound(probes.begin(), probes.end(), point);
  const size_t high = static_cast<size_t>(upper - probes.begin());
  const size_t low = high - 1;
  const double fraction = static_cast<double>(point - probes[low]) /
                          static_cast<double>(probes[high] - probes[low]);
  return std::lerp(values[low], values[high], fraction);
}

}  // namespace

std::string_view SpecCostModel::kind_name(Kind kind) {
  switch (kind) {
    case Kind::SPS:
      return "sps";
    case Kind::ADDITIVE:
      return "additive";
    case Kind::GRAPH:
      return "graph";
  }
  return {};
}

std::optional<SpecCostModel> SpecCostModel::from_json(std::string_view text) {
  const auto data =
      nlohmann::json::parse(text, nullptr, /*allow_exceptions=*/false);
  if (!data.is_object()) {
    return std::nullopt;
  }
  const bool sps = data.contains("sample_batch_tokens") ||
                   data.contains("sample_steps_per_sec") ||
                   data.contains("max_batch_tokens");
  const bool additive =
      data.contains("bias_seconds") || data.contains("bs_probes") ||
      data.contains("alpha_seconds") || data.contains("m_probes") ||
      data.contains("theta_seconds");
  const bool graph = data.contains("graph_costs");
  if (static_cast<uint32_t>(sps) + static_cast<uint32_t>(additive) +
          static_cast<uint32_t>(graph) !=
      1) {
    return std::nullopt;
  }
  SpecCostModel model;
  const auto kv_bound = data.find("max_kv_length");
  if (kv_bound != data.end() && (!read_index(*kv_bound, model.max_kv_length_) ||
                                 model.max_kv_length_ == 0)) {
    return std::nullopt;
  }
  if (graph) {
    model.kind_ = Kind::GRAPH;
    const auto& cells = data["graph_costs"];
    if (kv_bound == data.end() || !cells.is_array() || cells.empty()) {
      return std::nullopt;
    }
    model.graph_costs_ms_.reserve(cells.size());
    for (const auto& cell : cells) {
      uint32_t requests = 0;
      uint32_t rows = 0;
      if (!cell.is_object() || !cell.contains("draft_requests") ||
          !cell.contains("target_graph_rows") ||
          !cell.contains("step_time_ms") ||
          !read_index(cell["draft_requests"], requests) || requests == 0 ||
          !read_index(cell["target_graph_rows"], rows) || rows < requests ||
          !cell["step_time_ms"].is_number()) {
        return std::nullopt;
      }
      const double milliseconds = cell["step_time_ms"].get<double>();
      if (!std::isfinite(milliseconds) || milliseconds <= 0.0 ||
          !std::isfinite(1000.0 / milliseconds) ||
          !model.graph_costs_ms_
               .emplace(graph_cost_key(requests, rows), milliseconds)
               .second) {
        return std::nullopt;
      }
    }
    return model;
  }
  if (sps) {
    const auto maximum = data.find("max_batch_tokens");
    if (!read_axis(data,
                   "sample_batch_tokens",
                   "sample_steps_per_sec",
                   model.token_probes_,
                   model.token_values_) ||
        model.token_probes_.front() == 0 || maximum == data.end() ||
        !read_index(*maximum, model.max_batch_tokens_) ||
        model.max_batch_tokens_ < model.token_probes_.back()) {
      return std::nullopt;
    }
    for (double& value : model.token_values_) {
      if (value <= 0.0) {
        return std::nullopt;
      }
      value = 1000.0 / value;
      if (!std::isfinite(value) || value <= 0.0) {
        return std::nullopt;
      }
    }
    return model;
  }
  model.kind_ = Kind::ADDITIVE;
  const auto bias = data.find("bias_seconds");
  if (bias == data.end() || !bias->is_number() ||
      !read_axis(data,
                 "bs_probes",
                 "alpha_seconds",
                 model.bs_probes_,
                 model.alpha_seconds_) ||
      model.bs_probes_.front() == 0 ||
      !read_axis(data,
                 "m_probes",
                 "theta_seconds",
                 model.token_probes_,
                 model.token_values_)) {
    return std::nullopt;
  }
  model.bias_seconds_ = bias->get<double>();
  const auto alpha = std::minmax_element(model.alpha_seconds_.begin(),
                                         model.alpha_seconds_.end());
  const auto theta = std::minmax_element(model.token_values_.begin(),
                                         model.token_values_.end());
  const double minimum = model.bias_seconds_ + *alpha.first + *theta.first;
  const double maximum = model.bias_seconds_ + *alpha.second + *theta.second;
  // Fitted offsets can be negative. Validate the resulting full-step cost,
  // not each coefficient separately. Interpolation remains within these bounds.
  if (!std::isfinite(model.bias_seconds_) || model.bias_seconds_ <= 0.0 ||
      !std::isfinite(minimum) || minimum <= 0.0 ||
      !std::isfinite(maximum * 1000.0)) {
    return std::nullopt;
  }
  return model;
}

std::optional<double> SpecCostModel::step_time_ms(
    uint32_t draft_requests,
    uint32_t target_graph_rows) const {
  if (draft_requests == 0 || target_graph_rows < draft_requests ||
      target_graph_rows > max_batch_tokens_) {
    return std::nullopt;
  }
  if (kind_ == Kind::GRAPH) {
    const auto found =
        graph_costs_ms_.find(graph_cost_key(draft_requests, target_graph_rows));
    return found == graph_costs_ms_.end() ? std::nullopt
                                          : std::optional(found->second);
  }
  if (kind_ == Kind::SPS) {
    const auto upper = std::upper_bound(
        token_probes_.begin(), token_probes_.end(), target_graph_rows);
    const size_t index =
        upper == token_probes_.begin()
            ? 0
            : static_cast<size_t>(upper - token_probes_.begin() - 1);
    return token_values_[index];
  }
  return 1000.0 *
         (bias_seconds_ +
          interpolate(bs_probes_, alpha_seconds_, draft_requests) +
          interpolate(token_probes_, token_values_, target_graph_rows));
}

std::optional<std::vector<SpecGraphCost>> SpecCostModel::verification_costs(
    uint32_t draft_requests,
    uint32_t draft_width,
    std::span<const uint32_t> captured_graphs) const {
  if (draft_requests == 0 || draft_width == 0 || captured_graphs.empty() ||
      captured_graphs.front() == 0 ||
      std::adjacent_find(captured_graphs.begin(),
                         captured_graphs.end(),
                         [](uint32_t lhs, uint32_t rhs) {
                           return lhs >= rhs;
                         }) != captured_graphs.end()) {
    return std::nullopt;
  }
  const uint64_t full_rows = static_cast<uint64_t>(draft_requests) *
                             (static_cast<uint64_t>(draft_width) + 1);
  const auto full = std::lower_bound(
      captured_graphs.begin(), captured_graphs.end(), full_rows);
  if (full == captured_graphs.end() || !step_time_ms(draft_requests, *full)) {
    return std::nullopt;
  }
  std::vector<SpecGraphCost> costs;
  costs.reserve(captured_graphs.size());
  for (const uint32_t rows : captured_graphs) {
    if (rows < draft_requests || rows > *full) {
      continue;
    }
    const auto milliseconds = step_time_ms(draft_requests, rows);
    if (!milliseconds) {
      if (kind_ == Kind::GRAPH) {
        continue;
      }
      return std::nullopt;
    }
    costs.emplace_back(SpecGraphCost{rows, *milliseconds});
  }
  return costs;
}

}  // namespace xllm
