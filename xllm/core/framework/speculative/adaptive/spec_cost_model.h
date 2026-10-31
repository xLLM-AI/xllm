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
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace xllm {

enum class SpecCostScope : uint8_t { VERIFY_TAIL, FULL_STEP };

struct SpecGraphCost {
  uint32_t graph_rows = 0;
  // The scope selects verification-tail or full-step latency. Zero denotes
  // a warmed graph without a usable latency measurement.
  double target_time_ms = 0.0;
};

// Immutable, Host-only speculative step costs. SPS/additive JSON keeps the
// existing SGLang format; graph tables retain measured (Bcap, G) cells in ms.
// Evaluate physical graph rows, not logical B+K: padding does not reduce the
// replay cost. Bcap denotes the physical Draft request bucket, including padded
// requests, for both additive and graph tables.
class SpecCostModel final {
 public:
  enum class Kind : uint8_t { SPS, ADDITIVE, GRAPH };

  static std::string_view kind_name(Kind kind);

  // Reject malformed, ambiguous, non-finite or non-positive cost models.
  // max_kv_length bounds valid contexts and is required for exact graph cells.
  static std::optional<SpecCostModel> from_json(std::string_view text);

  Kind kind() const { return kind_; }
  uint32_t max_kv_length() const { return max_kv_length_; }

  std::optional<double> step_time_ms(uint32_t draft_requests,
                                     uint32_t target_graph_rows) const;

  // Only measured graph cells are eligible in GRAPH mode. Every returned
  // cost belongs to a captured bucket, and the smallest captured full-width
  // bucket is required before installing a profile. Other models keep their
  // existing interpolation/projection semantics.
  std::optional<std::vector<SpecGraphCost>> verification_costs(
      uint32_t draft_requests,
      uint32_t draft_width,
      std::span<const uint32_t> captured_graphs) const;

 private:
  SpecCostModel() = default;

  Kind kind_ = Kind::SPS;
  uint32_t max_batch_tokens_ = std::numeric_limits<uint32_t>::max();
  uint32_t max_kv_length_ = std::numeric_limits<uint32_t>::max();
  double bias_seconds_ = 0.0;
  std::vector<uint32_t> bs_probes_;
  std::vector<double> alpha_seconds_;
  std::vector<uint32_t> token_probes_;
  // SPS: full step milliseconds. Additive: theta seconds.
  std::vector<double> token_values_;
  // Packed (Bcap, G) keys are lookup-only; graph order comes from
  // captured_graphs.
  std::unordered_map<uint64_t, double> graph_costs_ms_;
};

}  // namespace xllm
