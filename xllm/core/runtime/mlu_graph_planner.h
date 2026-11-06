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

#include <cstddef>
#include <cstdint>
#include <variant>

#include "absl/container/inlined_vector.h"
#include "core/framework/model/model_params_view.h"
#include "core/runtime/options.h"
#include "models/model_registry.h"

namespace xllm {
class RecModelParams;
}  // namespace xllm

namespace xllm::mlu {

int64_t get_block_capacity(int64_t max_kv_seq_len, int32_t block_size);

int64_t get_graph_token_capacity(const runtime::Options& options);

struct GraphLayout {
  int64_t num_reqs = 0;
  int64_t padded_num_reqs = 0;
  int64_t tokens_per_request = 1;
  int64_t attention_row_query_len = 1;
  int64_t padded_num_tokens = 0;
  int64_t cache_pad_slot = 0;
};

struct GraphKey {
  int64_t padded_num_tokens = 0;
  int64_t padded_num_reqs = 0;
  int64_t attention_row_query_len = 1;
  bool has_input_embedding = false;
  bool grouped = false;
  absl::InlinedVector<int64_t, 3> multi_block_table_column_counts;
};

bool operator==(const GraphKey& lhs, const GraphKey& rhs);

class GraphKeyHash final {
 public:
  std::size_t operator()(const GraphKey& key) const;
};

enum class EagerReason : int8_t {
  NON_DECODE,
  UNSUPPORTED_INPUT,
  INVALID_LAYOUT,
  TOKEN_LIMIT,
  HISTORY_LIMIT,
};

struct GraphPlan {
  GraphLayout layout;
  GraphKey key;
  int64_t history_capacity = 0;
  int64_t main_block_table_columns = 0;
};

using GraphDecision = std::variant<GraphPlan, EagerReason>;

struct GraphPlannerConfig {
  runtime::Options options;
  MtpModelCapabilities capabilities;
  bool state_required = false;
  int64_t history_capacity = 0;
  int64_t max_tokens = 0;
};

class MluGraphPlanner final {
 public:
  explicit MluGraphPlanner(GraphPlannerConfig config);

  GraphDecision plan(int64_t tokens, const ModelParamsView& params) const;
  GraphDecision plan(int64_t tokens, const RecModelParams& params) const;

 private:
  GraphPlannerConfig config_;
};

}  // namespace xllm::mlu
