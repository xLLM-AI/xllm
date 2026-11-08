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

#include <algorithm>
#include <cstdint>
#include <optional>

#include "core/framework/model/model_params_view.h"
#include "core/framework/model/rec_model_params.h"
#include "core/util/rec_model_utils.h"

namespace xllm {

// ModelParamsView retains its cloned owner; Rec snapshots own the native
// aggregate directly. Tensor handles intentionally share storage.
inline ModelParamsView copy_graph_execution_input(
    const ModelParamsView& params) {
  return params.clone().view();
}

inline RecModelParams copy_graph_execution_input(const RecModelParams& params) {
  return params.clone();
}

// Borrow unchanged execution input. Detach host metadata only when graph
// preparation changes it; tensor storage deliberately remains shared.
// The source must outlive this adapter, including when it already owns a view.
template <typename Params>
class GraphExecutionInput final {
 public:
  explicit GraphExecutionInput(const Params& source) : source_(source) {}
  GraphExecutionInput(Params&&) = delete;
  GraphExecutionInput(const GraphExecutionInput&) = delete;
  GraphExecutionInput& operator=(const GraphExecutionInput&) = delete;
  GraphExecutionInput(GraphExecutionInput&&) = default;
  GraphExecutionInput& operator=(GraphExecutionInput&&) = delete;

  const Params& params() const {
    return owned_params_ ? *owned_params_ : source_;
  }

  Params& mutable_params() {
    if (!owned_params_) {
      owned_params_.emplace(copy_graph_execution_input(source_));
    }
    return *owned_params_;
  }

  void pad_dp_token_counts(int32_t token_count) {
    const auto& token_counts = params().parallel.dp_global_token_nums;
    if (token_counts.size() <= 1 || std::all_of(token_counts.begin(),
                                                token_counts.end(),
                                                [token_count](int32_t count) {
                                                  return count == token_count;
                                                })) {
      return;
    }
    auto& padded_counts = mutable_params().parallel.dp_global_token_nums;
    std::fill(padded_counts.begin(), padded_counts.end(), token_count);
  }

 private:
  const Params& source_;
  std::optional<Params> owned_params_;
};

inline bool is_rec_multi_round_graph_input(const ModelParamsView&) {
  return false;
}

inline bool is_rec_multi_round_graph_input(const RecModelParams&) {
  return is_rec_multi_round_mode();
}

inline bool has_llmrec_graph_input(const ModelParamsView&) { return false; }

inline bool has_llmrec_graph_input(const RecModelParams& params) {
  return params.has_llmrec_params();
}

}  // namespace xllm
