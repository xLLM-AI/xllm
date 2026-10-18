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

#include <torch/types.h>

#include <cstdint>
#include <vector>

#include "runtime/options.h"

namespace xllm {

// Decides per-seq speculative validate prefix lengths based on draft token
// path probabilities and profiled validate time. Greedy algorithm: candidates
// sorted by path_prob descending, accepted if estimated throughput improves.
class AdaptiveSpeculativeController final {
 public:
  explicit AdaptiveSpeculativeController(const runtime::Options& options);
  ~AdaptiveSpeculativeController() = default;

  bool enabled() const;
  // Per-step conditional accept probabilities [batch, n_spec]; the controller
  // chain-rule multiplies them into path acceptance for pruning.
  std::vector<int32_t> select_pruned_prefix_lengths(
      const torch::Tensor& selected_probs_by_step,
      double full_draft_time_ms,
      const std::vector<double>& per_seq_kv_lens) const;

 private:
  bool enabled_ = false;
  double min_gain_ = 0.0;
};

}  // namespace xllm
