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

#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/verify_layout.h"

namespace xllm {

struct SpecMetrics {
  // Accepted drafts at one zero-based position, summed over the batch.
  std::vector<int64_t> num_accepted_tokens_per_pos;
  std::vector<SpeculativeTokenStats> sequence_stats;
  // Batch totals, accumulated by make_spec_metrics alongside sequence_stats.
  int64_t num_committed_tokens = 0;
  int64_t num_draft_tokens = 0;
  int64_t num_accepted_tokens = 0;
  int64_t constrained_num_accepted_tokens = 0;
  int64_t constrained_num_draft_tokens = 0;
};

// The output contains one target replacement or bonus token, plus the accepted
// draft prefix. Column 0 is the first committed token.
SpecMetrics make_spec_metrics(const torch::Tensor& tokens,
                              const std::vector<VerifyRowLayout>& row_layouts);

void record_spec_metrics(const SpecMetrics& metrics);

}  // namespace xllm
