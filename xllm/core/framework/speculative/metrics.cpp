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

#include "core/framework/speculative/metrics.h"

#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/metrics.h"

namespace xllm {
namespace {

void accumulate_verify_rows(
    const std::vector<SpecVerifyStats::RequestStats>& requests,
    SpecMetrics& metrics) {
  for (size_t req_index = 0; req_index < requests.size(); ++req_index) {
    const SpecVerifyStats::RequestStats& req = requests[req_index];
    const int64_t num_draft_tokens = req.num_draft_tokens;

    SpeculativeTokenStats& request_stats = metrics.sequence_stats[req_index];
    request_stats.proposed_tokens = num_draft_tokens;
    request_stats.verified_rounds = 1;
    metrics.num_draft_tokens += num_draft_tokens;

    const int64_t accepted = static_cast<int64_t>(req.num_accepted_tokens);
    request_stats.accepted_tokens = accepted;
    metrics.num_accepted_tokens += accepted;
    auto& per_pos = metrics.num_accepted_tokens_per_pos;
    if (per_pos.size() < static_cast<size_t>(accepted)) {
      per_pos.resize(static_cast<size_t>(accepted), 0);
    }
    for (size_t pos = 0; pos < static_cast<size_t>(accepted); ++pos) {
      ++per_pos[pos];
    }
    // Fully-rejected constrained rows still count their proposals.
    if (req.json_constrained) {
      metrics.constrained_num_draft_tokens += num_draft_tokens;
      metrics.constrained_num_accepted_tokens += accepted;
    }
  }
}

// Per-thread counter handles: record_spec_metrics runs on both the brpc
// worker and the shm-polling threads.
bvar::Adder<double>* pos_counter(size_t pos) {
  thread_local std::vector<bvar::Adder<double>*> counters;
  if (counters.size() <= pos) {
    counters.resize(pos + 1, nullptr);
  }
  if (counters[pos] == nullptr) {
    counters[pos] = CHECK_NOTNULL(
        MULTI_COUNTER_speculative_num_accepted_tokens_per_pos.get_stats(
            {std::to_string(pos)}));
  }
  return counters[pos];
}

}  // namespace

SpecMetrics make_spec_metrics(const SpecVerifyStats& stats) {
  SpecMetrics metrics;
  metrics.sequence_stats.resize(stats.requests.size());

  accumulate_verify_rows(stats.requests, metrics);
  return metrics;
}

void record_spec_metrics(const SpecMetrics& metrics) {
  COUNTER_ADD(speculative_num_drafts_total, metrics.sequence_stats.size());
  COUNTER_ADD(speculative_num_draft_tokens_total, metrics.num_draft_tokens);
  COUNTER_ADD(speculative_num_accepted_tokens_total,
              metrics.num_accepted_tokens);
  COUNTER_ADD(speculative_num_accepted_tokens_constrained_total,
              metrics.constrained_num_accepted_tokens);
  COUNTER_ADD(
      speculative_num_accepted_tokens_plain_total,
      metrics.num_accepted_tokens - metrics.constrained_num_accepted_tokens);
  COUNTER_ADD(speculative_num_draft_tokens_constrained_total,
              metrics.constrained_num_draft_tokens);
  COUNTER_ADD(speculative_num_draft_tokens_plain_total,
              metrics.num_draft_tokens - metrics.constrained_num_draft_tokens);
  for (size_t pos = 0; pos < metrics.num_accepted_tokens_per_pos.size();
       ++pos) {
    const int64_t accepted = metrics.num_accepted_tokens_per_pos[pos];
    if (accepted == 0) {
      continue;
    }
    *pos_counter(pos) << accepted;
  }
}

}  // namespace xllm
