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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/metrics.h"
#include "util/tensor_helper.h"

namespace xllm {
namespace {

template <typename TokenT>
void accumulate_verify_rows(const TokenT* tokens,
                            int64_t batch_size,
                            int64_t token_width,
                            const std::vector<VerifyRowLayout>& row_layouts,
                            SpecMetrics& metrics) {
  for (int64_t row = 0; row < batch_size; ++row) {
    const VerifyRowLayout& row_layout = row_layouts[static_cast<size_t>(row)];
    const int64_t num_draft_tokens = row_layout.verify_len;

    SpeculativeTokenStats& row_stats =
        metrics.sequence_stats[static_cast<size_t>(row)];
    row_stats.proposed_tokens = num_draft_tokens;
    metrics.num_draft_tokens += num_draft_tokens;

    const TokenT* row_tokens = tokens + row * token_width;
    if (row_tokens[0] >= 0) {
      ++metrics.num_committed_tokens;
      for (int64_t column = 1; column <= num_draft_tokens; ++column) {
        if (row_tokens[column] < 0) {
          break;
        }
        ++metrics.num_committed_tokens;
        ++row_stats.accepted_tokens;
        ++metrics.num_accepted_tokens;
        ++metrics.num_accepted_tokens_per_pos[static_cast<size_t>(column - 1)];
      }
    }
    // Constrained rows count both their proposals and their acceptances,
    // including fully-rejected rows (accepted_tokens is zero there).
    if (row_layout.json_constrained) {
      metrics.constrained_num_draft_tokens += num_draft_tokens;
      metrics.constrained_num_accepted_tokens += row_stats.accepted_tokens;
    }
  }
}

// Metric handles for one position, resolved once so the per-step recording
// loop does not allocate labels or repeat the locked get_stats lookups.
MultiCounterGaugeHandles& position_handles(size_t position) {
  // Per-thread cache: record_spec_metrics runs on both the brpc worker and the
  // shm-polling threads.
  thread_local std::vector<MultiCounterGaugeHandles> handles;
  if (handles.size() <= position) {
    handles.resize(position + 1);
  }
  MultiCounterGaugeHandles& entry = handles[position];
  if (entry.counter == nullptr) {
    entry.counter =
        MULTI_COUNTER_speculative_num_accepted_tokens_per_pos.get_stats(
            {std::to_string(position)});
  }
  if (entry.gauge == nullptr) {
    entry.gauge =
        MULTI_GAUGE_speculative_conditional_acceptance_rate_per_pos.get_stats(
            {std::to_string(position)});
  }
  return entry;
}

}  // namespace

SpecMetrics make_spec_metrics(const torch::Tensor& tokens,
                              const std::vector<VerifyRowLayout>& row_layouts) {
  CHECK(tokens.defined()) << "speculative output tokens are undefined";
  CHECK_EQ(tokens.dim(), 2) << "speculative output tokens should be 2D";

  const int64_t batch_size = tokens.size(0);
  const int64_t token_width = tokens.size(1);
  CHECK_EQ(row_layouts.size(), static_cast<size_t>(batch_size))
      << "verify row layout batch mismatch";

  // Only the sign of each entry matters, so keep int32/int64 zero-copy and
  // convert any other dtype once. Schedule-overlap MTP and suffix hand the
  // verify tokens to the consumer still on device; the D2H copy here is the
  // single canonical host hand-off for metrics.
  const torch::ScalarType dtype = tokens.scalar_type();
  const torch::ScalarType token_type =
      dtype == torch::kInt32 || dtype == torch::kInt64 ? dtype : torch::kInt64;
  const torch::Tensor int_tokens = to_cpu_contiguous(tokens, token_type);

  SpecMetrics metrics;
  // Size num_accepted_tokens_per_pos to the full verified slot capacity of
  // this output, zero-filling positions past every row's verify_len: those
  // entries keep record_spec_metrics refreshing the already-created position
  // gauges each step instead of leaving them at stale values on pruned
  // batches.
  metrics.num_accepted_tokens_per_pos.resize(
      static_cast<size_t>(token_width - 1));
  metrics.sequence_stats.resize(static_cast<size_t>(batch_size));

  if (token_type == torch::kInt32) {
    accumulate_verify_rows(int_tokens.const_data_ptr<int32_t>(),
                           batch_size,
                           token_width,
                           row_layouts,
                           metrics);
  } else {
    accumulate_verify_rows(int_tokens.const_data_ptr<int64_t>(),
                           batch_size,
                           token_width,
                           row_layouts,
                           metrics);
  }
  return metrics;
}

void record_spec_metrics(const SpecMetrics& metrics) {
  const size_t width = metrics.num_accepted_tokens_per_pos.size();
  COUNTER_ADD(speculative_num_drafts_total, metrics.sequence_stats.size());
  COUNTER_ADD(speculative_num_draft_tokens_total, metrics.num_draft_tokens);
  COUNTER_ADD(speculative_num_accepted_tokens_total,
              metrics.num_accepted_tokens);
  COUNTER_ADD(speculative_num_committed_tokens_total,
              metrics.num_committed_tokens);
  COUNTER_ADD(speculative_num_accepted_tokens_constrained_total,
              metrics.constrained_num_accepted_tokens);
  COUNTER_ADD(
      speculative_num_accepted_tokens_plain_total,
      metrics.num_accepted_tokens - metrics.constrained_num_accepted_tokens);
  COUNTER_ADD(speculative_num_draft_tokens_constrained_total,
              metrics.constrained_num_draft_tokens);
  COUNTER_ADD(speculative_num_draft_tokens_plain_total,
              metrics.num_draft_tokens - metrics.constrained_num_draft_tokens);

  // Derive gauges from the process-wide counters, not per-instance values, so
  // multi-DP writers converge on one aggregate.
  const double drafts_total = COUNTER_VALUE(speculative_num_drafts_total);
  if (drafts_total > 0) {
    GAUGE_SET(
        speculative_mean_acceptance_length,
        COUNTER_VALUE(speculative_num_committed_tokens_total) / drafts_total);
  }
  // Chain-form conditional acceptance rate = P(accept position i | position
  // i-1 accepted) = accepted[i] / accepted[i-1]; position 0 divides by the
  // total draft count. Every position is refreshed each step — including
  // zero-accepted ones — so a position whose acceptance decays to zero
  // converges on the gauge instead of freezing at its last non-zero value.
  double prev_accepted = drafts_total;
  for (size_t position = 0; position < width; ++position) {
    const int64_t accepted = metrics.num_accepted_tokens_per_pos[position];
    MultiCounterGaugeHandles& entry = position_handles(position);
    if (entry.counter == nullptr) {
      continue;
    }
    *entry.counter << accepted;
    if (prev_accepted > 0) {
      const double accepted_total = entry.counter->get_value();
      if (entry.gauge != nullptr) {
        entry.gauge->set_value(accepted_total / prev_accepted);
      }
      prev_accepted = accepted_total;
    }
  }
}

}  // namespace xllm
