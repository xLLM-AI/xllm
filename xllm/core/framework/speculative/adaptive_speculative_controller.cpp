/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/framework/speculative/adaptive_speculative_controller.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <string>

#include "core/framework/config/speculative_config.h"
#include "core/framework/speculative/speculative_profile_registry.h"
#include "util/tensor_helper.h"

namespace xllm {
namespace {

// Returns true when the adaptive controller can operate on the given
// speculative algorithm name. Currently: MTP, DFlash, DSpark.
bool is_supported_algorithm(const std::string& algorithm) {
  return SpeculativeConfig::is_mtp_algorithm(algorithm) ||
         SpeculativeConfig::is_block_diffusion_algorithm(algorithm);
}

struct PruneCandidate {
  int32_t seq_id = 0;
  int32_t prefix_len = 0;
  double path_prob = 0.0;
};

}  // namespace

AdaptiveSpeculativeController::AdaptiveSpeculativeController(
    const runtime::Options& options)
    : enabled_(options.enable_adaptive_speculative_decode() &&
               options.num_speculative_tokens() > 1 &&
               is_supported_algorithm(options.speculative_algorithm()) &&
               !options.enable_graph()),
      min_gain_(options.adaptive_speculative_min_gain()) {}

bool AdaptiveSpeculativeController::enabled() const { return enabled_; }

// Greedy selection of per-seq validate prefix lengths.
// Candidates (seq_id, position, path_prob) are sorted by path_prob descending.
// Each candidate is accepted if it improves estimated throughput
// (score = expected_tokens / estimated_time).
std::vector<int32_t>
AdaptiveSpeculativeController::select_pruned_prefix_lengths(
    const torch::Tensor& selected_probs_by_step,
    double full_draft_time_ms,
    const std::vector<double>& per_seq_kv_lens) const {
  CHECK(selected_probs_by_step.defined())
      << "adaptive pruning requires draft selected probabilities";
  CHECK_EQ(selected_probs_by_step.dim(), 2)
      << "adaptive pruning expects selected probs [batch, speculative_tokens], "
      << "got " << selected_probs_by_step.sizes();

  torch::Tensor probs =
      safe_to(selected_probs_by_step, torch::kCPU).to(torch::kFloat64);
  probs = probs.clamp(0.0, 1.0).contiguous();
  const int32_t batch_size = static_cast<int32_t>(probs.size(0));
  const int32_t num_speculative_tokens = static_cast<int32_t>(probs.size(1));
  CHECK_GT(batch_size, 0) << "adaptive pruning batch size must be positive";
  CHECK_GT(num_speculative_tokens, 0)
      << "adaptive pruning speculative tokens must be positive";

  // path_probs laid out as a single [batch * num_speculative_tokens] buffer
  // indexed by `seq * num_speculative_tokens + token_idx` to avoid the
  // per-step batch_size heap allocations of vector<vector<double>>.
  std::vector<double> path_probs(
      static_cast<size_t>(batch_size) *
          static_cast<size_t>(num_speculative_tokens),
      0.0);
  std::vector<PruneCandidate> candidates;
  candidates.reserve(static_cast<size_t>(batch_size) *
                     static_cast<size_t>(num_speculative_tokens));
  const double* prob_data = probs.data_ptr<double>();
  for (int32_t seq_id = 0; seq_id < batch_size; ++seq_id) {
    double path_prob = 1.0;
    for (int32_t token_idx = 0; token_idx < num_speculative_tokens;
         ++token_idx) {
      const double step_prob =
          prob_data[seq_id * num_speculative_tokens + token_idx];
      // Chain rule cumulative product: a_{r,j} = ∏ c_i (paper Section 3.2.2
      // Algorithm 1). Works for MTP / DFlash sample-gathered probs and for
      // DSpark ConfidenceHead c_k alike; both are per-step conditional
      // probabilities.
      path_prob *= step_prob;
      if (!std::isfinite(path_prob)) {
        path_prob = 0.0;
      }
      path_prob = std::clamp(path_prob, 0.0, 1.0);
      path_probs[static_cast<size_t>(seq_id) *
                     static_cast<size_t>(num_speculative_tokens) +
                 static_cast<size_t>(token_idx)] = path_prob;
      candidates.push_back({seq_id, token_idx + 1, path_prob});
    }
  }

  std::sort(candidates.begin(),
            candidates.end(),
            [](const PruneCandidate& lhs, const PruneCandidate& rhs) {
              if (lhs.path_prob != rhs.path_prob) {
                return lhs.path_prob > rhs.path_prob;
              }
              if (lhs.prefix_len != rhs.prefix_len) {
                return lhs.prefix_len < rhs.prefix_len;
              }
              return lhs.seq_id < rhs.seq_id;
            });

  // Incremental greedy: maintain running expected_accepted and validate_time
  // instead of recomputing the whole batch per candidate (was O(batch^2 * S)).
  // validate_time = intercept + Σᵢ (query_token_ms·qᵢ +
  // query_prefix_ms·qᵢ·kvᵢ), qᵢ = prefix_len_i + 1. Fetch the predictor once
  // outside the loop.
  std::optional<SpeculativeProfileRegistry::ValidateTimePredictor> predictor =
      SpeculativeProfileRegistry::get_instance().validate_time_predictor();
  const bool has_predictor = predictor.has_value();
  const double query_token_ms = has_predictor ? predictor->query_token_ms : 0.0;
  const double query_prefix_ms =
      has_predictor ? predictor->query_prefix_ms : 0.0;

  const double intercept_ms = has_predictor ? predictor->intercept_ms : 0.0;

  std::vector<int32_t> prefix_lengths(static_cast<size_t>(batch_size), 0);
  double expected_accepted = 0.0;
  // Running validate_time for the current prefix_lengths (qᵢ = 0+1 = 1).
  // We include intercept so that the marginal cost of adding a draft
  // (query_token_ms·Δq + query_prefix_ms·Δq·kv) is weighed against the
  // *full* estimated validate time. Excluding intercept overweights the
  // marginal term and causes over-pruning when intercept dominates.
  double validate_time_raw = 1.0;  // fallback when predictor is unavailable
  if (has_predictor) {
    validate_time_raw = intercept_ms;
    for (int32_t seq_id = 0; seq_id < batch_size; ++seq_id) {
      const double kv_i = static_cast<size_t>(seq_id) < per_seq_kv_lens.size()
                              ? per_seq_kv_lens[static_cast<size_t>(seq_id)]
                              : 0.0;
      validate_time_raw += query_token_ms * 1.0 + query_prefix_ms * 1.0 * kv_i;
    }
  }
  auto score_of = [&](double accepted, double vtime_raw) {
    // Keep the divisor strictly positive; a 1e-6 lower bound is enough.
    const double estimated_time =
        std::max(full_draft_time_ms, 1.0e-6) + std::max(vtime_raw, 1.0e-6);
    return (static_cast<double>(batch_size) + accepted) / estimated_time;
  };
  double current_score = score_of(expected_accepted, validate_time_raw);
  const double min_gain = std::max(min_gain_, 0.0);

  for (const PruneCandidate& candidate : candidates) {
    int32_t& prefix_len = prefix_lengths[static_cast<size_t>(candidate.seq_id)];
    if (candidate.prefix_len <= prefix_len) {
      continue;
    }

    double candidate_expected_accepted = expected_accepted;
    const size_t seq_base = static_cast<size_t>(candidate.seq_id) *
                            static_cast<size_t>(num_speculative_tokens);
    for (int32_t token_idx = prefix_len; token_idx < candidate.prefix_len;
         ++token_idx) {
      candidate_expected_accepted +=
          path_probs[seq_base + static_cast<size_t>(token_idx)];
    }
    // Incremental validate_time: only this seq's qᵢ grows by (new - old).
    const double kv_i =
        static_cast<size_t>(candidate.seq_id) < per_seq_kv_lens.size()
            ? per_seq_kv_lens[static_cast<size_t>(candidate.seq_id)]
            : 0.0;
    const double delta_q =
        static_cast<double>(candidate.prefix_len - prefix_len);
    const double candidate_validate_time_raw =
        has_predictor ? validate_time_raw + query_token_ms * delta_q +
                            query_prefix_ms * delta_q * kv_i
                      : validate_time_raw;
    const double next_score =
        score_of(candidate_expected_accepted, candidate_validate_time_raw);
    const bool accept = (next_score > current_score * (1.0 + min_gain));
    if (!accept) {
      continue;
    }

    prefix_len = candidate.prefix_len;
    expected_accepted = candidate_expected_accepted;
    validate_time_raw = candidate_validate_time_raw;
    current_score = next_score;
  }

  return prefix_lengths;
}

}  // namespace xllm
