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

#include "runtime/adaptive_speculative_controller.h"

#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <utility>

#include "runtime/speculative_profile_registry.h"
#include "util/tensor_helper.h"

namespace xllm {
namespace {

bool is_mtp_algorithm(std::string algorithm) {
  std::transform(
      algorithm.begin(),
      algorithm.end(),
      algorithm.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return algorithm == "mtp";
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
               is_mtp_algorithm(options.speculative_algorithm()) &&
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
    double target_step_time_ms,
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

  std::vector<std::vector<double>> path_probs(
      static_cast<size_t>(batch_size),
      std::vector<double>(static_cast<size_t>(num_speculative_tokens), 0.0));
  std::vector<PruneCandidate> candidates;
  candidates.reserve(static_cast<size_t>(batch_size) *
                     static_cast<size_t>(num_speculative_tokens));
  const double* prob_data = probs.data_ptr<double>();
  for (int32_t seq_id = 0; seq_id < batch_size; ++seq_id) {
    double path_prob = 1.0;
    for (int32_t token_idx = 0; token_idx < num_speculative_tokens;
         ++token_idx) {
      path_prob *= prob_data[seq_id * num_speculative_tokens + token_idx];
      if (!std::isfinite(path_prob)) {
        path_prob = 0.0;
      }
      path_prob = std::clamp(path_prob, 0.0, 1.0);
      path_probs[static_cast<size_t>(seq_id)][static_cast<size_t>(token_idx)] =
          path_prob;
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

  std::vector<int32_t> prefix_lengths(static_cast<size_t>(batch_size), 0);
  double expected_accepted = 0.0;
  double current_score = score_for_pruned_state(batch_size,
                                                expected_accepted,
                                                prefix_lengths,
                                                per_seq_kv_lens,
                                                full_draft_time_ms,
                                                target_step_time_ms);
  const double min_gain = std::max(min_gain_, 0.0);
  for (const PruneCandidate& candidate : candidates) {
    int32_t& prefix_len = prefix_lengths[static_cast<size_t>(candidate.seq_id)];
    if (candidate.prefix_len <= prefix_len) {
      continue;
    }

    double candidate_expected_accepted = expected_accepted;
    const std::vector<double>& seq_path_probs =
        path_probs[static_cast<size_t>(candidate.seq_id)];
    for (int32_t token_idx = prefix_len; token_idx < candidate.prefix_len;
         ++token_idx) {
      candidate_expected_accepted +=
          seq_path_probs[static_cast<size_t>(token_idx)];
    }
    const int32_t old_prefix_len = prefix_len;
    prefix_len = candidate.prefix_len;
    const double next_score =
        score_for_pruned_state(batch_size,
                               candidate_expected_accepted,
                               prefix_lengths,
                               per_seq_kv_lens,
                               full_draft_time_ms,
                               target_step_time_ms);
    if (next_score <= current_score * (1.0 + min_gain)) {
      prefix_len = old_prefix_len;
      continue;
    }

    expected_accepted = candidate_expected_accepted;
    current_score = next_score;
  }

  return prefix_lengths;
}

// score = expected_emitted / (draft_time + validate_time)
// validate_time is estimated per-seq: sum of per-token costs from the profile.
double AdaptiveSpeculativeController::score_for_pruned_state(
    int32_t batch_size,
    double expected_accepted,
    const std::vector<int32_t>& prefix_lengths,
    const std::vector<double>& per_seq_kv_lens,
    double full_draft_time_ms,
    double target_step_time_ms) const {
  const double estimated_time =
      std::max(full_draft_time_ms, 1.0e-6) +
      std::max(estimate_validate_time(prefix_lengths, per_seq_kv_lens),
               target_step_time_ms * 0.1);
  const double expected_emitted =
      static_cast<double>(batch_size) + expected_accepted;
  return expected_emitted / estimated_time;
}

// Per-seq validate time: intercept + Σᵢ (batch_ms + query_token_ms * qᵢ +
// query_prefix_ms * qᵢ * kvᵢ) where qᵢ = prefix_lengths[i] + 1. Coefficients
// from ProfileManager's linear regression.
double AdaptiveSpeculativeController::estimate_validate_time(
    const std::vector<int32_t>& prefix_lengths,
    const std::vector<double>& per_seq_kv_lens) const {
  std::optional<SpeculativeProfileRegistry::ValidateTimePredictor> predictor =
      SpeculativeProfileRegistry::get_instance().validate_time_predictor();
  if (!predictor.has_value()) {
    return 1.0;
  }

  double time = predictor->intercept_ms;
  for (size_t i = 0; i < prefix_lengths.size(); ++i) {
    const double q_i = static_cast<double>(prefix_lengths[i] + 1);
    const double kv_i = i < per_seq_kv_lens.size() ? per_seq_kv_lens[i] : 0.0;
    time += predictor->batch_ms;
    time += predictor->query_token_ms * q_i;
    time += predictor->query_prefix_ms * q_i * kv_i;
  }
  return std::max(time, 0.0);
}

}  // namespace xllm
