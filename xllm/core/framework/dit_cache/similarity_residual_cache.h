/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <torch/torch.h>

#include <cstdint>
#include <limits>

#include "core/framework/dit_cache/dit_cache_config.h"
#include "core/framework/parallel_state/process_group.h"

namespace xllm {

// Outcome of one cache probe. `hit` requests reuse of the middle-block residual
// recorded by the most recent dense forward; `relative_l1` is the global
// (across the consensus group) relative-L1 distance between the current and
// previous front residuals; `prediction_scale` is the least-squares slope used
// by the optional linear residual extrapolation.
struct SimilarityResidualCacheDecision {
  bool hit = false;
  float relative_l1 = std::numeric_limits<float>::infinity();
  float prediction_scale = 0.0F;
};

// Request-local similarity residual cache backing PolicyType::CACHE_DIT.
//
// This is a small, generic, model-agnostic cache state: a supported DiT model
// creates one per request (never a resident cross-request member), recomputes
// only the front (and optional back) blocks each forward, and either reuses the
// stored middle-block residual on a hit or records a fresh one on a dense
// forward. The hit decision is made identical on every rank by all-reducing the
// raw similarity statistics over `consensus_group` before thresholding, so a
// sequence-parallel shard can never make a rank-local decision.
//
// The state holds no process-group ownership; `consensus_group` is borrowed per
// call and may be null (single rank), in which case the decision is purely
// local.
class SimilarityResidualCacheState final {
 public:
  explicit SimilarityResidualCacheState(const CacheDiTOptions& options);

  // Decide whether `step` may reuse the cached middle residual.
  // `front_residual` is (front-block output - transformer input) over the local
  // row shard; only the first `used_rows` rows (optionally further narrowed to
  // `row_indices`) contribute to the similarity statistic, so padding rows
  // never pollute it.
  SimilarityResidualCacheDecision decide(
      int64_t step,
      const torch::Tensor& front_residual,
      int64_t used_rows,
      ProcessGroup* consensus_group = nullptr,
      const torch::Tensor& row_indices = torch::Tensor());

  // Record the middle-block residual (full_output - front_output) of a dense
  // forward, replacing the previous stack. Steps must strictly increase.
  void record_dense(int64_t step,
                    const torch::Tensor& front_output,
                    const torch::Tensor& full_output);

  // Reconstruct a hit forward's output as front_output + reused residual.
  torch::Tensor apply(int64_t step, const torch::Tensor& front_output) const;

  bool has_stack_residual() const { return stack_residual_.defined(); }
  int64_t dense_forwards() const { return dense_forwards_; }
  int64_t cache_hits() const { return cache_hits_; }
  int64_t similarity_checks() const { return similarity_checks_; }
  int64_t consecutive_hits() const { return consecutive_hits_; }
  int64_t front_blocks() const { return options_.front_blocks; }
  int64_t back_blocks() const { return options_.back_blocks; }

 private:
  CacheDiTOptions options_;
  torch::Tensor previous_previous_front_residual_;
  torch::Tensor previous_front_residual_;
  torch::Tensor previous_stack_residual_;
  torch::Tensor stack_residual_;
  int64_t previous_stack_step_ = -1;
  int64_t stack_step_ = -1;
  float prediction_scale_ = 0.0F;
  int64_t dense_forwards_ = 0;
  int64_t cache_hits_ = 0;
  int64_t similarity_checks_ = 0;
  int64_t consecutive_hits_ = 0;
};

}  // namespace xllm
