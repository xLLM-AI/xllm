/* Copyright 2025-2026 The xLLM Authors.

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
#include <optional>
#include <vector>

#include "framework/block/block.h"

namespace xllm {

struct BeamSourceInfo {
  size_t suffix_start_idx = 0;
  std::vector<int32_t> generated_token_ids;
  std::vector<std::optional<float>> generated_logprobs;
  std::vector<Block> src_blocks;
};

// BeamCandidate structure for beam search sorting
struct BeamCandidate {
  size_t source_index = 0;
  float logprob_sum = 0.0f;
  bool override_last_token = false;
  int32_t last_token_id = 0;
  std::optional<float> last_token_logprob;

  BeamCandidate() = default;

  bool operator<(const BeamCandidate& other) const {
    return logprob_sum > other.logprob_sum;
  }
};

template <typename CandidateType>
class SimpleTopKOptimizer {
 private:
  std::priority_queue<CandidateType> min_heap_;
  size_t k_;

 public:
  explicit SimpleTopKOptimizer(size_t k) : k_(k) {}

  void clear() {
    while (!min_heap_.empty()) {
      min_heap_.pop();
    }
  }

  void insert(const CandidateType& candidate) {
    if (min_heap_.size() < k_) {
      min_heap_.push(candidate);
    } else if (candidate.logprob_sum > min_heap_.top().logprob_sum) {
      min_heap_.pop();
      min_heap_.push(candidate);
    }
  }

  void insert(CandidateType&& candidate) {
    if (min_heap_.size() < k_) {
      min_heap_.push(std::move(candidate));
    } else if (candidate.logprob_sum > min_heap_.top().logprob_sum) {
      min_heap_.pop();
      min_heap_.push(std::move(candidate));
    }
  }

  void insert_batch(const std::vector<CandidateType>& candidates) {
    for (const auto& candidate : candidates) {
      insert(candidate);
    }
  }

  std::vector<CandidateType> getTopK() {
    std::vector<CandidateType> result;
    result.reserve(min_heap_.size());

    while (!min_heap_.empty()) {
      result.emplace_back(
          std::move(const_cast<CandidateType&>(min_heap_.top())));
      min_heap_.pop();
    }

    return result;
  }

  std::vector<CandidateType>&& getTopKMove() {
    std::vector<CandidateType> result;
    result.reserve(min_heap_.size());

    while (!min_heap_.empty()) {
      result.emplace_back(
          std::move(const_cast<CandidateType&>(min_heap_.top())));
      min_heap_.pop();
    }

    return std::move(result);
  }

  std::vector<CandidateType> getTopKSorted() {
    std::vector<CandidateType> result = getTopK();
    std::reverse(result.begin(), result.end());
    return result;
  }

  size_t size() const { return min_heap_.size(); }

  bool empty() const { return min_heap_.empty(); }

  bool worthInserting(float logprob_sum) const {
    return min_heap_.size() < k_ || logprob_sum > min_heap_.top().logprob_sum;
  }

  float getMinLogprob() const {
    return min_heap_.empty() ? -std::numeric_limits<float>::infinity()
                             : min_heap_.top().logprob_sum;
  }
};

using SimpleTopKOptimizerBeamCandidate = SimpleTopKOptimizer<BeamCandidate>;

}  // namespace xllm
